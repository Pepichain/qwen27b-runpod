#!/usr/bin/env python3
"""
Proxy OpenAI-compatible local (127.0.0.1:8081) -> endpoint RunPod serverless
de Qwen3.8-27B-OBLITERATED.

Hermes (perfil sams) habla contra este proxy como si fuera un servidor
llama.cpp local; el proxy traduce a las llamadas run/status del job queue
de RunPod (el endpoint real, que escala a 0 y no corre localmente).

Soporta streaming (SSE) porque los clientes OpenAI-compatible (Hermes
Desktop incluido) suelen pedir stream=true y esperan text/event-stream;
como RunPod no da streaming real, se simula: se espera la respuesta
completa y se emite como UN SOLO delta chunk + [DONE].

Uso:
    python3 proxy_server.py            # foreground
    nohup python3 proxy_server.py > proxy.log 2>&1 &   # background

Expone:
    GET  /v1/models
    POST /v1/chat/completions   (soporta stream=true y stream=false/ausente)
"""
import json
import os
import re
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen_runpod as qr

PORT = int(os.environ.get("PROXY_PORT", "8081"))
MODEL_NAME = os.environ.get("MODEL_NAME", "qwen3.8-27b-obliterated")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[proxy] {self.address_string()} - {fmt % args}", flush=True)

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/v1/models"):
            self._send_json(200, {"object": "list", "data": [
                {"id": MODEL_NAME, "object": "model", "owned_by": "runpod"}]})
        elif self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/v1/chat/completions"):
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b"{}"
            req = json.loads(raw)
        except Exception as e:
            self._send_json(400, {"error": f"bad request: {e}"})
            return

        messages = req.get("messages") or []
        max_tokens = req.get("max_tokens") or req.get("max_completion_tokens") or 1800
        temperature = req.get("temperature", 0.25)
        want_stream = bool(req.get("stream"))

        # Reenvia TODO lo demas tal cual (tools, tool_choice, response_format,
        # top_p, stop, seed, presence_penalty, etc.) — antes solo se pasaban
        # 2-3 campos fijos y se perdian tools/tool_choice, lo que dejaba al
        # modelo sin saber que podia usar herramientas en tareas agenticas
        # complejas (Hermes) y a veces devolvia content vacio.
        SKIP = {"messages", "max_tokens", "max_completion_tokens", "temperature",
                "stream", "model", "chat_template_kwargs"}
        kw = {k: v for k, v in req.items() if k not in SKIP and v is not None}
        ctk = req.get("chat_template_kwargs")
        if ctk is None:
            ctk = {"enable_thinking": False}
        kw["chat_template_kwargs"] = ctk

        # La plantilla Hermes tool_use NO respeta enable_thinking, asi que Qwen
        # sigue gastando tokens (y segundos) razonando en <think> antes de cada
        # respuesta. Se corta por prompt con la instruccion /no_think en el
        # system. NO usar stop=["<think>"]: el modelo abre el bloque en el primer
        # token, el stop lo corta ahi mismo y la respuesta llega VACIA.
        # NO_THINK=0 desactiva esta inyeccion.
        if os.environ.get("NO_THINK", "1") == "1":
            messages = list(messages)
            marker = "/no_think"
            if messages and messages[0].get("role") == "system":
                if marker not in (messages[0].get("content") or ""):
                    messages[0] = {**messages[0],
                                   "content": f"{messages[0].get('content','')}\n\n{marker}"}
            else:
                messages.insert(0, {"role": "system", "content": marker})

        t0 = time.time()
        try:
            out = qr.raw_chat(messages, max_tokens=max_tokens, temperature=temperature,
                               timeout=900, **kw)
        except Exception as e:
            self._send_json(502, {"error": f"runpod: {type(e).__name__}: {e}"})
            return

        if isinstance(out, dict) and out.get("error"):
            self._send_json(502, {"error": out["error"], "diag": out.get("diag")})
            return

        if not (isinstance(out, dict) and out.get("choices")):
            self._send_json(502, {"error": "respuesta inesperada de runpod", "raw": out})
            return

        msg = out["choices"][0]["message"]
        content = msg.get("content") or ""
        raw_content = content
        # Quitar el bloque de razonamiento <think>...</think> que Qwen antepone:
        # ensucia el texto visible y la plantilla Hermes no respeta enable_thinking.
        content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL)
        if "<think>" in content and "</think>" not in content:
            # Thinking sin cerrar = la generacion se corto por max_tokens dentro
            # del razonamiento. Si se filtra queda vacio y el cliente ve "sin
            # respuesta"; mejor devolver el razonamiento como texto que nada.
            head, _, tail = content.partition("<think>")
            content = head if head.strip() else tail
        content = content.lstrip("\n")
        if not content.strip() and not msg.get("tool_calls"):
            # Nunca devolver content vacio sin tool_calls: el cliente lo reporta
            # como "empty content". Conservar el original aunque traiga <think>.
            content = raw_content
        msg["content"] = content
        tool_calls = msg.get("tool_calls")
        finish_reason = out["choices"][0].get("finish_reason", "stop")
        cid = out.get("id") or f"chatcmpl-{uuid.uuid4().hex}"
        created = out.get("created") or int(time.time())

        if want_stream:
            self._stream_response(cid, created, content, tool_calls, finish_reason)
        else:
            out.setdefault("model", MODEL_NAME)
            self._send_json(200, out)
        print(f"[proxy] chat.completions OK en {time.time()-t0:.1f}s "
              f"(stream={want_stream})", flush=True)

    def _stream_response(self, cid, created, content, tool_calls, finish_reason):
        """Emula SSE: RunPod no da streaming real, así que se manda la
        respuesta completa como un solo delta chunk seguido de [DONE].
        Es suficiente para que clientes OpenAI-compatible (Hermes incluido)
        muestren la respuesta en vez de quedarse esperando bytes de stream."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        # 'close' y NO keep-alive: el cuerpo SSE no lleva Content-Length ni
        # chunked encoding, asi que el cliente solo sabe que terminamos cuando
        # se cierra la conexion. Con keep-alive el cliente se queda esperando
        # bytes para siempre aunque ya hayamos mandado [DONE] (se veia como
        # "180s sin output" en Hermes aunque el log del proxy decia 200 OK).
        self.send_header("Connection", "close")
        self.end_headers()

        def _chunk(delta, finish=None):
            obj = {
                "id": cid, "object": "chat.completion.chunk", "created": created,
                "model": MODEL_NAME,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            data = f"data: {json.dumps(obj)}\n\n".encode("utf-8")
            self.wfile.write(data)
            self.wfile.flush()

        try:
            delta = {"role": "assistant", "content": content}
            if tool_calls:
                # Los clientes OpenAI-compatible esperan 'index' en cada
                # tool_call dentro de un delta de streaming; sin el, algunos
                # ignoran la llamada a herramienta por completo.
                delta["tool_calls"] = [
                    {**tc, "index": tc.get("index", i)} for i, tc in enumerate(tool_calls)
                ]
            _chunk(delta)
            _chunk({}, finish=finish_reason)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            # NO cerrar wfile a mano: BaseHTTPRequestHandler vuelve a hacer
            # flush() al terminar y reventaria con "I/O operation on closed
            # file". Marcar close_connection basta: el server cierra el socket
            # al salir del handler, que es la senal de fin de cuerpo SSE.
            self.close_connection = True
        except (BrokenPipeError, ConnectionResetError):
            print("[proxy] cliente cerro conexion antes de terminar el stream", flush=True)


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"[proxy] escuchando en http://127.0.0.1:{PORT} -> RunPod {qr.ENDPOINT_ID}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
