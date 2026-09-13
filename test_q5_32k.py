#!/usr/bin/env python3
"""Prueba Q5_K_M (32K) con UNA sola peticion asincrona y sondeo por job id.

Clave: NO encolar una peticion nueva cada vez (eso saturaba el endpoint);
se manda un job y se consulta su estado hasta que termina.
"""
import json, os, time, tomllib, requests

cfg = tomllib.load(open(os.path.expanduser("~/.runpod/config.toml"), "rb"))
KEY = cfg.get("apikey") or cfg["default"]["api_key"]
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}
B = "https://api.runpod.ai/v2/wztjitidybz1ts"

TOOLS = [{"type": "function", "function": {
    "name": "browser_exec",
    "description": ("Ejecuta codigo Python con helpers de navegador ya importados: "
                    "new_tab(url) abre y navega, wait_for_load() espera la carga, "
                    "js(expr) evalua JavaScript y DEVUELVE el valor (usalo para leer "
                    "el texto de la pagina), page_info() da url/titulo. "
                    "Usa print() para que la salida te regrese."),
    "parameters": {"type": "object", "properties": {
        "code": {"type": "string"}}, "required": ["code"]}}}]

SYS = ("Eres un asistente con acceso a un navegador real. Usa las herramientas para "
       "obtener informacion de paginas web.")
TAREA = ("Abre el navegador y busca que contiene la bebida energetica Enerchill de "
         "Cafenio. Necesito los ingredientes reales.")


def submit(inp):
    r = requests.post(f"{B}/run", headers=H, json={"input": inp}, timeout=60)
    r.raise_for_status()
    return r.json()["id"]


def poll(jid, max_wait=2700, every=20):
    t0 = time.time()
    last = ""
    while time.time() - t0 < max_wait:
        s = requests.get(f"{B}/status/{jid}", headers=H, timeout=40).json()
        st = s.get("status")
        if st != last:
            print(f"[{int(time.time()-t0):5d}s] {st}", flush=True)
            last = st
        if st == "COMPLETED":
            return s.get("output")
        if st == "FAILED":
            print("FAILED:", json.dumps(s)[:800], flush=True)
            return None
        time.sleep(every)
    print("TIMEOUT", flush=True)
    return None


def main():
    print("== 1) diag: esperando que cargue Q5_K_M 32K ==", flush=True)
    out = poll(submit({"diag": True}))
    if out:
        d = out[0] if isinstance(out, list) else out
        d = d.get("diag") or d
        print("  phase :", d.get("phase"))
        print("  gpu   :", d.get("gpu"))
        print("  modelo:", d.get("model_size_gb"), "GB")
        print("  tpl tool_call:", d.get("server_template_has_tool_call"))
        if d.get("error"):
            print("  ERROR :", str(d.get("error"))[:400])
            print("  detail:", str(d.get("detail"))[:600])
            return

    print("\n== 2) prueba de encadenado de tools ==", flush=True)
    out = poll(submit({"messages": [{"role": "system", "content": SYS},
                                    {"role": "user", "content": TAREA}],
                       "tools": TOOLS, "tool_choice": "auto",
                       "max_tokens": 700, "temperature": 0}))
    if not out:
        return
    o = out[0] if isinstance(out, list) else out
    ch = (o.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    tc = msg.get("tool_calls") or []
    print("=" * 64)
    if not tc:
        print("SIN tool_calls:", repr((msg.get("content") or "")[:300]))
        return
    code = json.loads(tc[0]["function"]["arguments"]).get("code", "")
    print(code[:600])
    print("-" * 64)
    lee = any(k in code for k in ("innerText", "textContent", "js(", "page_info"))
    print(f"  abre        : {'new_tab' in code or 'goto_url' in code}")
    print(f"  espera carga: {'wait_for_load' in code}")
    print(f"  LEE pagina  : {lee}")
    print(f"  VEREDICTO   : {'ENCADENA COMO ORCAROUTER' if lee else 'solo abre (igual que IQ4_XS)'}")


if __name__ == "__main__":
    main()
