#!/usr/bin/env python3
"""MISMO modelo, MISMA tarea: OrcaRouter vs mi RunPod.

Prueba las DOS cosas que importan:
  1) primera tool_call
  2) si al devolverle el resultado CONTINUA (lee la pagina) o se atora

El paso 2 es el que falla en guru: repite new_tab una y otra vez.
"""
import json, os, time, tomllib, requests, re

# --- credenciales (nunca se imprimen) ---
def load_env(path):
    d = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                d[k.strip()] = v.strip().strip('"').strip("'")
    return d

ENV = load_env("/home/chain1/.hermes/profiles/guru/.env")
ORCA_KEY = ENV.get("ORCAROUTER_API_KEY")
cfg = tomllib.load(open(os.path.expanduser("~/.runpod/config.toml"), "rb"))
RP_KEY = cfg.get("apikey") or cfg["default"]["api_key"]
RP_URL = "https://api.runpod.ai/v2/wztjitidybz1ts/runsync"

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


def extract(code):
    lee = any(k in code for k in ("innerText", "textContent", "js(", "page_info"))
    return lee


def call_orca(messages):
    r = requests.post("https://api.orcarouter.ai/v1/chat/completions",
                      headers={"Authorization": f"Bearer {ORCA_KEY}",
                               "Content-Type": "application/json"},
                      json={"model": "obsidian/Qwen3.8-27B", "messages": messages,
                            "tools": TOOLS, "tool_choice": "auto",
                            "max_tokens": 700, "temperature": 0}, timeout=300)
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}: {r.text[:200]}"
    return (r.json().get("choices") or [{}])[0].get("message") or {}, None


def call_runpod(messages):
    r = requests.post(RP_URL, headers={"Authorization": f"Bearer {RP_KEY}",
                                       "Content-Type": "application/json"},
                      json={"input": {"messages": messages, "tools": TOOLS,
                                      "tool_choice": "auto", "max_tokens": 700,
                                      "temperature": 0}}, timeout=600)
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}: {r.text[:200]}"
    out = r.json().get("output") or {}
    if isinstance(out, list):
        out = out[0] if out else {}
    return (out.get("choices") or [{}])[0].get("message") or {}, None


def prueba(label, fn):
    print(f"\n{'='*64}\n{label}\n{'='*64}")
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": TAREA}]
    t = time.time()
    msg, err = fn(msgs)
    if err:
        print("  ERROR:", err)
        return
    tc = msg.get("tool_calls") or []
    if not tc:
        print("  PASO 1: SIN tool_calls. Texto:", repr((msg.get("content") or "")[:200]))
        return
    code1 = json.loads(tc[0]["function"]["arguments"]).get("code", "")
    print(f"  PASO 1 ({time.time()-t:.1f}s): {code1[:150]}")
    print(f"          lee pagina: {extract(code1)}")

    # Le devolvemos un resultado REALISTA y vemos si continua
    msgs.append({"role": "assistant", "content": msg.get("content") or "",
                 "tool_calls": tc})
    msgs.append({"role": "tool", "tool_call_id": tc[0].get("id", "call_1"),
                 "content": json.dumps({"success": True, "output": "ok\n"})})
    t = time.time()
    msg2, err = fn(msgs)
    if err:
        print("  PASO 2 ERROR:", err)
        return
    tc2 = msg2.get("tool_calls") or []
    if tc2:
        code2 = json.loads(tc2[0]["function"]["arguments"]).get("code", "")
        lee2 = extract(code2)
        print(f"  PASO 2 ({time.time()-t:.1f}s): {code2[:150]}")
        print(f"          lee pagina: {lee2}")
        repite = code2.strip() == code1.strip()
        print(f"  VEREDICTO: {'SE ATORA (repite lo mismo)' if repite else ('AVANZA Y LEE' if lee2 else 'avanza pero no lee')}")
    else:
        print(f"  PASO 2 ({time.time()-t:.1f}s): sin tool_call. Texto:",
              repr((msg2.get("content") or "")[:200]))
        print("  VEREDICTO: SE RINDE (contesta sin leer la pagina)")


if __name__ == "__main__":
    if ORCA_KEY:
        prueba("ORCAROUTER  obsidian/qwen3.8-27b", call_orca)
    else:
        print("sin ORCAROUTER_API_KEY")
    prueba("MI RUNPOD   Qwen3.8-27B-OBLITERATED IQ4_XS", call_runpod)
