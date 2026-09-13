#!/usr/bin/env python3
"""Compara el MISMO modelo con thinking ON vs OFF en una tarea de tools
multi-paso (la que falla en guru: buscar en el navegador Y LEER la pagina).

Golpea el endpoint de RunPod DIRECTO para poder controlar si se inyecta
/no_think o no (el proxy siempre lo inyecta).
"""
import json, os, time, tomllib, requests

cfg = tomllib.load(open(os.path.expanduser("~/.runpod/config.toml"), "rb"))
KEY = cfg.get("apikey") or cfg["default"]["api_key"]
EP = "wztjitidybz1ts"
URL = f"https://api.runpod.ai/v2/{EP}/runsync"
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}

TOOLS = [
    {"type": "function", "function": {
        "name": "browser_exec",
        "description": "Ejecuta codigo Python con helpers de navegador: new_tab(url), wait_for_load(), js(expr) para leer el DOM, page_info().",
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string", "description": "Codigo Python a ejecutar"}},
            "required": ["code"]}}}
]

TAREA = ("Abre el navegador y busca que contiene la bebida energetica "
         "Enerchill de Cafenio. Necesito los ingredientes reales.")

SYS = ("Eres un asistente con acceso a un navegador real. Usa las herramientas "
       "para obtener informacion de paginas web.")


def run(label, no_think):
    sys_msg = SYS + (" /no_think" if no_think else "")
    payload = {"input": {
        "messages": [{"role": "system", "content": sys_msg},
                     {"role": "user", "content": TAREA}],
        "tools": TOOLS, "tool_choice": "auto",
        "max_tokens": 700, "temperature": 0}}
    t = time.time()
    r = requests.post(URL, headers=H, json=payload, timeout=600)
    dt = time.time() - t
    d = r.json()
    out = d.get("output") or {}
    if isinstance(out, list):
        out = out[0] if out else {}
    ch = (out.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    tc = msg.get("tool_calls") or []
    content = msg.get("content") or ""

    print(f"\n{'='*62}\n{label}   ({dt:.1f}s, finish={ch.get('finish_reason')})\n{'='*62}")
    if tc:
        code = ""
        try:
            code = json.loads(tc[0]["function"]["arguments"]).get("code", "")
        except Exception:
            code = str(tc[0])
        print("TOOL_CALL -> browser_exec, codigo enviado:")
        print("-" * 62)
        print(code[:900])
        print("-" * 62)
        # Lo que decide si la tarea sirve: leer la pagina, no solo abrirla
        lee = any(k in code for k in ("innerText", "js(", "page_info", "textContent"))
        espera = "wait_for_load" in code
        print(f"  abre pestana : {'new_tab' in code or 'goto_url' in code}")
        print(f"  espera carga : {espera}")
        print(f"  LEE la pagina: {lee}   <-- lo que guru NUNCA hace")
        print(f"  VEREDICTO    : {'SIRVE' if lee else 'INUTIL (abre y no lee)'}")
    else:
        print("SIN tool_calls. Texto:", repr(content[:400]))


if __name__ == "__main__":
    run("A) CON /no_think  (configuracion actual de guru)", True)
    run("B) SIN /no_think  (thinking habilitado)", False)
