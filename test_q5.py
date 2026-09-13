#!/usr/bin/env python3
"""Espera a que el worker descargue/cargue Q5_K_M y luego mide lo unico
que importa: si encadena los pasos del navegador en UNA sola tool_call.
"""
import json, os, time, tomllib, requests

cfg = tomllib.load(open(os.path.expanduser("~/.runpod/config.toml"), "rb"))
KEY = cfg.get("apikey") or cfg["default"]["api_key"]
URL = "https://api.runpod.ai/v2/wztjitidybz1ts/runsync"
H = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}

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


def post(payload, timeout=900):
    r = requests.post(URL, headers=H, json=payload, timeout=timeout)
    return r.status_code, (r.json() if r.status_code == 200 else r.text[:300])


def wait_ready(max_wait=3600):
    """Sondea hasta que el modelo cargue (la descarga de 18GB tarda)."""
    t0 = time.time()
    last = ""
    while time.time() - t0 < max_wait:
        st, d = post({"input": {"diag": True}}, timeout=300)
        if st == 200:
            out = d.get("output") or {}
            if isinstance(out, list):
                out = out[0] if out else {}
            diag = out.get("diag") or out
            phase = str(diag.get("phase") or diag)[:120]
            if phase != last:
                print(f"[{int(time.time()-t0):5d}s] {phase}", flush=True)
                last = phase
            if "ready" in phase.lower():
                return True
        time.sleep(20)
    return False


def probe():
    msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": TAREA}]
    t = time.time()
    st, d = post({"input": {"messages": msgs, "tools": TOOLS,
                            "tool_choice": "auto", "max_tokens": 700,
                            "temperature": 0}})
    if st != 200:
        print("ERROR", st, d)
        return
    out = d.get("output") or {}
    if isinstance(out, list):
        out = out[0] if out else {}
    ch = (out.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    tc = msg.get("tool_calls") or []
    print(f"\n{'='*64}\nQ5_K_M — primera tool_call ({time.time()-t:.1f}s)\n{'='*64}")
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
    print("esperando carga de Q5_K_M (descarga ~18GB la primera vez)...", flush=True)
    if wait_ready():
        probe()
    else:
        print("TIMEOUT esperando ready")
