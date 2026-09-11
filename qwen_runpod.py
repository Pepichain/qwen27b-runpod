#!/usr/bin/env python3
"""
Cliente OpenAI-compatible para el endpoint RunPod de Qwen3.8-27B-OBLITERATED.

Uso rápido:
    python3 qwen_runpod.py "tu pregunta"

Como librería:
    from qwen_runpod import chat
    print(chat("hola"))

Detalles del endpoint:
    endpoint_id : wztjitidybz1ts
    modelo      : Qwen3.8-27B-OBLITERATED (IQ4_XS, sin censura)
    GPU         : RTX 4090 / 3090 / A6000 (fallback automático)
    escala a 0  : sí (no cobra si no lo usas; idle 5 min)
    velocidad   : ~51 tokens/s
    cold start  : ~3-4 min la primera vez del día (descarga+carga), ~1 min si el
                  worker sigue caliente
"""
import json
import os
import re
import sys
import time

import requests

ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "wztjitidybz1ts")
BASE = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"


def _api_key():
    k = os.environ.get("RUNPOD_API_KEY")
    if k:
        return k
    cfg = os.path.expanduser("~/.runpod/config.toml")
    if os.path.isfile(cfg):
        m = re.search(r"api_key\s*=\s*['\"]([^'\"]+)", open(cfg).read())
        if m:
            return m.group(1)
        m = re.search(r"apikey\s*=\s*['\"]([^'\"]+)", open(cfg).read())
        if m:
            return m.group(1)
    raise RuntimeError("falta RUNPOD_API_KEY o ~/.runpod/config.toml")


HEADERS = {"Authorization": f"Bearer {_api_key()}", "Content-Type": "application/json"}


def raw_chat(messages, max_tokens=512, temperature=0.7, timeout=900, **kw):
    """Manda una conversación y devuelve la respuesta completa (formato OpenAI)."""
    payload = {"messages": messages, "max_tokens": max_tokens,
               "temperature": temperature, **kw}
    r = requests.post(f"{BASE}/run", headers=HEADERS, json={"input": payload}, timeout=60)
    r.raise_for_status()
    jid = r.json()["id"]
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(3)
        s = requests.get(f"{BASE}/status/{jid}", headers=HEADERS, timeout=30).json()
        st = s.get("status")
        if st == "COMPLETED":
            return s["output"]
        if st in ("FAILED", "CANCELLED", "TIMED_OUT"):
            raise RuntimeError(f"job {st}: {json.dumps(s)[:500]}")
    raise TimeoutError(f"sin respuesta en {timeout}s")


def chat(prompt, system=None, **kw):
    """Atajo: manda un prompt de texto y devuelve solo el texto de respuesta."""
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": prompt}]
    out = raw_chat(msgs, **kw)
    if isinstance(out, dict) and out.get("choices"):
        return out["choices"][0]["message"]["content"].strip()
    return out


def status():
    """Estado del endpoint: workers y cola."""
    return requests.get(f"{BASE}/health", headers=HEADERS, timeout=30).json()


def warmup():
    """Despierta el worker (útil antes de una sesión de trabajo)."""
    return raw_chat([{"role": "user", "content": "ok"}], max_tokens=1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        print(json.dumps(status(), indent=2))
    elif len(sys.argv) > 1:
        print(chat(" ".join(sys.argv[1:])))
    else:
        print(__doc__)
