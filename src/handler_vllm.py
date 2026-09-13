"""RunPod Serverless — Qwen3.8-27B-OBLITERATED W4A16 vía vLLM.

Motivo del cambio frente a llama.cpp: la arquitectura Qwen3_5 (atencion
hibrida GatedDeltaNet) no funciona en llama.cpp con su plantilla nativa, y la
plantilla generica que hacia falta para tool-calling le quitaba la capacidad de
encadenar varios pasos. vLLM soporta Qwen3_5 de forma nativa (kernels de Flash
Linear Attention) y trae parser de tool-calls de Qwen, asi que conserva la
plantilla original del modelo.

El handler arranca vLLM en background y hace de proxy OpenAI, igual que el de
llama.cpp: el worker toma jobs desde el primer segundo y reporta fase de carga.
"""
import os
import shutil
import subprocess
import threading
import time

import requests
import runpod

VOL = os.environ.get("RPVOL", "/runpod-volume/qwen27b")
CACHE_DIR = os.path.join(VOL, "hf")
MODEL_REPO = os.environ.get("MODEL_REPO",
                            "quocbao747/Qwen3.8-27B-OBLITERATED-W4A16-24GB")
CTX_SIZE = int(os.environ.get("CTX_SIZE", "32768"))
GPU_UTIL = float(os.environ.get("GPU_UTIL", "0.92"))
PORT = int(os.environ.get("VLLM_PORT", "8000"))
BASE_URL = f"http://127.0.0.1:{PORT}"
SERVED_NAME = os.environ.get("SERVED_NAME", "qwen3.8-27b-obliterated")

STATE = {"phase": "cold", "detail": "", "error": None, "cmd": None}
_LOCK = threading.Lock()
_STARTED = False


def _log(msg):
    STATE["detail"] = (STATE["detail"] + "\n" + str(msg))[-4000:]
    print("[handler]", msg, flush=True)


def _alive():
    try:
        return requests.get(f"{BASE_URL}/health", timeout=5).status_code == 200
    except Exception:
        return False


def _launch():
    """Arranca vLLM una sola vez, en un hilo aparte."""
    global _STARTED
    with _LOCK:
        if _STARTED:
            return
        _STARTED = True
    threading.Thread(target=_serve, daemon=True).start()


def _serve():
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        env = dict(os.environ)
        # El volumen persiste los pesos entre workers: sin esto cada arranque
        # en frio volveria a bajar 18 GB.
        env["HF_HOME"] = CACHE_DIR
        env["HUGGINGFACE_HUB_CACHE"] = CACHE_DIR
        env["VLLM_LOGGING_LEVEL"] = "INFO"

        binp = shutil.which("vllm") or "vllm"
        cmd = [binp, "serve", MODEL_REPO,
               "--host", "127.0.0.1", "--port", str(PORT),
               "--served-model-name", SERVED_NAME,
               "--max-model-len", str(CTX_SIZE),
               "--gpu-memory-utilization", str(GPU_UTIL),
               "--download-dir", CACHE_DIR,
               # Conserva la plantilla del modelo y activa el parser de Qwen:
               # esto es justo lo que llama.cpp no podia hacer.
               "--enable-auto-tool-choice",
               "--tool-call-parser", os.environ.get("TOOL_PARSER", "hermes"),
               "--trust-remote-code"]
        if os.environ.get("REASONING_PARSER"):
            cmd += ["--reasoning-parser", os.environ["REASONING_PARSER"]]

        STATE["phase"] = "loading"
        STATE["cmd"] = " ".join(cmd)
        _log("arrancando: " + " ".join(cmd))

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, env=env)

        def _drain():
            for line in iter(proc.stdout.readline, b""):
                s = line.decode(errors="replace").rstrip()
                _log(s[-300:]) if ("Error" in s or "error" in s) else None
                print("[vllm]", s, flush=True)

        threading.Thread(target=_drain, daemon=True).start()

        # La primera vez baja ~18 GB al volumen; despues es solo carga en GPU.
        deadline = time.time() + int(os.environ.get("BOOT_TIMEOUT", "2700"))
        while time.time() < deadline:
            if _alive():
                STATE["phase"] = "ready"
                _log("vLLM listo")
                return
            if proc.poll() is not None:
                STATE["phase"] = "error"
                STATE["error"] = f"vllm murio con codigo {proc.returncode}"
                _log(STATE["error"])
                return
            time.sleep(5)
        STATE["phase"] = "error"
        STATE["error"] = "timeout esperando a vLLM"
    except Exception as e:
        STATE["phase"] = "error"
        STATE["error"] = f"{type(e).__name__}: {e}"
        _log(STATE["error"])


def _diag():
    info = {"phase": STATE["phase"], "detail": STATE["detail"][-800:],
            "error": STATE["error"], "cmd": STATE.get("cmd"),
            "model_repo": MODEL_REPO, "ctx": CTX_SIZE}
    try:
        r = requests.get(f"{BASE_URL}/v1/models", timeout=5)
        info["models"] = r.json() if r.status_code == 200 else r.status_code
    except Exception as e:
        info["models"] = f"err {type(e).__name__}"
    try:
        info["gpu"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
             "--format=csv,noheader"], capture_output=True, text=True,
            timeout=15).stdout.strip()
    except Exception as e:
        info["gpu"] = f"err {e}"
    info["vol_exists"] = os.path.isdir("/runpod-volume")
    return info


def handler(job):
    inp = job.get("input") or {}
    _launch()

    if inp.get("diag"):
        return _diag()

    if STATE["phase"] == "error":
        return {"error": STATE["error"], "diag": _diag()}

    if not _alive():
        return {"error": "modelo aun cargando", "diag": _diag()}

    inp.pop("diag", None)
    payload = dict(inp)
    payload.setdefault("model", SERVED_NAME)
    # vLLM responde en el formato OpenAI que el proxy local ya entiende.
    payload["stream"] = False
    try:
        r = requests.post(f"{BASE_URL}/v1/chat/completions", json=payload,
                          timeout=int(os.environ.get("GEN_TIMEOUT", "600")))
        if r.status_code != 200:
            return {"error": f"vllm HTTP {r.status_code}", "body": r.text[:600]}
        return r.json()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "diag": _diag()}


runpod.serverless.start({"handler": handler})
