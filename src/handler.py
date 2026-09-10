"""
RunPod Serverless handler — Qwen3.8-27B-OBLITERATED V3 IQ4_XS vía llama.cpp.
Todo vive en el network volume: binarios llama.cpp + modelo GGUF (cacheado).
Levanta llama-server (OpenAI-compatible) y proxea peticiones /v1/*.
"""
import os
import subprocess
import sys
import time

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import requests
import runpod

VOL = os.environ.get("RPVOL", "/runpod-volume/qwen27b")
MODEL_DIR = os.path.join(VOL, "models")
LLAMA_DIR = os.path.join(VOL, "llama")
MODEL_REPO = os.environ.get("MODEL_REPO", "OBLITERATUS/Qwen3.8-27B-OBLITERATED")
MODEL_FILE = os.environ.get("MODEL_FILE", "Qwen3.8-27B-OBLITERATED-IQ4_XS.gguf")
LLAMA_PORT = int(os.environ.get("LLAMA_PORT", "8080"))
CTX_SIZE = int(os.environ.get("CTX_SIZE", "8192"))
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "99"))
PARALLEL = int(os.environ.get("PARALLEL", "1"))
BASE_URL = f"http://127.0.0.1:{LLAMA_PORT}"


def ensure_model() -> str:
    """Descarga el GGUF al volume si no está (queda cacheado para los siguientes boots)."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, MODEL_FILE)
    if os.path.isfile(path) and os.path.getsize(path) > 10 * 1024**3:
        print(f"Modelo cacheado: {path} ({os.path.getsize(path)/1024**3:.1f} GB)", flush=True)
        return path
    from huggingface_hub import hf_hub_download
    print(f"Descargando {MODEL_REPO}/{MODEL_FILE} -> {MODEL_DIR}", flush=True)
    p = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE, local_dir=MODEL_DIR)
    print(f"Descarga completa: {p} ({os.path.getsize(p)/1024**3:.1f} GB)", flush=True)
    return p


def ensure_llama_server() -> str:
    """llama-server viene en la imagen base (ghcr ggml-org/llama.cpp:server-cuda) o en el volume."""
    for cand in ["/app/llama-server",
                 os.path.join(LLAMA_DIR, "build", "bin", "llama-server"),
                 os.path.join(LLAMA_DIR, "llama-server"),
                 "/usr/local/bin/llama-server"]:
        if os.path.isfile(cand):
            return cand
    raise RuntimeError(f"llama-server no encontrado en {LLAMA_DIR}; correr setup_llama.sh en el pod dev")


def start_llama_server(model_path: str):
    bin_path = ensure_llama_server()
    bin_dir = os.path.dirname(bin_path)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = f"{bin_dir}:" + env.get("LD_LIBRARY_PATH", "")
    cmd = [
        bin_path, "-m", model_path,
        "--port", str(LLAMA_PORT),
        "-ngl", str(N_GPU_LAYERS),
        "-c", str(CTX_SIZE),
        "--parallel", str(PARALLEL),
        "--host", "127.0.0.1",
        "--jinja",
        "--no-webui",
    ]
    print("Arrancando:", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    deadline = time.time() + 900
    while time.time() < deadline:
        if proc.poll() is not None:
            out = proc.stdout.read().decode() if proc.stdout else ""
            raise RuntimeError(f"llama-server salió con {proc.returncode}:\n{out[-4000:]}")
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200 and r.json().get("status") == "ok":
                print("llama-server listo", flush=True)
                return proc
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(1)
    proc.kill()
    raise RuntimeError("llama-server no estuvo healthy en 900s")


model_path = ensure_model()
server_proc = start_llama_server(model_path)


def _forward(job_input):
    endpoint = job_input.pop("endpoint", "/v1/chat/completions")
    stream = bool(job_input.get("stream", False))
    resp = requests.post(f"{BASE_URL}{endpoint}", json=job_input, stream=stream, timeout=900)
    resp.raise_for_status()
    return resp, stream


def handler(job):
    resp, stream = _forward(dict(job["input"]))
    if stream:
        return "".join(chunk.decode(errors="ignore") for chunk in resp.iter_content(8192))
    return resp.json()


def generator(job):
    ji = dict(job["input"])
    ji["stream"] = True
    resp, _ = _forward(ji)
    for line in resp.iter_lines():
        if line:
            yield line.decode(errors="ignore")


runpod.serverless.start({"handler": handler, "generator": generator})
