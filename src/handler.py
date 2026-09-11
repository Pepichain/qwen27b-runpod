"""
RunPod Serverless — Qwen3.8-27B-OBLITERATED IQ4_XS vía llama.cpp.

Diseño a prueba de fallos: el loop de RunPod arranca PRIMERO (así el worker
siempre toma jobs y puede reportar errores), y el modelo se carga en background.
Si algo falla, el error vuelve en la respuesta del job en vez de morir en silencio.
"""
import os
import shutil
import subprocess
import threading
import time
import traceback

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import requests
import runpod

VOL = os.environ.get("RPVOL", "/runpod-volume/qwen27b")
# Escribir en disco local del contenedor es MUCHO más rápido que el network volume.
# Si hay volume, se usa como caché persistente: se copia ahí al terminar.
LOCAL_DIR = os.environ.get("LOCAL_MODEL_DIR", "/models")
MODEL_DIR = LOCAL_DIR
CACHE_DIR = os.path.join(VOL, "models")
MODEL_REPO = os.environ.get("MODEL_REPO", "OBLITERATUS/Qwen3.8-27B-OBLITERATED")
MODEL_FILE = os.environ.get("MODEL_FILE", "Qwen3.8-27B-OBLITERATED-IQ4_XS.gguf")
LLAMA_PORT = int(os.environ.get("LLAMA_PORT", "8080"))
CTX_SIZE = int(os.environ.get("CTX_SIZE", "8192"))
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "99"))
PARALLEL = int(os.environ.get("PARALLEL", "1"))
BASE_URL = f"http://127.0.0.1:{LLAMA_PORT}"

STATE = {"phase": "starting", "detail": "", "error": None, "model_path": None}
_lock = threading.Lock()
_boot_started = False


def _log(msg):
    print(f"[qwen] {msg}", flush=True)


def _find_llama_server():
    for cand in ["/app/llama-server", "/usr/local/bin/llama-server",
                 os.path.join(VOL, "llama", "bin", "llama-server")]:
        if os.path.isfile(cand):
            return cand
    found = shutil.which("llama-server")
    if found:
        return found
    raise RuntimeError("llama-server no encontrado en la imagen")


def _download_model():
    """Descarga directa por HTTP con reanudación, a disco LOCAL (rápido).
    Si el volume ya tiene el modelo cacheado, lo usa desde ahí sin descargar."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, MODEL_FILE)
    if os.path.isfile(path) and os.path.getsize(path) > 10 * 1024**3:
        STATE["detail"] = f"local {os.path.getsize(path)/1024**3:.1f} GB"
        return path

    # ¿está cacheado en el network volume de corridas anteriores?
    cached = os.path.join(CACHE_DIR, MODEL_FILE)
    if os.path.isfile(cached) and os.path.getsize(cached) > 10 * 1024**3:
        STATE["phase"] = "cache-hit"
        STATE["detail"] = f"usando caché del volume: {os.path.getsize(cached)/1024**3:.1f} GB"
        _log("modelo encontrado en el network volume")
        return cached

    url = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{MODEL_FILE}?download=true"
    tmp = path + ".part"
    STATE["phase"] = "downloading"
    total = 0
    for attempt in range(60):
        have = os.path.getsize(tmp) if os.path.isfile(tmp) else 0
        headers = {"User-Agent": "Mozilla/5.0"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as r:
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code} en descarga")
                total = have + int(r.headers.get("Content-Length", 0))
                STATE["detail"] = f"{have/1024**3:.1f}/{total/1024**3:.1f} GB (intento {attempt+1})"
                mode = "ab" if have and r.status_code == 206 else "wb"
                if mode == "wb":
                    have = 0
                last = time.time()
                with open(tmp, mode) as f:
                    for chunk in r.iter_content(8 * 1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        have += len(chunk)
                        if time.time() - last > 15:
                            STATE["detail"] = f"{have/1024**3:.2f}/{total/1024**3:.2f} GB"
                            last = time.time()
            if os.path.getsize(tmp) > 10 * 1024**3:
                os.replace(tmp, path)
                _log(f"descarga OK: {os.path.getsize(path)/1024**3:.2f} GB")
                # copiar al volume en background para futuros cold starts
                def _cache():
                    try:
                        os.makedirs(CACHE_DIR, exist_ok=True)
                        dst = os.path.join(CACHE_DIR, MODEL_FILE)
                        if not (os.path.isfile(dst) and os.path.getsize(dst) > 10 * 1024**3):
                            shutil.copyfile(path, dst + ".part")
                            os.replace(dst + ".part", dst)
                            _log("modelo cacheado en el volume")
                    except Exception as ce:
                        _log(f"cache al volume fallo (no critico): {ce}")
                threading.Thread(target=_cache, daemon=True).start()
                return path
            raise RuntimeError(f"descarga corta: {os.path.getsize(tmp)} bytes")
        except Exception as e:
            _log(f"reintento {attempt+1} tras error: {e}")
            STATE["detail"] = f"reintento {attempt+1}: {e}"
            time.sleep(5)
    raise RuntimeError(f"no se pudo descargar tras 60 intentos: {STATE['detail']}")


def _start_server(model_path):
    binp = _find_llama_server()
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.path.dirname(binp) + ":" + env.get("LD_LIBRARY_PATH", "")
    cmd = [binp, "-m", model_path, "--port", str(LLAMA_PORT), "-ngl", str(N_GPU_LAYERS),
           "-c", str(CTX_SIZE), "--parallel", str(PARALLEL), "--host", "127.0.0.1",
           "--jinja", "--no-webui"]
    STATE["phase"] = "loading"
    STATE["detail"] = " ".join(cmd)
    _log("arrancando: " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    tail = []

    def _drain():
        for line in iter(proc.stdout.readline, b""):
            s = line.decode(errors="replace").rstrip()
            tail.append(s)
            del tail[:-40]
            print("[llama]", s, flush=True)

    threading.Thread(target=_drain, daemon=True).start()

    deadline = time.time() + 1800
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server salió {proc.returncode}: " + "\n".join(tail[-20:]))
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=3)
            if r.status_code == 200 and r.json().get("status") == "ok":
                STATE["phase"] = "ready"
                STATE["detail"] = "llama-server healthy"
                _log("listo")
                return proc
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("timeout esperando llama-server: " + "\n".join(tail[-20:]))


def _boot():
    try:
        mp = _download_model()
        STATE["model_path"] = mp
        _start_server(mp)
    except Exception as e:
        STATE["phase"] = "error"
        STATE["error"] = f"{type(e).__name__}: {e}"
        STATE["detail"] = traceback.format_exc()[-2000:]
        _log("ERROR BOOT: " + STATE["error"])


def _ensure_boot():
    global _boot_started
    with _lock:
        if not _boot_started:
            _boot_started = True
            threading.Thread(target=_boot, daemon=True).start()


def _diag():
    info = {"phase": STATE["phase"], "detail": STATE["detail"][-800:], "error": STATE["error"]}
    try:
        info["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                                      "--format=csv,noheader"], capture_output=True,
                                     text=True, timeout=15).stdout.strip()
    except Exception as e:
        info["gpu"] = f"err {e}"
    info["llama_server"] = next((c for c in ["/app/llama-server", "/usr/local/bin/llama-server"]
                                 if os.path.isfile(c)), shutil.which("llama-server") or "NO ENCONTRADO")
    info["vol_exists"] = os.path.isdir("/runpod-volume")
    try:
        mp = STATE.get("model_path") or os.path.join(MODEL_DIR, MODEL_FILE)
        info["model_size_gb"] = round(os.path.getsize(mp) / 1024**3, 2) if os.path.isfile(mp) else 0
        # progreso real: archivos .incomplete de huggingface_hub
        partials = []
        for root, _dirs, files in os.walk(MODEL_DIR):
            for f in files:
                p = os.path.join(root, f)
                try:
                    sz = os.path.getsize(p)
                except OSError:
                    continue
                if sz > 50 * 1024**2:
                    partials.append({"f": f[-45:], "gb": round(sz / 1024**3, 2)})
        info["partials"] = sorted(partials, key=lambda x: -x["gb"])[:5]
        st = os.statvfs(VOL) if os.path.isdir(VOL) else os.statvfs("/")
        info["free_gb"] = round(st.f_bavail * st.f_frsize / 1024**3, 1)
    except Exception as e:
        info["model_size_gb"] = f"err {e}"
    return info


def handler(job):
    inp = dict(job.get("input") or {})
    if inp.get("diag"):
        _ensure_boot()
        return _diag()

    _ensure_boot()
    wait_s = int(inp.pop("wait_ready", 600))
    deadline = time.time() + wait_s
    while STATE["phase"] != "ready" and time.time() < deadline:
        if STATE["phase"] == "error":
            return {"error": STATE["error"], "trace": STATE["detail"][-1200:], "diag": _diag()}
        time.sleep(3)
    if STATE["phase"] != "ready":
        return {"error": "modelo aún cargando", "diag": _diag()}

    endpoint = inp.pop("endpoint", "/v1/chat/completions")
    inp.pop("diag", None)
    try:
        resp = requests.post(f"{BASE_URL}{endpoint}", json=inp, timeout=900)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "diag": _diag()}


_ensure_boot()
runpod.serverless.start({"handler": handler})
