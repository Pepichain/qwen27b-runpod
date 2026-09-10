"""Debug handler: devuelve el contenido de archivos del network volume."""
import os
import runpod

VOL = "/runpod-volume/qwen27b"


def handler(job):
    inp = job.get("input") or {}
    fname = inp.get("file", "boot.log")
    path = os.path.join(VOL, fname) if not fname.startswith("/") else fname
    if not os.path.isfile(path):
        # listar lo que hay
        listing = []
        for root, dirs, files in os.walk(VOL):
            for f in files:
                p = os.path.join(root, f)
                listing.append(f"{p} ({os.path.getsize(p)} bytes)")
            if len(listing) > 100:
                break
        return {"error": f"no existe {path}", "listing": listing[:100]}
    tail = int(inp.get("tail_bytes", 20000))
    with open(path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - tail))
        data = f.read().decode(errors="replace")
    return {"file": path, "size": size, "content": data}


runpod.serverless.start({"handler": handler})
