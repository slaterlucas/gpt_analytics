#!/usr/bin/env python3
import argparse, os, subprocess, sys, venv, pathlib, shutil

ROOT = pathlib.Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
PIP  = VENV / ("Scripts/pip.exe" if os.name == "nt" else "bin/pip")
PY   = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

def ensure_venv():
    if VENV.exists(): return
    print("📦  Creating virtual-env …")
    venv.create(VENV, with_pip=True)
    subprocess.check_call([str(PIP), "install", "-r", "api/requirements.txt"])

def serve():
    ensure_venv()
    os.chdir(ROOT)
    cmd = [
        "npx", "concurrently", "-k",
        f"\"{PY}\" -m uvicorn api.app.main:app --reload --port 8000",
        "\"npm\" --workspace web run dev"
    ]
    subprocess.call(" ".join(cmd), shell=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--install", action="store_true")
    p.add_argument("--serve",   action="store_true")
    args = p.parse_args()
    if args.install: ensure_venv()
    if args.serve:   serve() 