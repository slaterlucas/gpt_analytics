#!/usr/bin/env python3
import argparse, os, subprocess, sys, venv, pathlib, shutil

ROOT = pathlib.Path(__file__).resolve().parent.parent
VENV = ROOT / ".venv"
PIP  = VENV / ("Scripts/pip.exe" if os.name == "nt" else "bin/pip")
PY   = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

WEB = ROOT / "web"

def ensure_venv():
    if VENV.exists(): return
    print("📦  Creating virtual-env …")
    venv.create(VENV, with_pip=True)
    subprocess.check_call([str(PIP), "install", "-r", "api/requirements.txt"])

def ensure_web_deps():
    """Install npm packages in web/ if node_modules is missing"""
    node_modules_dir = WEB / "node_modules"
    if node_modules_dir.exists():
        return
    print("📦  Installing frontend dependencies …")
    subprocess.check_call(["npm", "install"], cwd=str(WEB))
    print("✅  Frontend dependencies installed")

def serve():
    ensure_venv()
    ensure_web_deps()
    os.chdir(ROOT)
    cmd = [
        "npx", "concurrently", "-k",
        f"{PY} -m uvicorn api.app.main:app --reload --port 8000",
        "cd web && npm run dev"
    ]
    subprocess.call(cmd)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--install", action="store_true")
    p.add_argument("--serve",   action="store_true")
    args = p.parse_args()
    if args.install: ensure_venv()
    if args.serve:   serve() 