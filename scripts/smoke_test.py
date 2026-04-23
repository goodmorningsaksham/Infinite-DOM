"""Smoke test — start server in subprocess, hit endpoints, shut down."""
import subprocess
import sys
import time

import httpx


def main():
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "infinite_dom.server.app:app",
         "--host", "127.0.0.1", "--port", "8001"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**__import__("os").environ, "PYTHONPATH": "."},
    )
    try:
        for _ in range(30):
            try:
                r = httpx.get("http://127.0.0.1:8001/health", timeout=1)
                if r.status_code == 200:
                    break
            except Exception:
                time.sleep(0.5)
        else:
            raise RuntimeError("server did not start")
        print("[OK] /health responded")

        r = httpx.post(
            "http://127.0.0.1:8001/reset",
            json={"task_id": 1, "seed": 42},
            timeout=90,
        )
        r.raise_for_status()
        obs = r.json()
        print(f"[OK] /reset returned instruction={obs.get('task_instruction', '')!r}")

        r = httpx.get("http://127.0.0.1:8001/", timeout=5)
        r.raise_for_status()
        assert "Infinite DOM" in r.text
        print("[OK] / (dashboard) responded")

        print("\n[SMOKE TEST PASSED]")
    finally:
        proc.terminate()
        proc.wait(timeout=5)


if __name__ == "__main__":
    main()
