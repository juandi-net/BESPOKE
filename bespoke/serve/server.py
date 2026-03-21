"""Manage llama.cpp server with LoRA adapter hot-swap."""

import subprocess
import time
import requests
from pathlib import Path
from typing import Optional

from bespoke.config import config


def start_server(
    adapter_path: Optional[Path] = None,
    port: int = None,
) -> subprocess.Popen:
    """Start llama-server with the base model and optional LoRA adapter.

    Returns the subprocess handle.
    """
    if port is None:
        port = config.base_model.llama_server_port

    cmd = [
        "llama-server",
        "--model", str(config.base_model.inference_model_path),
        "--port", str(port),
        "--n-gpu-layers", str(config.base_model.gpu_layers),
        "--ctx-size", str(config.base_model.context_size),
    ]

    if adapter_path:
        cmd.extend(["--lora", str(adapter_path)])

    print(f"Starting llama-server on port {port}...")
    print(f"  Model: {config.base_model.inference_model_path}")
    if adapter_path:
        print(f"  Adapter: {adapter_path}")

    proc = subprocess.Popen(cmd)

    # Wait for server to be ready
    try:
        for i in range(30):
            try:
                r = requests.get(f"http://localhost:{port}/health", timeout=1)
                if r.status_code == 200:
                    print(f"Server ready on port {port}")
                    return proc
            except requests.ConnectionError:
                pass
            time.sleep(1)
    except BaseException:
        proc.kill()
        proc.wait()
        raise

    proc.kill()
    proc.wait()
    raise RuntimeError("Server failed to start within 30 seconds")


def main():
    """CLI entry point for serving."""
    import argparse

    parser = argparse.ArgumentParser(description="BESPOKE: Serve specialist model")
    parser.add_argument("--adapter", type=Path, help="Path to GGUF adapter file")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    args = parser.parse_args()

    proc = start_server(adapter_path=args.adapter, port=args.port)

    print(f"\nServer running. API endpoint: http://localhost:{args.port}/v1")
    print("Press Ctrl+C to stop.\n")

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
