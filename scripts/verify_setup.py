"""Verify the complete BESPOKE environment is ready."""

import sys
from pathlib import Path


def check(name: str, condition: bool, detail: str = ""):
    status = "\u2713" if condition else "\u2717"
    msg = f"  {status} {name}"
    if detail:
        msg += f" \u2014 {detail}"
    print(msg)
    return condition


def main():
    print("BESPOKE Environment Verification")
    print("=" * 50)
    all_ok = True

    # Python version
    v = sys.version_info
    all_ok &= check("Python 3.11+", v.major == 3 and v.minor >= 11, f"{v.major}.{v.minor}.{v.micro}")

    # Core imports
    for mod in ["sqlite3", "sqlite_vec", "openai", "onnxruntime", "yaml", "mlx", "mlx_lm", "peft", "transformers", "watchdog"]:
        try:
            __import__(mod)
            all_ok &= check(f"Import {mod}", True)
        except ImportError as e:
            all_ok &= check(f"Import {mod}", False, str(e))

    # Database
    db_path = Path.home() / ".bespoke" / "bespoke.db"
    all_ok &= check("Database exists", db_path.exists(), str(db_path))

    # Models
    gguf = Path.home() / ".bespoke" / "models" / "qwen3.5-4b-gguf" / "Qwen3.5-4B-Q4_K_M.gguf"
    all_ok &= check("Base model (GGUF)", gguf.exists(), f"{gguf.stat().st_size / 1e9:.2f} GB" if gguf.exists() else "MISSING")

    # Embedding model (ONNX)
    embed = Path.home() / ".bespoke" / "models" / "embeddinggemma-300m" / "onnx" / "model.onnx"
    all_ok &= check("Embedding model (EmbeddingGemma 300M ONNX)", embed.exists())

    # llama.cpp
    import shutil
    all_ok &= check("llama-server", shutil.which("llama-server") is not None)

    # Directories
    for d in ["adapters", "scorecards", "benchmark", "backups"]:
        p = Path.home() / ".bespoke" / d
        all_ok &= check(f"Directory ~/.bespoke/{d}", p.exists())

    print("=" * 50)
    if all_ok:
        print("All checks passed. Ready to build.")
    else:
        print("Some checks failed. Fix issues above before proceeding.")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
