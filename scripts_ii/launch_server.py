"""Launch the futudiffu FastAPI inference server.

Thin orchestration script: parses arguments, creates the GPU backend,
creates the FastAPI app, and runs it via uvicorn. No algorithm logic.

Usage:
    python launch_server.py                          # uses model_paths.py defaults
    python launch_server.py --port 8000              # override port only
    python launch_server.py --fp8-diff /other/model  # override one path
"""

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main():
    from src_ii.model_paths import FP8_PATH, TE_PATH, VAE_PATH, TOKENIZER_PATH

    parser = argparse.ArgumentParser(
        description="futudiffu inference server (FastAPI/HTTP)")
    parser.add_argument("--port", type=int, default=8000,
                        help="HTTP port (default 8000)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind address (default 0.0.0.0)")
    parser.add_argument("--fp8-diff", default=FP8_PATH,
                        help=f"FP8 diffusion model (default: model_paths.py)")
    parser.add_argument("--te", default=TE_PATH,
                        help=f"Text encoder (default: model_paths.py)")
    parser.add_argument("--vae", default=VAE_PATH,
                        help=f"VAE (default: model_paths.py)")
    parser.add_argument("--tokenizer", default=TOKENIZER_PATH,
                        help="Tokenizer directory (default: model_paths.py)")
    parser.add_argument("--device", default="cuda",
                        help="Device (default cuda)")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="Working dtype (default bfloat16)")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="Per-request timeout in seconds (default 600)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of uvicorn workers (default 1, must be 1 for GPU)")
    args = parser.parse_args()

    if args.workers != 1:
        print("WARNING: workers > 1 is not supported for GPU-owning servers. "
              "Each worker would try to load the model independently. Using workers=1.",
              file=sys.stderr)
        args.workers = 1

    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    from src_ii.server import GPUModelBackend, create_app

    print(f"Creating GPU backend:")
    print(f"  FP8 diff:  {args.fp8_diff}")
    print(f"  TE:        {args.te}")
    print(f"  VAE:       {args.vae}")
    print(f"  Tokenizer: {args.tokenizer}")
    print(f"  Device:    {args.device}, dtype: {args.dtype}")
    print(f"  Timeout:   {args.timeout}s per request")

    backend = GPUModelBackend(
        fp8_diff_path=args.fp8_diff,
        te_path=args.te,
        vae_path=args.vae,
        tokenizer_path=args.tokenizer,
        device=args.device,
        dtype=args.dtype,
    )

    app = create_app(backend, request_timeout_s=args.timeout)

    import uvicorn
    print(f"\nStarting server on http://{args.host}:{args.port}")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
