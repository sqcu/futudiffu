"""Launch the client_yeetums BFF server.

Thin orchestration script: parses arguments, creates the FastAPI app, and
runs it via uvicorn. No GPU, no torch, no model logic.

Usage:
    python launch_yeetums.py --inference-url http://localhost:8000 --port 8001
"""

import argparse
import logging
import os
import sys

# Add project root to sys.path so src_ii is importable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main():
    parser = argparse.ArgumentParser(
        description="client_yeetums: diegetic web UI for futudiffu")
    parser.add_argument("--port", type=int, default=8001,
                        help="HTTP port for the UI (default 8001)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind address (default 0.0.0.0)")
    parser.add_argument("--inference-url", default="http://localhost:8000",
                        help="URL of the GPU inference server (default http://localhost:8000)")
    parser.add_argument("--gallery-dir", default="yeetums_gallery",
                        help="Directory for gallery images (default yeetums_gallery)")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="Request timeout for inference server calls (default 600s)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    from src_ii.client_yeetums import create_app

    print(f"client_yeetums BFF:")
    print(f"  Inference server: {args.inference_url}")
    print(f"  Gallery dir:      {args.gallery_dir}")
    print(f"  Timeout:          {args.timeout}s")
    print()

    app = create_app(
        inference_url=args.inference_url,
        gallery_dir=args.gallery_dir,
        timeout_s=args.timeout,
    )

    import uvicorn
    print(f"Starting yeetums on http://{args.host}:{args.port}")
    print(f"Open http://localhost:{args.port} in your browser")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
