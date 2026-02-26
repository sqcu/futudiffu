"""client_yeetums: Diegetic web UI for futudiffu inference.

Torch-free BFF (backend-for-frontend) that proxies to the GPU inference server.
"""

from src_ii.client_yeetums.app import create_app

__all__ = ["create_app"]
