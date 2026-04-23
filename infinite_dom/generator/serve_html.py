"""
Local HTTP server for generated pages.

Pages are registered in memory keyed by generation_id. Playwright
navigates to http://localhost:{port}/page/<generation_id> to access them.
"""
from __future__ import annotations

import logging
import socket

from aiohttp import web

from infinite_dom.config import CONFIG

logger = logging.getLogger(__name__)


def _find_free_port(preferred: int) -> int:
    """Return preferred port if free, otherwise pick a random free port."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", preferred))
            return preferred
    except OSError:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]


class PageServer:
    def __init__(self, port: int = CONFIG.page_server_port):
        self.port = port
        self.pages: dict[str, str] = {}
        self.app = web.Application()
        self.app.router.add_get("/page/{gen_id}", self._handle_page)
        self.app.router.add_get("/results", self._handle_results_placeholder)
        self.app.router.add_get("/confirm", self._handle_results_placeholder)
        self.app.router.add_get("/done", self._handle_results_placeholder)
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None

    def register_page(self, gen_id: str, html: str) -> str:
        """Register HTML under an ID. Returns the URL to navigate to."""
        self.pages[gen_id] = html
        return f"http://localhost:{self.port}/page/{gen_id}"

    def unregister_page(self, gen_id: str) -> None:
        self.pages.pop(gen_id, None)

    async def _handle_page(self, request: web.Request) -> web.Response:
        gen_id = request.match_info["gen_id"]
        html = self.pages.get(gen_id)
        if html is None:
            return web.Response(status=404, text="page not found")
        return web.Response(text=html, content_type="text/html")

    async def _handle_results_placeholder(self, request: web.Request) -> web.Response:
        return web.Response(text="<html><body>Results</body></html>", content_type="text/html")

    async def start(self) -> None:
        self._runner = web.AppRunner(self.app)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, "127.0.0.1", self.port)
        await self._site.start()
        logger.info("Page server started on port %d", self.port)

    async def stop(self) -> None:
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()


_INSTANCE: PageServer | None = None


async def get_page_server() -> PageServer:
    global _INSTANCE
    if _INSTANCE is None:
        port = _find_free_port(CONFIG.page_server_port)
        _INSTANCE = PageServer(port=port)
        await _INSTANCE.start()
    return _INSTANCE


async def stop_page_server() -> None:
    global _INSTANCE
    if _INSTANCE is not None:
        await _INSTANCE.stop()
        _INSTANCE = None
