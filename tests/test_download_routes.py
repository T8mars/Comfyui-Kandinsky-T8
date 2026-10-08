"""Use actual aiohttp requests to check cancellation owns the download lock."""
import asyncio
import importlib.util
from pathlib import Path
import threading

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
import pytest

spec = importlib.util.spec_from_file_location("k6_download_routes", Path(__file__).parents[1] / "model_downloads.py")
downloads = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloads)


@pytest.mark.parametrize("worker_fails", [False, True])
def test_repeated_handler_cancel_keeps_lock_until_worker_finishes(tmp_path, monkeypatch, worker_fails):
    started, finish = threading.Event(), threading.Event()
    workers = []
    handlers = []

    def worker(*args):
        workers.append("start")
        started.set()
        assert finish.wait(5)
        workers.append("finish")
        if worker_fails and len(workers) == 2:
            raise OSError("write failed after cancellation")
        return "reused"

    monkeypatch.setattr(downloads, "model_plan", lambda *args: [{"directory": "vae", "name": "test.bin"}])
    monkeypatch.setattr(downloads, "download_model", worker)

    class Routes(web.RouteTableDef):
        def post(self, path, **kwargs):
            decorate = super().post(path, **kwargs)

            def capture(function):
                async def route(request):
                    handlers.append(asyncio.current_task())
                    return await function(request)
                return decorate(route)
            return capture

    async def run():
        routes = Routes()
        downloads.register_routes(routes, "test", tmp_path, tmp_path, {})
        app = web.Application()
        app.add_routes(routes)
        client = TestClient(TestServer(app))
        await client.start_server()
        first = None
        try:
            first = await client.post("/test/download-models", json={"confirm": True})
            assert first.status == 200
            await first.content.readline()
            assert await asyncio.to_thread(started.wait, 2)
            owner = handlers[0]
            for _ in range(2):
                owner.cancel()
                await asyncio.sleep(0.01)
                conflicting = await client.post("/test/download-models", json={"confirm": True})
                assert conflicting.status == 409
                await conflicting.read()
            assert workers == ["start"]
            finish.set()
            outcome = await asyncio.gather(owner, return_exceptions=True)
            assert isinstance(outcome[0], asyncio.CancelledError)
            resumed = await client.post("/test/download-models", json={"confirm": True})
            assert resumed.status == 200
            assert b'"status": "complete"' in await resumed.read()
            assert workers == ["start", "finish", "start", "finish"]
        finally:
            finish.set()
            if first is not None:
                first.close()
            await client.close()

    asyncio.run(run())
