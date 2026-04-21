"""Local hot Qwen service.

Keeps Qwen-VL loaded so extraction requests can run without repeated model
startup latency.
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image

from src.extractor import ModelManager, QwenExtractor
from src.qwen_bbox_grounder import QwenBBoxGrounder


HOST = "127.0.0.1"
PORT = 8765


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("QwenHotServer")


shared_manager = ModelManager()
extractor = QwenExtractor(shared_manager)
grounder = QwenBBoxGrounder(model_manager=shared_manager)


def load_image(path_str: str) -> Image.Image:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(path)
    return Image.open(path).convert("RGB")


class Handler(BaseHTTPRequestHandler):
    def _json_response(self, status: int, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json_response(200, {"status": "ok", "model": "qwen-hot"})
            return
        self._json_response(404, {"error": "not_found"})

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            self._json_response(400, {"error": f"bad_json: {exc}"})
            return

        try:
            if self.path == "/extract_crop":
                image = load_image(payload["image_path"])
                field_name = str(payload.get("field_name") or "field")
                result = extractor.extract_field_fallback(image, field_name)
                self._json_response(200, result)
                return

            if self.path == "/ground_field":
                image = load_image(payload["image_path"])
                bbox, confidence = grounder.ground_field(
                    image,
                    field_name=str(payload.get("field_name") or "field"),
                    field_description=str(payload.get("field_description") or payload.get("field_name") or "field"),
                    renderer=str(payload.get("renderer") or "text"),
                    page_num=payload.get("page_num"),
                )
                self._json_response(200, {"bbox": list(bbox), "confidence": confidence})
                return

            self._json_response(404, {"error": "not_found"})
        except Exception as exc:
            logger.exception("Request failed")
            self._json_response(500, {"error": str(exc)})

    def log_message(self, format, *args):
        logger.info("%s - %s", self.address_string(), format % args)


def main():
    logger.info("Preloading Qwen model...")
    shared_manager.get_qwen()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    logger.info("Qwen hot server listening on http://%s:%s", HOST, PORT)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        shared_manager.unload_all()


if __name__ == "__main__":
    main()
