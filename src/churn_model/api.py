from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .config import DEMO_DIR, MODEL_PATH
from .predict import load_artifact, predict_records


class ChurnAPIHandler(BaseHTTPRequestHandler):
    artifact_path = MODEL_PATH
    artifact: dict[str, Any] | None = None

    server_version = "TelecomChurnAPI/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _headers(self, status: HTTPStatus, content_type: str = "application/json") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _json_response(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        self._headers(status)
        self.wfile.write(json.dumps(payload, indent=2).encode("utf-8"))

    def _read_json(self) -> Any:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        raw_body = self.rfile.read(length).decode("utf-8")
        return json.loads(raw_body)

    def _serve_demo(self) -> None:
        demo_path = DEMO_DIR / "index.html"
        if not demo_path.exists():
            self._json_response({"error": f"Demo file not found: {demo_path}"}, HTTPStatus.NOT_FOUND)
            return
        self._headers(HTTPStatus.OK, "text/html; charset=utf-8")
        self.wfile.write(demo_path.read_bytes())

    def do_OPTIONS(self) -> None:
        self._headers(HTTPStatus.NO_CONTENT)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        artifact = self.__class__.artifact

        if path in {"/", "/demo", "/index.html"}:
            self._serve_demo()
            return
        if path == "/health":
            self._json_response({"status": "ok", "model_loaded": artifact is not None})
            return
        if artifact is None:
            self._json_response(
                {"error": f"Model artifact not found: {self.__class__.artifact_path}. Run `python train.py`."},
                HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return
        if path == "/schema":
            self._json_response(
                {
                    "numeric_columns": artifact["numeric_columns"],
                    "categorical_columns": artifact["categorical_columns"],
                    "category_levels": artifact["preprocessor"]["category_levels"],
                    "threshold": artifact["threshold"],
                }
            )
            return
        if path == "/metrics":
            self._json_response(
                {
                    "metrics": artifact["metrics"],
                    "baseline": artifact["baseline"],
                    "threshold": artifact["threshold"],
                    "data": artifact["data"],
                }
            )
            return
        self._json_response({"error": f"Unknown endpoint: {path}"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/predict":
            self._json_response({"error": f"Unknown endpoint: {path}"}, HTTPStatus.NOT_FOUND)
            return

        artifact = self.__class__.artifact
        if artifact is None:
            self._json_response(
                {"error": f"Model artifact not found: {self.__class__.artifact_path}. Run `python train.py`."},
                HTTPStatus.SERVICE_UNAVAILABLE,
            )
            return

        try:
            payload = self._read_json()
            if isinstance(payload, list):
                records = payload
            elif "customers" in payload:
                records = payload["customers"]
            elif "customer" in payload:
                records = [payload["customer"]]
            else:
                records = [payload]
            predictions = predict_records(records, artifact=artifact)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            self._json_response({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        response: dict[str, Any] = {"predictions": predictions, "count": len(predictions)}
        if len(predictions) == 1:
            response.update(predictions[0])
        self._json_response(response)


def build_server(host: str, port: int, artifact_path: Path) -> ThreadingHTTPServer:
    ChurnAPIHandler.artifact_path = artifact_path
    ChurnAPIHandler.artifact = load_artifact(artifact_path) if artifact_path.exists() else None
    return ThreadingHTTPServer((host, port), ChurnAPIHandler)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the telecom churn prediction API and demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    args = parser.parse_args(argv)

    server = build_server(args.host, args.port, args.model)
    print(f"Serving churn API and demo at http://{args.host}:{args.port}")
    print("Endpoints: GET /health, GET /metrics, GET /schema, POST /predict")
    server.serve_forever()


if __name__ == "__main__":
    main()

