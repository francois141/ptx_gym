"""Flask web server exposing the serializable Triton PTX API over HTTP.

Run it with::

    python -m triton_ptx.server --host 0.0.0.0 --port 5000
"""

from __future__ import annotations

import argparse
from typing import Any

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from triton_ptx.api import (
    dump_kernel_ptx,
    evaluate_candidate,
    get_kernel_data,
    is_gpu_available,
    list_kernels,
)


class BadRequest(Exception):
    """Raised for malformed requests; mapped to HTTP 400."""


def _body() -> dict[str, Any]:
    """Return the JSON request body as a dict, or raise ``BadRequest``."""
    data = request.get_json(silent=True)
    if data is None:
        raise BadRequest("Request body must be a JSON object.")
    if not isinstance(data, dict):
        raise BadRequest("Request body must be a JSON object.")
    return data


def _required(data: dict[str, Any], key: str) -> Any:
    """Return ``data[key]`` or raise ``BadRequest`` if it is missing."""
    if key not in data:
        raise BadRequest(f'Missing required field "{key}".')
    return data[key]


def create_app() -> Flask:
    """Build and return the Flask application (no side effects)."""
    app = Flask(__name__)

    @app.errorhandler(BadRequest)
    def _handle_bad_request(exc: BadRequest):
        return jsonify(error=str(exc)), 400

    @app.errorhandler(ValueError)
    def _handle_value_error(exc: ValueError):
        # api.py raises ValueError for unknown kernel ids and malformed payloads.
        return jsonify(error=str(exc)), 400

    @app.errorhandler(HTTPException)
    def _handle_http_exception(exc: HTTPException):
        # Preserve routing errors (404, 405, ...) as JSON with their own status.
        return jsonify(error=exc.description), exc.code or 500

    @app.errorhandler(Exception)
    def _handle_unexpected(exc: Exception):
        return jsonify(error=str(exc)), 500

    @app.get("/health")
    def health():
        return jsonify(status="ok")

    @app.post("/list_kernels")
    def _list_kernels():
        return jsonify(kernels=list_kernels())

    @app.post("/is_gpu_available")
    def _is_gpu_available():
        return jsonify(gpu_available=is_gpu_available())

    @app.post("/dump_kernel_ptx")
    def _dump_kernel_ptx():
        kernel_id = _required(_body(), "kernel_id")
        return jsonify(ptx=dump_kernel_ptx(kernel_id))

    @app.post("/get_kernel_data")
    def _get_kernel_data():
        kernel_id = _required(_body(), "kernel_id")
        return jsonify(get_kernel_data(kernel_id))

    @app.post("/evaluate_candidate")
    def _evaluate_candidate():
        data = _body()
        kernel_id = _required(data, "kernel_id")
        payload = _required(data, "payload")
        if not isinstance(payload, dict):
            raise BadRequest('Field "payload" must be a JSON object.')
        return jsonify(evaluate_candidate(kernel_id, payload))

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address.")
    parser.add_argument("--port", type=int, default=5000, help="Bind port.")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode.")
    args = parser.parse_args()

    create_app().run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
