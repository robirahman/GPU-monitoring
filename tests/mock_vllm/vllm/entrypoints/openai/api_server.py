"""Test double for ``python -m vllm.entrypoints.openai.api_server`` (NOT vLLM).

Serves /v1/models and /v1/chat/completions so orchestrators (llm_moe_serving.py,
disagg_kv_vllm.py) can be exercised end to end without a GPU. Put tests/mock_vllm on PYTHONPATH.
Understands --port, ignores every other vLLM flag, and echoes NIXL kv_transfer_params so the
disagg proxy flow works against it too.
"""
import http.server
import json
import sys


def _arg(name, default=None):
    argv = sys.argv
    return argv[argv.index(name) + 1] if name in argv else default


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._json({"data": [{"id": "mock"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        ktp = body.get("kv_transfer_params") or {}
        out = {"usage": {"prompt_tokens": 100, "completion_tokens": int(body.get("max_tokens") or 1)}}
        if ktp.get("do_remote_decode"):                      # prefiller role
            out["kv_transfer_params"] = {"remote_engine_id": "mock", "remote_block_ids": [list(range(7))],
                                         "remote_host": "localhost", "remote_port": 5600}
        self._json(out)


if __name__ == "__main__":
    http.server.ThreadingHTTPServer(("127.0.0.1", int(_arg("--port", "8000"))), H).serve_forever()
