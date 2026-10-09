#!/usr/bin/env python3
"""Binary completion and reasoning-swap integration tests (Python stdlib only).

Generate the model with tests/gen-tiny-qwen35-mtp.py, then run:
  python3 tests/test-reasoning-swap.py --server build/bin/llama-server \
      --driver build/bin/llama-reasoning-swap --model tiny-qwen35-mtp.gguf
"""

import argparse
import contextlib
import http.client
import http.server
import json
import pathlib
import socket
import socketserver
import struct
import subprocess
import tempfile
import threading
import time
import unittest


MAGIC = b"LLMTOK01"
ARGS = None


def packet(metadata, tokens):
    encoded = json.dumps(metadata, separators=(",", ":")).encode()
    return MAGIC + struct.pack("<II", len(encoded), len(tokens)) + encoded + struct.pack(f"<{len(tokens)}I", *tokens)


def unpack(body):
    if len(body) < 16 or body[:8] != MAGIC:
        raise ValueError("invalid packet header")
    size, count = struct.unpack_from("<II", body, 8)
    if len(body) != 16 + size + 4 * count:
        raise ValueError("invalid packet size")
    return json.loads(body[16:16 + size]), list(struct.unpack_from(f"<{count}I", body, 16 + size))


class UnixHTTP(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__("localhost", timeout=30)
        self.path = str(path)

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.path)


def request(path, route, data=None, binary=False):
    with contextlib.closing(UnixHTTP(path)) as connection:
        body = data if binary else json.dumps(data).encode() if data is not None else None
        connection.request("POST" if data is not None else "GET", route, body, {
            "Content-Type": "application/octet-stream" if binary else "application/json",
        })
        response = connection.getresponse()
        return response.status, response.read()


@contextlib.contextmanager
def actual_server(mtp):
    with tempfile.TemporaryDirectory(prefix="llama-swap-") as directory:
        path = pathlib.Path(directory) / "test.sock"
        with tempfile.TemporaryFile() as log:
            command = [ARGS.server, "-m", ARGS.model, "--host", str(path), "-ngl", "0",
                       "-c", "512", "-b", "64", "-ub", "64", "-t", "2", "-np", "1",
                       "--flash-attn", "off", "--no-webui", "--no-warmup", "--no-context-shift"]
            if mtp:
                command += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
            process = subprocess.Popen(command, stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline and process.poll() is None:
                    try:
                        if request(path, "/health")[0] == 200:
                            break
                    except (OSError, http.client.HTTPException):
                        pass
                    time.sleep(0.05)
                else:
                    log.seek(0)
                    raise RuntimeError("server failed to start:\n" + log.read().decode(errors="replace"))
                yield path
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


class ProtocolTests(unittest.TestCase):
    def test_completion_and_validation(self):
        for mtp in (False, True):
            with self.subTest(mtp=mtp), actual_server(mtp) as path:
                status, body = request(path, "/tokens/info")
                self.assertEqual(status, 200, body)
                info = json.loads(body)
                self.assertEqual(info["protocol"], MAGIC.decode())
                self.assertEqual(info["n_vocab"], 259, "use gen-tiny-qwen35-mtp.py's fixture")
                self.assertTrue(info["fingerprint"])
                self.assertEqual(info["fingerprint"], json.loads(request(path, "/tokens/info")[1])["fingerprint"])

                prompt = [1, 100, 101, 102]
                options = {"n_predict": 12, "temperature": 0, "seed": 7, "ignore_eos": True,
                           "cache_prompt": False, "return_tokens": True}
                status, body = request(path, "/completion", dict(options, prompt=prompt))
                self.assertEqual(status, 200, body)
                ordinary = json.loads(body)
                status, body = request(path, "/completion/tokens", packet(options, prompt), binary=True)
                self.assertEqual(status, 200, body)
                metadata, tokens = unpack(body)
                self.assertEqual(tokens, ordinary["tokens"])
                self.assertEqual(len(tokens), 12)
                self.assertEqual(metadata["stop_type"], ordinary["stop_type"])
                self.assertEqual(metadata["tokens_evaluated"], len(prompt))
                if mtp:
                    self.assertGreater(metadata["timings"]["draft_n"], 0)

                # Replace an already evaluated reasoning suffix, then compare
                # against a full re-evaluation of the replacement prompt.
                base = [1] + list(range(80, 112))
                cached = dict(options, cache_prompt=True)
                status, body = request(path, "/completion/tokens", packet(cached, base + [41, 42, 43]), binary=True)
                self.assertEqual(status, 200, body)
                swapped_prompt = base + [51, 52, 53, 54]
                status, body = request(path, "/completion/tokens", packet(cached, swapped_prompt), binary=True)
                self.assertEqual(status, 200, body)
                _, swapped_tokens = unpack(body)
                status, body = request(path, "/completion", dict(options, prompt=swapped_prompt))
                self.assertEqual(status, 200, body)
                self.assertEqual(swapped_tokens, json.loads(body)["tokens"])

                # Stop text is removed from content but its original ID must survive.
                stopped = dict(options, stop=["q"], logit_bias=[[3 + ord("q"), 1000]])
                status, body = request(path, "/completion/tokens", packet(stopped, prompt), binary=True)
                self.assertEqual(status, 200, body)
                metadata, tokens = unpack(body)
                self.assertEqual(tokens, [3 + ord("q")])
                self.assertEqual(metadata["content"], "")
                self.assertEqual(metadata["stop_type"], "word")
                self.assertEqual(metadata["stopping_word"], "q")

                # The reasoning boundary is a control token in real models.
                stopped = dict(options, stop=["<s>"], preserved_tokens=["<s>"], logit_bias=[[1, 1000]])
                status, body = request(path, "/completion/tokens", packet(stopped, prompt), binary=True)
                self.assertEqual(status, 200, body)
                metadata, tokens = unpack(body)
                self.assertEqual(tokens, [1])
                self.assertEqual(metadata["content"], "")
                self.assertEqual(metadata["stop_type"], "word")
                self.assertEqual(metadata["stopping_word"], "<s>")

                valid = packet(options, prompt)
                bad_packets = {
                    "short header": MAGIC,
                    "bad magic": b"BADTOK01" + valid[8:],
                    "metadata length": MAGIC + struct.pack("<II", 65537, 0),
                    "token count": valid[:12] + struct.pack("<I", 999) + valid[16:],
                    "trailing byte": valid + b"x",
                    "negative ID": packet(options, [0xffffffff]),
                    "out of range ID": packet(options, [259]),
                    "empty prompt": packet(options, []),
                    "metadata not object": packet([], prompt),
                    "malformed metadata": MAGIC + struct.pack("<II", 1, 1) + b"{" + struct.pack("<I", 1),
                    "context overflow": packet(options, [1] * info["context_size"]),
                }
                for name, fields in {
                    "stream": {"stream": True}, "n": {"n": 2}, "n_cmpl": {"n_cmpl": 2},
                    "token return": {"return_tokens": False}, "probabilities": {"n_probs": 1},
                    "probabilities alias": {"logprobs": 1},
                    "response filtering": {"response_fields": ["content"]},
                    "prompt in metadata": {"prompt": [1]}, "unbounded generation": {"n_predict": -1},
                }.items():
                    bad_packets[name] = packet(dict(options, **fields), prompt)
                for name, body in bad_packets.items():
                    with self.subTest(mtp=mtp, invalid=name):
                        status, response = request(path, "/completion/tokens", body, binary=True)
                        self.assertEqual(status, 400, response)
                self.assertEqual(request(path, "/health")[0], 200)


class FakeServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


class Peer:
    def __init__(self, path, index, incomplete=False, fingerprint="same-vocabulary"):
        self.index = index
        self.base = [1, 11 + index]
        self.reasoning = [21 + index * 10, 22 + index * 10, 99]
        self.calls = []
        self.errors = []
        peer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.respond({"protocol": "LLMTOK01", "n_vocab": 128, "fingerprint": fingerprint,
                              "eog_ids": [2], "context_size": 8192})

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                try:
                    if self.path == "/completion/tokens":
                        metadata, tokens = unpack(body)
                        peer.calls.append((metadata, tokens))
                        thinking = len(peer.calls) == 1
                        result = peer.reasoning[:-1] if incomplete and thinking else peer.reasoning if thinking else [70 + index, 2]
                        output = {"content": f"answer-{index}", "stop_type": "limit" if incomplete and thinking else "word" if thinking else "eos",
                                  "stopping_word": "</think>" if thinking and not incomplete else "", "truncated": False,
                                  "tokens_predicted": len(result), "tokens_evaluated": len(tokens), "tokens_cached": 0,
                                  "id_slot": 0, "timings": {"prompt_ms": 1, "predicted_ms": 1}}
                        self.respond(packet(output, result))
                    elif self.path == "/apply-template":
                        self.respond({"prompt": f"user-{index}\nassistant\n<think>\n"})
                    elif self.path == "/tokenize":
                        data = json.loads(body)
                        self.respond({"tokens": [99] if data["content"] == "</think>" else peer.base})
                    else:
                        raise ValueError("unexpected route: " + self.path)
                except Exception as error:
                    peer.errors.append(str(error))
                    self.respond({"error": str(error)}, 500)

            def respond(self, body, status=200):
                binary = isinstance(body, bytes)
                body = body if binary else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/octet-stream" if binary else "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self.server = FakeServer(str(path), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


class DriverTests(unittest.TestCase):
    def run_driver(self, incomplete=False, mismatch=False):
        with tempfile.TemporaryDirectory(prefix="llama-peers-") as directory:
            paths = [pathlib.Path(directory) / f"{index}.sock" for index in range(2)]
            peers = [Peer(paths[0], 0), Peer(paths[1], 1, incomplete, "different" if mismatch else "same-vocabulary")]
            try:
                result = subprocess.run([ARGS.driver, "-a", str(paths[0]), "-b", str(paths[1]),
                                         "-p", "check the argument", "-n", "32", "--answer-tokens", "16"],
                                        capture_output=True, text=True, timeout=15)
            finally:
                for peer in peers:
                    peer.close()
            for peer in peers:
                self.assertEqual(peer.errors, [])
            return result, peers

    def test_exact_cross_swap(self):
        result, peers = self.run_driver()
        self.assertEqual(result.returncode, 0, result.stderr)
        for index, peer in enumerate(peers):
            self.assertEqual(len(peer.calls), 2)
            self.assertEqual(peer.calls[0][1], peer.base)
            self.assertEqual(peer.calls[1][1], peer.base + peers[1 - index].reasoning)
            self.assertEqual(peer.calls[0][0]["stop"], ["</think>"])
            self.assertIn("</think>", peer.calls[0][0]["preserved_tokens"])
            self.assertFalse(peer.calls[0][0]["return_content"])
            self.assertIn(f"answer-{index}", result.stdout)

    def test_incomplete_reasoning_is_not_used(self):
        result, peers = self.run_driver(incomplete=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(result.stderr)
        self.assertTrue(all(len(peer.calls) == 1 for peer in peers))

    def test_tokenizer_mismatch_is_rejected(self):
        result, peers = self.run_driver(mismatch=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(all(not peer.calls for peer in peers))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", required=True)
    parser.add_argument("--driver", required=True)
    parser.add_argument("--model", required=True)
    ARGS, remaining = parser.parse_known_args()
    ARGS.server, ARGS.driver, ARGS.model = (str(pathlib.Path(path).resolve()) for path in (ARGS.server, ARGS.driver, ARGS.model))
    unittest.main(argv=[__file__, *remaining])
