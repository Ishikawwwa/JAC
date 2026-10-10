import asyncio
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from jac import CompatibleClient, OpenAICompatibleDecider, OpenAICompatibleEmbedder
from jevcache.decisions import Candidate
from jevcache.evaluate import ORACLE, FixtureEmbedder, run_evaluation


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((self.path, payload, self.headers.get("Authorization")))
        if self.path == "/v1/embeddings":
            vectors = FixtureEmbedder().embed(payload["input"])
            body = {"data": [{"index": i, "embedding": vector}
                             for i, vector in reversed(list(enumerate(vectors)))]}
        elif payload["model"] == "judge":
            state = json.loads(payload["messages"][-1]["content"])
            expected = ORACLE[state["request"]["text"]]
            rows = [{"id": c["id"],
                     "equivalent": ORACLE[c["request"]["text"]] == expected,
                     "satisfies": ORACLE[c["request"]["text"]] == expected}
                    for c in state["candidates"]]
            content = self.server.decision_override or json.dumps({"judgments": rows})
            body = {"choices": [{"message": {"content": content}}]}
        else:
            text = payload["messages"][-1]["content"]
            body = {"choices": [{"message": {"content": json.dumps({"answer": ORACLE[text]})}}]}
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class ProviderTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.server.requests = []
        cls.server.decision_override = None
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    async def test_real_http_adapter(self):
        client = CompatibleClient(base_url=self.url, model="generator", api_key="test-only")
        result = await client.complete(messages=[{"role": "user", "content": next(iter(ORACLE))}])
        self.assertIn("Paris", result["choices"][0]["message"]["content"])
        self.assertEqual(self.server.requests[-1][2], "Bearer test-only")
        self.assertNotIn("test-only", json.dumps(client.cache_identity))
        encoder = OpenAICompatibleEmbedder(base_url=self.url, model="encoder")
        vectors = await asyncio.to_thread(encoder.embed, list(ORACLE)[:2])
        self.assertEqual(vectors[0], vectors[1])
        judge = OpenAICompatibleDecider(base_url=self.url, model="judge")
        candidate = Candidate("known", {"text": next(iter(ORACLE))}, "Paris", 100)
        decision = await judge.decide({"text": list(ORACLE)[1]}, [candidate], "answer")
        self.assertEqual(decision.candidate_id, "known")
        self.assertEqual(decision.probabilities, {})

    async def test_strict_decision_json(self):
        judge = OpenAICompatibleDecider(base_url=self.url, model="judge")
        candidate = Candidate("known", {"text": next(iter(ORACLE))}, "Paris", 100)
        malformed = [
            '{"judgments":[]}',
            '{"judgments":[{"id":"unknown","equivalent":true,"satisfies":true}]}',
            '{"judgments":[{"id":"known","equivalent":1,"satisfies":true}]}',
            '{"judgments":[],"judgments":[]}',
            '```json\n{"judgments":[]}\n```',
        ]
        try:
            for content in malformed:
                self.server.decision_override = content
                with self.subTest(content=content), self.assertRaises((ValueError, KeyError)):
                    await judge.decide({"text": list(ORACLE)[1]}, [candidate], "answer")
        finally:
            self.server.decision_override = None

    async def test_live_evaluation_path_with_http_server(self):
        config = {
            "generation": {"model": "generator", "base_url": self.url},
            "decision": {"provider": "openai-compatible", "model": "judge",
                         "base_url": self.url},
            "embedding": {"provider": "openai-compatible", "model": "encoder",
                          "base_url": self.url},
        }
        report = await run_evaluation(config=config, repeats=1)
        self.assertFalse(report["fixture"])
        self.assertEqual(len(report["runs"]), 6)
        for row in report["runs"].values():
            self.assertEqual(row["incorrect_responses"], 0)
            self.assertEqual(row["calibration_incorrect_decisions"], 0)
            self.assertEqual(row["requests"], 14)

    async def test_endpoint_validation(self):
        for url in ("file:///tmp", "http://key@localhost/v1", "https://host/v1?token=secret"):
            with self.assertRaises(ValueError):
                CompatibleClient(model="model", base_url=url)
