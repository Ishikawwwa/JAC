"""Client proxy tests use local fakes and never call a live provider."""

import functools
import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from jevcache.adapters import wrap_client


class FakeModel:
    def model_dump(self, **kwargs):
        return {"text": "cached output"}

    @classmethod
    def model_validate(cls, value):
        return cls()


class FakeCache:
    namespace = "test-app"

    def __init__(self):
        self.calls = []

    def wrap(self, method, *, response_type=None, namespace=None):
        self.calls.append(
            {"method": method, "response_type": response_type, "namespace": namespace}
        )
        return method


class FakeClient:
    def __init__(self, endpoint="chat", base_url="https://provider.example/v1", async_mode=False):
        self.base_url = base_url
        self.organization = "org-test"
        self.project = "project-test"
        self.default_headers = {"Authorization": "Bearer fake-secret"}
        self.default_query = {}
        self.closed = False
        self.invocations = []

        def create(*, model, messages=None, input=None, stream=False):
            self.invocations.append((model, messages, input, stream))
            return {"text": "normal output", "stream": stream}

        async def async_create(*, model, messages=None, input=None, stream=False):
            return create(model=model, messages=messages, input=input, stream=stream)

        method = async_create if async_mode else create
        resource = SimpleNamespace(
            create=method,
            retrieve=lambda identifier: identifier,
            stream=lambda: "original-stream-helper",
        )
        if endpoint == "chat":
            self.chat = SimpleNamespace(completions=resource)
        else:
            setattr(self, endpoint, resource)
        self.models = SimpleNamespace(list=lambda: ["model-a"])

    def close(self):
        self.closed = True

    def with_options(self, **kwargs):
        return FakeClient(base_url=kwargs.get("base_url", self.base_url))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.close()


class AdapterTests(unittest.TestCase):
    def test_installed_openai_chat_typed_hit_without_network(self):
        try:
            from openai import OpenAI
            from openai.types.chat import ChatCompletion
        except ImportError:
            self.skipTest("optional OpenAI SDK is not installed")
        from jevcache.call_cache import JAC

        generations = []
        client = OpenAI(api_key="offline-placeholder")

        def create(**kwargs):
            generations.append(kwargs)
            return ChatCompletion(
                id="offline-response",
                created=0,
                model="offline-model",
                object="chat.completion",
                choices=[
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "local response"},
                    }
                ],
            )

        client.chat.completions.create = create
        try:
            with JAC(path=":memory:") as cache:
                proxy = cache.wrap_client(client)
                arguments = {
                    "model": "offline-model",
                    "messages": [{"role": "user", "content": "hello"}],
                }
                first = proxy.chat.completions.create(**arguments)
                second = proxy.chat.completions.create(**arguments)
                self.assertIsInstance(second, ChatCompletion)
                self.assertEqual(first.model_dump(), second.model_dump())
                self.assertEqual(second.choices[0].message.content, "local response")
                self.assertEqual(len(generations), 1)
                self.assertEqual(cache.last_trace.llm_calls, 0)
        finally:
            client.close()

    def test_sdk_omitted_headers_do_not_randomize_namespace(self):
        Omit = type("Omit", (), {"__module__": "openai"})

        class ClientWithOmittedHeaders(FakeClient):
            def __init__(self):
                super().__init__()

            @property
            def default_headers(self):
                return {"Authorization": "Bearer fake-key", "OpenAI-Organization": Omit()}

            @default_headers.setter
            def default_headers(self, value):
                pass

        client = ClientWithOmittedHeaders()
        cache = FakeCache()
        create = wrap_client(cache, client).chat.completions.create
        create(model="model-a")
        create(model="model-a")
        self.assertEqual(cache.calls[0]["namespace"], cache.calls[1]["namespace"])

    def test_endpoint_exposes_calibration_and_metrics_helpers(self):
        class ObservedCache(FakeCache):
            last_trace = "local-trace"

            def __init__(self):
                super().__init__()
                self.observations = []

            def wrap(self, method, *, response_type=None, namespace=None):
                super().wrap(method, response_type=response_type, namespace=namespace)

                @functools.wraps(method)
                def decorated(*args, **kwargs):
                    return method(*args, **kwargs)

                decorated.observe_reuse = lambda *args, **kwargs: self.observations.append(
                    (args, kwargs, namespace)
                )
                return decorated

        cache = ObservedCache()
        create = wrap_client(cache, FakeClient()).chat.completions.create
        request = {"model": "model-a", "messages": [{"role": "user", "content": "hello"}]}
        create.observe_reuse(request, valid_hit=True, decision_ms=10)
        self.assertEqual(cache.observations[0][0], (request,))
        self.assertEqual(
            cache.observations[0][1], {"valid_hit": True, "decision_ms": 10, "request_args": ()}
        )
        self.assertEqual(create.cache_info(), "local-trace")
        self.assertIs(create.cache, cache)

    def test_generic_compatible_surface_uses_json_without_vendor_imports(self):
        cache = FakeCache()
        client = FakeClient()
        with patch("jevcache.adapters.importlib.import_module") as importer:
            wrapped = wrap_client(cache, client)
            result = wrapped.chat.completions.create(model="model-a", messages=[{"role": "user"}])
            importer.assert_not_called()
        self.assertEqual(result["text"], "normal output")
        self.assertEqual(len(client.invocations), 1)
        self.assertIsNone(cache.calls[0]["response_type"])
        self.assertIn("test-app:openai:chat.completions.create:", cache.calls[0]["namespace"])
        self.assertNotIn("fake-secret", cache.calls[0]["namespace"])
        self.assertNotIn("provider.example", cache.calls[0]["namespace"])

    def test_known_sdk_loads_only_the_accessed_response_codec(self):
        cache = FakeCache()
        client = FakeClient()
        with patch.object(FakeClient, "__module__", "openai._client"):
            with patch(
                "jevcache.adapters.importlib.import_module",
                return_value=SimpleNamespace(ChatCompletion=FakeModel),
            ) as importer:
                wrapped = wrap_client(cache, client)
                importer.assert_not_called()
                wrapped.chat.completions.create(model="model-a")
        importer.assert_called_once_with("openai.types.chat")
        self.assertIs(cache.calls[0]["response_type"], FakeModel)

    def test_optional_responses_are_independent_of_chat_support(self):
        cache = FakeCache()
        client = FakeClient()
        with patch.object(FakeClient, "__module__", "openai._client"):
            with patch(
                "jevcache.adapters.importlib.import_module",
                return_value=SimpleNamespace(ChatCompletion=FakeModel),
            ) as importer:
                wrapped = wrap_client(cache, client)
                wrapped.chat.completions.create(model="model-a")
                self.assertFalse(hasattr(wrapped, "responses"))
        importer.assert_called_once_with("openai.types.chat")

    def test_known_vendor_without_codec_preserves_original_method_and_result(self):
        cache = FakeCache()
        client = FakeClient(endpoint="responses")
        original = client.responses.create
        with patch.object(FakeClient, "__module__", "openai._client"):
            with patch("jevcache.adapters.importlib.import_module", side_effect=ImportError):
                wrapped = wrap_client(cache, client)
                self.assertIs(wrapped.responses.create, original)
                self.assertEqual(
                    wrapped.responses.create(model="model-a", input="hello")["text"],
                    "normal output",
                )
        self.assertEqual(cache.calls, [])

    def test_responses_and_anthropic_codecs(self):
        for endpoint, provider, module, model_name in (
            ("responses", "openai", "openai.types.responses", "Response"),
            ("messages", "anthropic", "anthropic.types", "Message"),
        ):
            cache = FakeCache()
            client = FakeClient(endpoint=endpoint)
            with patch.object(FakeClient, "__module__", provider):
                with patch(
                    "jevcache.adapters.importlib.import_module",
                    return_value=SimpleNamespace(**{model_name: FakeModel}),
                ) as importer:
                    wrapped = wrap_client(cache, client)
                    getattr(wrapped, endpoint).create(model="model-a")
            with self.subTest(endpoint=endpoint):
                importer.assert_called_once_with(module)
                self.assertIs(cache.calls[0]["response_type"], FakeModel)
                self.assertIn(f":{provider}:{endpoint}.create:", cache.calls[0]["namespace"])

    def test_base_url_and_client_configuration_partition_namespaces(self):
        cache = FakeCache()
        client = FakeClient()
        wrapped = wrap_client(cache, client)
        create = wrapped.chat.completions.create
        create(model="model-a")
        initial = cache.calls[-1]["namespace"]
        wrapped.base_url = "https://other.example/v1"
        create(model="model-a")
        self.assertNotEqual(initial, cache.calls[-1]["namespace"])
        client.base_url = "https://provider.example/v1"
        client.default_headers = {"Authorization": "Bearer different-fake-secret"}
        create(model="model-a")
        self.assertNotEqual(initial, cache.calls[-1]["namespace"])

    def test_other_attributes_and_methods_pass_through(self):
        client = FakeClient()
        wrapped = wrap_client(FakeCache(), client)
        self.assertEqual(wrapped.base_url, client.base_url)
        self.assertIs(wrapped.models, client.models)
        self.assertEqual(wrapped.chat.completions.retrieve("id-1"), "id-1")
        self.assertEqual(wrapped.chat.completions.stream(), "original-stream-helper")
        wrapped.close()
        self.assertTrue(client.closed)
        self.assertIn("chat", dir(wrapped))

    def test_call_signature_and_stream_argument_are_preserved(self):
        client = FakeClient()
        cache = FakeCache()
        wrapped = wrap_client(cache, client)
        self.assertEqual(
            inspect.signature(wrapped.chat.completions.create),
            inspect.signature(client.chat.completions.create),
        )
        result = wrapped.chat.completions.create(model="model-a", stream=True)
        self.assertTrue(result["stream"])
        self.assertTrue(client.invocations[-1][-1])

    def test_cloned_clients_remain_wrapped(self):
        cache = FakeCache()
        client = FakeClient()
        wrapped = wrap_client(cache, client)
        cloned = wrapped.with_options(base_url="https://other.example/v1")
        cloned.chat.completions.create(model="model-a")
        wrapped.chat.completions.create(model="model-a")
        self.assertNotEqual(cache.calls[0]["namespace"], cache.calls[1]["namespace"])

    def test_sync_context_manager_retains_proxy(self):
        client = FakeClient()
        cache = FakeCache()
        with wrap_client(cache, client) as wrapped:
            wrapped.chat.completions.create(model="model-a")
        self.assertEqual(len(cache.calls), 1)
        self.assertTrue(client.closed)

    def test_unsupported_client_has_actionable_error(self):
        with self.assertRaisesRegex(TypeError, "cache.wrap"):
            wrap_client(FakeCache(), SimpleNamespace())


class AsyncAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_create_preserves_await_and_signature(self):
        client = FakeClient(async_mode=True)
        cache = FakeCache()
        wrapped = wrap_client(cache, client)
        self.assertTrue(inspect.iscoroutinefunction(wrapped.chat.completions.create))
        self.assertEqual(
            inspect.signature(wrapped.chat.completions.create),
            inspect.signature(client.chat.completions.create),
        )
        result = await wrapped.chat.completions.create(model="model-a", messages=[])
        self.assertEqual(result["text"], "normal output")
        self.assertEqual(len(cache.calls), 1)

    async def test_async_context_manager_retains_proxy(self):
        client = FakeClient(async_mode=True)
        cache = FakeCache()
        async with wrap_client(cache, client) as wrapped:
            await wrapped.chat.completions.create(model="model-a")
        self.assertTrue(client.closed)
        self.assertEqual(len(cache.calls), 1)


if __name__ == "__main__":
    unittest.main()
