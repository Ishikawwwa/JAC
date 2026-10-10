"""Optional SDK proxies: keep the provider API while caching create calls.

No provider package is imported when this module is loaded. Known response codecs
are loaded only for the corresponding endpoint. Streaming and unsupported request
shapes are handled by the call cache's normal pass-through behavior.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import inspect
import json
from collections.abc import Mapping
from typing import Any

_ENDPOINTS = {
    ("chat", "completions", "create"): ("openai", "openai.types.chat", "ChatCompletion"),
    ("responses", "create"): ("openai", "openai.types.responses", "Response"),
    ("messages", "create"): ("anthropic", "anthropic.types", "Message"),
}


def _accessible_path(client: Any, path: tuple[str, ...]) -> bool:
    current = client
    try:
        for name in path:
            current = getattr(current, name)
    except AttributeError:
        return False
    return callable(current)


def _known_vendor(client: Any, provider: str) -> bool:
    return any(parent.__module__.split(".", 1)[0] == provider for parent in type(client).__mro__)


def _optional_model(module: str, name: str) -> type | None:
    try:
        response_type = getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError):
        return None
    if not isinstance(response_type, type) or not (
        callable(getattr(response_type, "model_dump", None))
        and callable(getattr(response_type, "model_validate", None))
    ):
        return None
    return response_type


def _namespace(cache: Any, client: Any, provider: str, path: tuple[str, ...]) -> str:
    # Use only public SDK configuration. Headers include account and workspace
    # selectors; hash them rather than persisting credentials in namespace text.
    configuration: dict[str, Any] = {"base_url": str(getattr(client, "base_url", ""))}
    for field in ("organization", "project", "api_version", "default_headers", "default_query"):
        value = getattr(client, field, None)
        if isinstance(value, Mapping):
            configuration[field] = {
                str(key): str(item)
                for key, item in value.items()
                if not (
                    type(item).__module__.split(".", 1)[0] in {"openai", "anthropic"}
                    and type(item).__name__ in {"Omit", "NotGiven"}
                )
            }
        elif value is None or isinstance(value, (str, bool, int, float)):
            configuration[field] = value
        else:
            configuration[field] = str(value)
    digest = hashlib.sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"{getattr(cache, 'namespace', 'default')}:{provider}:{'.'.join(path)}:{digest}"


def wrap_client(cache: Any, client: Any) -> Any:
    """Wrap supported create endpoints without changing their call signatures.

    OpenAI chat/completions, OpenAI responses, and Anthropic messages are supported
    when present. Known SDK responses retain their model type through an optional
    codec. Compatible non-SDK clients use the generic JSON response contract.
    Other attributes and methods pass through. This function performs no network
    operation and does not require provider credentials.
    """
    if not callable(getattr(cache, "wrap", None)):
        raise TypeError("cache must provide wrap(existing_callable, response_type, namespace)")
    endpoints = {
        path: codec for path, codec in _ENDPOINTS.items() if _accessible_path(client, path)
    }
    if not endpoints:
        raise TypeError("unsupported client; use cache.wrap(existing_callable) instead")
    return _ClientProxy(cache, client, client, (), endpoints)


class _ClientProxy:
    def __init__(self, cache, target, client, prefix, endpoints):
        object.__setattr__(self, "_cache", cache)
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_client", client)
        object.__setattr__(self, "_prefix", prefix)
        object.__setattr__(self, "_endpoints", endpoints)
        object.__setattr__(self, "_children", {})

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._target, name)
        path = self._prefix + (name,)
        if path in self._endpoints:
            return self._create(value, path, self._endpoints[path])
        if any(endpoint[: len(path)] == path for endpoint in self._endpoints):
            existing = self._children.get(name)
            if existing is None or existing._target is not value:
                existing = _ClientProxy(self._cache, value, self._client, path, self._endpoints)
                self._children[name] = existing
            return existing
        if not self._prefix and name in {"with_options", "copy"} and callable(value):

            @functools.wraps(value)
            def clone(*args, **kwargs):
                return wrap_client(self._cache, value(*args, **kwargs))

            return clone
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._target, name, value)

    def __dir__(self) -> list[str]:
        return sorted(set(dir(self._target)) | set(super().__dir__()))

    def _create(self, method, path, codec):
        provider, module, model_name = codec
        response_type = None
        if _known_vendor(self._client, provider):
            response_type = _optional_model(module, model_name)
            if response_type is None:
                # Preserve vendor response objects if an old SDK has no compatible
                # model codec. Never silently turn a model into a cached dict.
                return method

        def wrapped():
            return self._cache.wrap(
                method,
                response_type=response_type,
                namespace=_namespace(self._cache, self._client, provider, path),
            )

        # Compute the namespace at invocation, including after mutable SDK client
        # settings change or when the application retains a reference to create.
        if inspect.iscoroutinefunction(method) or inspect.iscoroutinefunction(
            getattr(method, "__call__", None)
        ):

            @functools.wraps(method)
            async def async_create(*args, **kwargs):
                return await wrapped()(*args, **kwargs)

            create = async_create
        else:

            @functools.wraps(method)
            def sync_create(*args, **kwargs):
                return wrapped()(*args, **kwargs)

            create = sync_create

        def observe_reuse(request_kwargs, *, valid_hit, decision_ms, request_args=()):
            observer = getattr(wrapped(), "observe_reuse", None)
            if not callable(observer):
                raise AttributeError("the cache wrapper does not support reuse observations")
            return observer(
                request_kwargs,
                valid_hit=valid_hit,
                decision_ms=decision_ms,
                request_args=request_args,
            )

        create.observe_reuse = observe_reuse
        create.cache_info = lambda: getattr(self._cache, "last_trace", None)
        create.cache = self._cache
        return create

    def __enter__(self):
        entered = self._target.__enter__()
        return self if entered is self._target else entered

    def __exit__(self, *args):
        return self._target.__exit__(*args)

    async def __aenter__(self):
        entered = await self._target.__aenter__()
        return self if entered is self._target else entered

    async def __aexit__(self, *args):
        return await self._target.__aexit__(*args)
