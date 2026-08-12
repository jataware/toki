"""Install extras must not pull in other extras' runtime dependencies.

`toki[openrouter]` in particular must be able to import `OpenRouterModel`
without litellm (which belongs to `toki[openai]` / `toki[anthropic]` /
`toki[google]`).
"""

import sys

import pytest


_BACKEND_PREFIXES = (
    'toki.litellm',
    'toki.anthropic',
    'toki.openrouter',
    'toki.openai',
    'toki.google',
    'toki.ollama',
    'toki.local',
)


def _unload_backends() -> None:
    for name in list(sys.modules):
        if name in _BACKEND_PREFIXES or name.startswith(tuple(p + '.' for p in _BACKEND_PREFIXES)):
            del sys.modules[name]


@pytest.fixture
def no_litellm(monkeypatch):
    """Simulate a venv where litellm is not installed, with backend modules fresh."""
    _unload_backends()
    monkeypatch.setitem(sys.modules, 'litellm', None)
    yield
    _unload_backends()


def test_openrouter_imports_without_litellm(no_litellm):
    from toki import OpenRouterModel

    assert 'toki.litellm' not in sys.modules
    assert 'toki.litellm.model' not in sys.modules
    model = OpenRouterModel("anthropic/claude-3.5-haiku", api_key="dummy")
    assert model.model == "anthropic/claude-3.5-haiku"


def test_anthropic_utils_import_without_litellm(no_litellm):
    from toki.anthropic.utils import apply_cache_markers, get_anthropic_api_key

    assert callable(apply_cache_markers)
    assert callable(get_anthropic_api_key)
    assert 'toki.anthropic.model' not in sys.modules
    assert 'toki.litellm.model' not in sys.modules
