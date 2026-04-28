"""Dev tooling for the Ollama backend.

`list_ollama_chat_models` and `_create_models_types_file` together maintain the
curated `toki/ollama/models.py` snapshot.

Source: scrapes `https://ollama.com/library?sort=popular` for the popularity
ranking and capability tags (`tools`, `thinking`, `vision`, `embedding`, `cloud`)
which the registry HTTP API doesn't expose. For each family we then fetch
`https://ollama.com/library/<family>` to extract the context-window string.

Imports of `requests`, `bs4`, and `tqdm` live in the `dev` dependency group
rather than the `ollama` runtime extra.
"""

import importlib.util
import re
from os import PathLike
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


here = Path(__file__).parent

LIBRARY_URL = 'https://ollama.com/library'
LIBRARY_LIST_URL = f'{LIBRARY_URL}?sort=popular'

# capabilities surfaced as badge spans on the listing card. `embedding` flags
# embedding-only models which we drop (toki targets chat models).
KNOWN_CAPABILITIES = {'tools', 'thinking', 'vision', 'embedding', 'cloud'}

# Parameter-size tags use a lower-case suffix: 0.6b, 1.7b, 8b, 30b, 235b, 270m, 1m, etc.
# Some families also publish e.g. `7b-instruct` or `70b-chat` as separate badges; we keep
# only the canonical numeric+suffix form for the Literal (the registry accepts the bare
# numeric tag and resolves the alias).
SIZE_TAG_RE = re.compile(r'^\d+(?:\.\d+)?[bm]$')

# Context window strings: "32K context window", "128K context window", "1M context window".
CONTEXT_RE = re.compile(r'(\d+(?:\.\d+)?)\s*([kKmM])\s*context\s+window', re.IGNORECASE)

USER_AGENT = 'toki-fetch-ollama-models (+https://github.com/jataware/toki)'


def _get(url: str) -> BeautifulSoup:
    r = requests.get(url, timeout=30, headers={'User-Agent': USER_AGENT})
    r.raise_for_status()
    return BeautifulSoup(r.text, 'html.parser')


def _parse_listing(soup: BeautifulSoup) -> list[dict]:
    """Walk the library listing and return the family rows in popularity order.

    Each row: {family, capabilities: set[str], sizes: list[str]}.
    """
    out: list[dict] = []
    seen: set[str] = set()
    # Each card on the listing wraps a link to `/library/<family>` (no trailing slash, no `/tags`).
    for a in soup.find_all('a', href=True):
        href: str = a['href']
        if not href.startswith('/library/'):
            continue
        family = href[len('/library/'):].strip('/')
        if not family or '/' in family:
            continue
        if family in seen:
            continue
        # filter out the bare "/library" sort/filter links by requiring the anchor to look like a card,
        # which we detect heuristically by presence of a heading (h2/h3) inside.
        heading = a.find(['h2', 'h3'])
        if heading is None or heading.get_text(strip=True) != family:
            continue
        seen.add(family)

        text = a.get_text(' ', strip=True)
        tokens = text.split()
        capabilities: set[str] = {t for t in tokens if t in KNOWN_CAPABILITIES}
        sizes: list[str] = []
        for t in tokens:
            if SIZE_TAG_RE.match(t) and t not in sizes:
                sizes.append(t)

        out.append({'family': family, 'capabilities': capabilities, 'sizes': sizes})
    return out


def _fetch_context_size(family: str) -> int | None:
    """Pull "<N>K context window" / "<N>M context window" off the family page and convert to int tokens."""
    try:
        soup = _get(f'{LIBRARY_URL}/{family}')
    except requests.RequestException:
        return None
    text = soup.get_text(' ', strip=True)
    m = CONTEXT_RE.search(text)
    if m is None:
        return None
    n = float(m.group(1))
    unit = m.group(2).lower()
    multiplier = 1024 if unit == 'k' else 1024 * 1024
    return int(n * multiplier)


def list_ollama_chat_models(
    *,
    top_k: int = 100,
    known_attrs: dict[str, dict] | None = None,
) -> list[dict]:
    """Return the current top `top_k` non-embedding ollama families with their tags.

    Output rows are one per concrete `family:tag` (one for each parameter size on
    the card, plus `<family>:latest`), shaped:

        {id, context_size, supports_tools, supports_thinking}

    `known_attrs` is an optional `{id: {context_size, supports_tools, supports_thinking}}`
    map of already-known tags. Ids appearing here skip the per-family HTTP fetch
    and reuse the cached attributes - making refreshes cheap.
    """
    known_attrs = known_attrs or {}
    listing_soup = _get(LIBRARY_LIST_URL)
    rows = _parse_listing(listing_soup)
    rows = [r for r in rows if 'embedding' not in r['capabilities']]
    rows = rows[:top_k]

    out: list[dict] = []
    for row in tqdm(rows, desc='Fetching family pages'):
        family = row['family']
        supports_tools = 'tools' in row['capabilities']
        supports_thinking = 'thinking' in row['capabilities']
        sizes = row['sizes'] or []
        # always include `:latest`; per-size tags follow
        tag_ids = [f'{family}:latest'] + [f'{family}:{s}' for s in sizes]

        context_size: int | None = None
        for tid in tag_ids:
            if tid in known_attrs:
                ctx = known_attrs[tid].get('context_size')
                if isinstance(ctx, int):
                    context_size = ctx
                    break
        if context_size is None:
            context_size = _fetch_context_size(family)
        if context_size is None:
            continue  # skip families we can't determine a context size for

        for tid in tag_ids:
            out.append({
                'id': tid,
                'context_size': context_size,
                'supports_tools': supports_tools,
                'supports_thinking': supports_thinking,
            })

    return out


def _load_existing_attrs(file: Path) -> dict[str, dict]:
    """Read id -> {context_size, supports_tools, supports_thinking} from an existing models.py.

    Returns {} if the file doesn't exist, isn't loadable, or its `Attr` shape
    has changed - in any of which cases the caller should regenerate from scratch.
    """
    if not file.exists():
        return {}
    spec = importlib.util.spec_from_file_location('_toki_ollama_models_existing', file)
    if spec is None or spec.loader is None:
        return {}
    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception:
        return {}
    attrs_map = getattr(mod, 'attributes_map', None)
    if not isinstance(attrs_map, dict):
        return {}
    out: dict[str, dict] = {}
    for k, v in attrs_map.items():
        if hasattr(v, 'context_size') and hasattr(v, 'supports_tools') and hasattr(v, 'supports_thinking'):
            out[k] = {
                'context_size': v.context_size,
                'supports_tools': v.supports_tools,
                'supports_thinking': v.supports_thinking,
            }
    return out


def _create_models_types_file(*, top_k: int = 100, file: PathLike = here / 'models.py'):
    """Dev helper: regenerate `toki/ollama/models.py`.

    Pulls the *current* top `top_k` families from the ollama.com library (sorted
    by popularity) and merges their tags into the snapshot, **pruning any tags
    that are no longer in the registry**. Unlike the local-models snapshot, this
    is not purely additive - tags can disappear from ollama.
    """
    file = Path(file)
    existing = _load_existing_attrs(file)
    if existing:
        print(f"Loaded {len(existing)} existing entries from {file.relative_to(here.parent)}")

    fresh = list_ollama_chat_models(top_k=top_k, known_attrs=existing)
    if not fresh:
        raise RuntimeError("No models found - check the listing page structure / network")

    fresh_ids = {m['id'] for m in fresh}
    pruned = [k for k in existing if k not in fresh_ids]
    new_ids = [m['id'] for m in fresh if m['id'] not in existing]
    print(f"Top {len(fresh)} family-tags: {len(new_ids)} new, {len(fresh) - len(new_ids)} already known, {len(pruned)} pruned")

    merged: dict[str, dict] = {
        m['id']: {
            'context_size': m['context_size'],
            'supports_tools': m['supports_tools'],
            'supports_thinking': m['supports_thinking'],
        }
        for m in fresh
    }

    sorted_ids = sorted(merged.keys(), key=str.lower)

    print(f"Writing {file.relative_to(here.parent)} with {len(sorted_ids)} models")
    name_lines = ',\n    '.join(f"'{i}'" for i in sorted_ids)
    attributes_lines = ',\n    '.join(
        f'''{f'"{i}":':<48}Attr(context_size={merged[i]["context_size"]}, supports_tools={merged[i]["supports_tools"]}, supports_thinking={merged[i]["supports_thinking"]})'''
        for i in sorted_ids
    )
    file.write_text(f'''\
# DO NOT EDIT THIS FILE MANUALLY
# This file is generated by calling `_create_models_types_file()` in toki/ollama/fetch_models.py
# (or via the `toki-fetch-ollama-models` script).
#
# NOTE: this list is NEITHER EXHAUSTIVE NOR GUARANTEED CURRENT. It is a curated
# snapshot of popular ollama.com/library models, refreshed by the codegen
# script. Many other valid tags exist (quantizations, fine-tunes, third-party
# uploads); the ollama backend accepts any string the daemon can pull, so
# passing a tag outside this Literal still works. The Literal is here purely
# for IDE autocomplete on the popular cases.

from typing import Literal
from dataclasses import dataclass


OllamaModelName = Literal[
    {name_lines}
]


@dataclass
class Attr:
    context_size: int
    supports_tools: bool
    supports_thinking: bool


attributes_map: dict[OllamaModelName, Attr] = {{
    {attributes_lines}
}}
''')
