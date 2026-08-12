"""Dev tooling for the local backend.

`list_local_chat_models` and `_create_models_types_file` together maintain the
curated `toki/local/models.py` snapshot of HuggingFace chat models. Imports
here (`huggingface_hub`, `tqdm`) live in the `dev` dependency group rather
than the `local` runtime extra.
"""

import importlib.util
import json
from os import PathLike
from pathlib import Path
from typing import Literal

from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from huggingface_hub.utils import HfHubHTTPError, disable_progress_bars

try:
    from huggingface_hub.errors import GatedRepoError
except ImportError:  # huggingface_hub < 0.23
    from huggingface_hub.utils import GatedRepoError

from tqdm import tqdm

# silence the per-file progress bars that hf_hub_download prints for tiny
# config / chat_template files - they completely clobber our outer tqdm.
disable_progress_bars()


here = Path(__file__).parent


# pipeline tags whose models work with toki's chat-message API
# (multimodal models can still be driven text-only)
DEFAULT_PIPELINE_TAGS = ("text-generation", "image-text-to-text")

# common keys for the model's max sequence length, in priority order. Newer
# multimodal configs nest these under `text_config` / `language_config`.
_CONTEXT_SIZE_KEYS = ("max_position_embeddings", "n_positions", "seq_length", "max_seq_len", "max_sequence_length")
_NESTED_CONFIG_KEYS = ("text_config", "language_config", "llm_config", "decoder", "model_config")

# HF sort modes:
#   - "downloads" is HF's ~30-day download count (NOT all-time; HF doesn't expose
#     an all-time sort), so this is a recent-popularity signal.
#   - "likes" is cumulative all-time likes — the historical "this model mattered"
#     signal. Classics like Mistral-7B stay near the top long after their 30-day
#     download rank falls off.
#   - "trending_score" is a much shorter-window buzz signal that mixes downloads,
#     likes, and discussion velocity. Use this to surface brand-new hot models.
SortBy = Literal["downloads", "likes", "trending_score"]


def _hub_score(m, sort_by: SortBy) -> int:
    if sort_by == "downloads":
        return m.downloads or 0
    if sort_by == "likes":
        return getattr(m, "likes", 0) or 0
    return getattr(m, "trending_score", 0) or 0


def _hub_candidates(
    *,
    sort_by: SortBy,
    limit: int,
    pipeline_tags: tuple[str, ...],
    min_downloads: int,
) -> list:
    """Hub repos for `sort_by`, de-duped across pipeline tags, highest score first."""
    api = HfApi()
    seen: set[str] = set()
    scored: list[tuple[int, object]] = []
    for tag in pipeline_tags:
        for m in api.list_models(
            pipeline_tag=tag,
            sort=sort_by,
            limit=limit,
        ):
            if m.id in seen:
                continue
            seen.add(m.id)
            if min_downloads and (m.downloads or 0) < min_downloads:
                continue
            scored.append((_hub_score(m, sort_by), m))
    scored.sort(key=lambda x: -x[0])
    return [m for _, m in scored]


def _verify_chat_models(
    candidates,
    *,
    top_k: int,
    known_attrs: dict[str, dict],
    desc: str,
) -> list[dict]:
    """Walk Hub candidates until `top_k` repos with a chat template are verified."""
    verified: list[dict] = []
    with tqdm(total=top_k, desc=desc) as pbar:
        for m in candidates:
            if m.id in known_attrs:
                verified.append({"id": m.id, **known_attrs[m.id]})
            else:
                try:
                    attrs = _fetch_model_attrs(m.id)
                except _HubSkip:
                    continue
                verified.append({"id": m.id, **attrs})
            pbar.update(1)
            if len(verified) >= top_k:
                break
    return verified


def list_local_chat_models(
    *,
    top_k: int = 100,
    min_downloads: int = 1000,
    pipeline_tags: tuple[str, ...] = DEFAULT_PIPELINE_TAGS,
    sort_by: SortBy = "downloads",
    known_attrs: dict[str, dict] | None = None,
) -> list[dict]:
    """
    Return the current top `top_k` instruction-tuned chat models on HuggingFace
    Hub, sorted by `sort_by` and verified to ship a tokenizer `chat_template`.

    For each kept repo also reports `context_size` (from `config.json`) and
    `supports_tools` (inferred from the template body referencing `tools`).

    `known_attrs` is an optional `{id: {context_size, supports_tools}}` map of
    already-verified repos. Ids appearing here are counted toward `top_k`
    using the supplied attrs, with no HTTP work - making refreshes cheap when
    most of the popular list is unchanged.

    Returns a list of dicts: {id, context_size, supports_tools}.
    """
    known_attrs = known_attrs or {}
    overshoot = top_k * 4  # popular list contains many base/non-chat repos, so over-fetch
    candidates = _hub_candidates(
        sort_by=sort_by,
        limit=overshoot,
        pipeline_tags=pipeline_tags,
        min_downloads=min_downloads,
    )
    return _verify_chat_models(
        candidates,
        top_k=top_k,
        known_attrs=known_attrs,
        desc=f"Verifying ({sort_by})",
    )


class _HubSkip(Exception):
    """Repo cannot be included: gated, missing template, missing context size, etc."""
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _hub_download(repo_id: str, filename: str) -> Path | None:
    try:
        return Path(hf_hub_download(repo_id, filename))
    except GatedRepoError as e:
        raise _HubSkip("gated repo — set HF_TOKEN and accept the model license on HuggingFace") from e
    except HfHubHTTPError:
        return None


def _hub_json(repo_id: str, filename: str) -> dict | list | None:
    path = _hub_download(repo_id, filename)
    if path is None:
        return None
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return data


def _template_from_obj(data: object) -> str | None:
    """Pull a Jinja chat-template string out of tokenizer_config / chat_template.json shapes."""
    if isinstance(data, str) and data.strip():
        return data
    if isinstance(data, list):
        named = [e for e in data if isinstance(e, dict) and isinstance(e.get("template"), str)]
        for e in named:
            if e.get("name") in (None, "default"):
                return e["template"]
        if named:
            return named[0]["template"]
        for e in data:
            if isinstance(e, str) and e.strip():
                return e
        return None
    if isinstance(data, dict):
        for key in ("chat_template", "default"):
            got = _template_from_obj(data.get(key))
            if got:
                return got
        for v in data.values():
            if isinstance(v, str) and ("{%" in v or "{{" in v):
                return v
    return None


def _fetch_model_attrs(repo_id: str) -> dict:
    """Fetch chat template + context size.

    Raises `_HubSkip` if the repo is gated, has no chat template, or has no
    readable context size.
    """
    chat_template = _get_chat_template_text(repo_id)
    if chat_template is None:
        raise _HubSkip("no chat_template (tokenizer_config / chat_template.jinja / chat_template.json)")
    context_size = _get_context_size(repo_id)
    if context_size is None:
        raise _HubSkip("no context_size in config.json")
    return {
        "context_size": context_size,
        "supports_tools": "tools" in chat_template,
    }


def _get_chat_template_text(repo_id: str) -> str | None:
    """Locate a chat template the way recent transformers does.

    Order: `chat_template.jinja`, `chat_template.json`, then `chat_template`
    on `tokenizer_config.json` / `processor_config.json` (string or named-list).
    """
    try:
        files = list_repo_files(repo_id)
    except GatedRepoError as e:
        raise _HubSkip("gated repo — set HF_TOKEN and accept the model license on HuggingFace") from e
    except HfHubHTTPError:
        return None

    file_set = set(files)
    candidates: list[str] = []
    for name in ("chat_template.jinja", "chat_template.json"):
        if name in file_set:
            candidates.append(name)
        candidates.extend(sorted(f for f in files if f.endswith("/" + name)))
    for name in ("tokenizer_config.json", "processor_config.json"):
        if name in file_set:
            candidates.append(name)

    for filename in candidates:
        if filename.endswith(".jinja"):
            path = _hub_download(repo_id, filename)
            if path is None:
                continue
            try:
                text = path.read_text()
            except OSError:
                continue
            if text.strip():
                return text
            continue
        data = _hub_json(repo_id, filename)
        if data is None:
            continue
        if filename.endswith("tokenizer_config.json") or filename.endswith("processor_config.json"):
            got = _template_from_obj(data.get("chat_template")) if isinstance(data, dict) else None
        else:
            got = _template_from_obj(data)
        if got:
            return got
    return None


def _context_from_dict(cfg: dict) -> int | None:
    for key in _CONTEXT_SIZE_KEYS:
        v = cfg.get(key)
        if isinstance(v, int) and v > 0:
            return v
    return None


def _get_context_size(repo_id: str) -> int | None:
    cfg = _hub_json(repo_id, "config.json")
    if not isinstance(cfg, dict):
        return None
    n = _context_from_dict(cfg)
    if n is not None:
        return n
    for key in _NESTED_CONFIG_KEYS:
        nested = cfg.get(key)
        if isinstance(nested, dict):
            n = _context_from_dict(nested)
            if n is not None:
                return n
    return None


def _load_existing_attrs(file: Path) -> dict[str, dict]:
    """Read id -> {context_size, supports_tools} from an existing models.py.

    Returns {} if the file doesn't exist, isn't loadable, or its `Attr` shape
    has changed - in any of which cases the caller should regenerate from
    scratch.
    """
    if not file.exists():
        return {}
    spec = importlib.util.spec_from_file_location("_toki_local_models_existing", file)
    if spec is None or spec.loader is None:
        return {}
    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception:
        return {}
    attrs_map = getattr(mod, "attributes_map", None)
    if not isinstance(attrs_map, dict):
        return {}
    out: dict[str, dict] = {}
    for k, v in attrs_map.items():
        if hasattr(v, "context_size") and hasattr(v, "supports_tools"):
            out[k] = {"context_size": v.context_size, "supports_tools": v.supports_tools}
    return out


def _load_curated_ids(path: Path) -> list[str]:
    """Repo ids from `curated.txt`: one per line, `#` comments and blanks skipped."""
    ids: list[str] = []
    seen: set[str] = set()
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line in seen:
            continue
        seen.add(line)
        ids.append(line)
    return ids


def _verify_curated_models(ids: list[str]) -> tuple[list[dict], list[tuple[str, str]]]:
    """Run the chat-template / context-size check on curated ids.

    Returns (kept, skipped) where skipped is `(repo_id, reason)`.
    """
    kept: list[dict] = []
    skipped: list[tuple[str, str]] = []
    with tqdm(total=len(ids), desc="Verifying (curated)") as pbar:
        for repo_id in ids:
            try:
                attrs = _fetch_model_attrs(repo_id)
            except _HubSkip as e:
                skipped.append((repo_id, e.reason))
            else:
                kept.append({"id": repo_id, **attrs})
            pbar.update(1)
    return kept, skipped


def _create_models_types_file(
    *,
    download_k: int = 200,
    likes_k: int = 200,
    min_downloads: int = 1000,
    file: PathLike = here / "models.py",
):
    """
    Dev helper: regenerate `toki/local/models.py` with a `LocalModelName` Literal
    of popular instruction-tuned chat models for IDE autocomplete.

    The snapshot is the union of:

      - ids in `toki/local/curated.txt` that pass the chat-template check
      - top `download_k` chat models by 30-day downloads (currently used)
      - top `likes_k` chat models by all-time likes (historical classics that
        no longer chart on the 30-day list)

    Popularity buckets are bounded; curated ids persist across refreshes so
    long as they remain chat-compatible. The local backend still accepts any
    HF repo id as a plain string; this Literal is only autocomplete.
    """
    file = Path(file)
    existing = _load_existing_attrs(file)
    if existing:
        print(f"Loaded {len(existing)} existing entries from {file.relative_to(here.parent)}")

    known = dict(existing)
    buckets: list[tuple[str, list[dict]]] = []

    curated_ids = _load_curated_ids(here / "curated.txt")
    curated_kept, curated_skipped = _verify_curated_models(curated_ids)
    buckets.append(("curated", curated_kept))
    for m in curated_kept:
        known[m["id"]] = {"context_size": m["context_size"], "supports_tools": m["supports_tools"]}
    if curated_skipped:
        print("Curated ids skipped:")
        for repo_id, reason in curated_skipped:
            print(f"  {repo_id}: {reason}")

    if download_k:
        by_downloads = list_local_chat_models(
            top_k=download_k,
            min_downloads=min_downloads,
            sort_by="downloads",
            known_attrs=known,
        )
        buckets.append(("downloads", by_downloads))
        for m in by_downloads:
            known[m["id"]] = {"context_size": m["context_size"], "supports_tools": m["supports_tools"]}
    if likes_k:
        # No 30-day download floor: likes is how historical models qualify.
        by_likes = list_local_chat_models(
            top_k=likes_k,
            min_downloads=0,
            sort_by="likes",
            known_attrs=known,
        )
        buckets.append(("likes", by_likes))

    merged: dict[str, dict] = {}
    for _, rows in buckets:
        for m in rows:
            merged.setdefault(m["id"], {"context_size": m["context_size"], "supports_tools": m["supports_tools"]})

    if not merged:
        raise RuntimeError("No models in result - check filters / network")

    fresh_ids = set(merged)
    pruned = [k for k in existing if k not in fresh_ids]
    new_ids = [i for i in fresh_ids if i not in existing]
    for label, rows in buckets:
        print(f"Top {len(rows)} by {label!r}")
    print(f"Union {len(merged)}: {len(new_ids)} new, {len(merged) - len(new_ids)} already known, {len(pruned)} pruned")

    sorted_ids = sorted(merged.keys(), key=str.lower)

    print(f"Writing {file.relative_to(here.parent)} with {len(sorted_ids)} models")
    name_lines = ',\n    '.join(f"'{i}'" for i in sorted_ids)
    attributes_lines = ',\n    '.join(
        f'''{f'"{i}":':<60}Attr(context_size={merged[i]["context_size"]}, supports_tools={merged[i]["supports_tools"]})'''
        for i in sorted_ids
    )
    file.write_text(f'''\
# DO NOT EDIT THIS FILE MANUALLY
# This file is generated by calling `_create_models_types_file()` in toki/local/fetch_models.py
# (or via the `toki-fetch-local-models` script).
#
# NOTE: this list is NEITHER EXHAUSTIVE NOR GUARANTEED CURRENT. It is a
# snapshot of HuggingFace chat models: `toki/local/curated.txt` (ids that
# pass the chat-template check) union the current top repos by 30-day
# downloads and by all-time likes. Many other valid chat models exist on
# the Hub; the local backend accepts any HF repo id, so passing a string
# outside this Literal still works. The Literal is here purely for IDE
# autocomplete on the popular / curated cases.

from typing import Literal
from dataclasses import dataclass


LocalModelName = Literal[
    {name_lines}
]


@dataclass
class Attr:
    context_size: int
    supports_tools: bool
    # TBD: may add more in the future


attributes_map: dict[LocalModelName, Attr] = {{
    {attributes_lines}
}}
''')
