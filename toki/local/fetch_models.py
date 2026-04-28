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
from tqdm import tqdm

# silence the per-file progress bars that hf_hub_download prints for tiny
# config / chat_template files - they completely clobber our outer tqdm.
disable_progress_bars()


here = Path(__file__).parent


# pipeline tags whose models work with toki's chat-message API
# (multimodal models can still be driven text-only)
DEFAULT_PIPELINE_TAGS = ("text-generation", "image-text-to-text")

# common keys for the model's max sequence length, in priority order
_CONTEXT_SIZE_KEYS = ("max_position_embeddings", "n_positions", "seq_length", "max_seq_len")

# HF sort modes:
#   - "downloads" is HF's ~30-day download count (NOT all-time; HF doesn't expose
#     an all-time sort), so this is already a recent-popularity signal.
#   - "trending_score" is a much shorter-window buzz signal that mixes downloads,
#     likes, and discussion velocity. Use this to surface brand-new hot models.
SortBy = Literal["downloads", "trending_score"]


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
    api = HfApi()
    seen: set[str] = set()
    candidates = []
    overshoot = top_k * 4  # popular list contains many base/non-chat repos, so over-fetch
    for tag in pipeline_tags:
        for m in api.list_models(
            pipeline_tag=tag,
            sort=sort_by,
            limit=overshoot,
        ):
            if m.id in seen:
                continue
            seen.add(m.id)
            if (m.downloads or 0) < min_downloads:
                continue
            score = m.downloads if sort_by == "downloads" else (getattr(m, "trending_score", 0) or 0)
            candidates.append((score, m))

    candidates.sort(key=lambda x: -x[0])

    verified: list[dict] = []
    with tqdm(total=top_k, desc="Verifying") as pbar:
        for _, m in candidates:
            if m.id in known_attrs:
                verified.append({"id": m.id, **known_attrs[m.id]})
            else:
                attrs = _fetch_model_attrs(m.id)
                if attrs is None:
                    continue
                verified.append({"id": m.id, **attrs})
            pbar.update(1)
            if len(verified) >= top_k:
                break

    return verified


def _fetch_model_attrs(repo_id: str) -> dict | None:
    """Fetch chat template + context size for a repo. Returns None if it isn't a chat model or required metadata is missing."""
    chat_template = _get_chat_template_text(repo_id)
    if chat_template is None:
        return None
    context_size = _get_context_size(repo_id)
    if context_size is None:
        return None
    return {
        "context_size": context_size,
        "supports_tools": "tools" in chat_template,
    }


def _get_chat_template_text(repo_id: str) -> str | None:
    try:
        files = list_repo_files(repo_id)
    except HfHubHTTPError:
        return None
    if "chat_template.jinja" in files:
        try:
            path = hf_hub_download(repo_id, "chat_template.jinja")
        except HfHubHTTPError:
            return None
        try:
            return Path(path).read_text()
        except OSError:
            return None
    if "tokenizer_config.json" not in files:
        return None
    try:
        cfg_path = hf_hub_download(repo_id, "tokenizer_config.json")
    except HfHubHTTPError:
        return None
    try:
        cfg = json.loads(Path(cfg_path).read_text())
    except (json.JSONDecodeError, OSError):
        return None
    template = cfg.get("chat_template")
    return template if isinstance(template, str) else None


def _get_context_size(repo_id: str) -> int | None:
    try:
        cfg_path = hf_hub_download(repo_id, "config.json")
    except HfHubHTTPError:
        return None
    try:
        cfg = json.loads(Path(cfg_path).read_text())
    except (json.JSONDecodeError, OSError):
        return None
    for key in _CONTEXT_SIZE_KEYS:
        v = cfg.get(key)
        if isinstance(v, int):
            return v
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


def _create_models_types_file(
    *,
    top_k: int = 100,
    min_downloads: int = 1000,
    sort_by: SortBy = "trending_score",
    file: PathLike = here / "models.py",
):
    """
    Dev helper: regenerate `toki/local/models.py` with a `LocalModelName` Literal
    of popular instruction-tuned chat models for IDE autocomplete.

    Pulls the *current* top `top_k` chat models from HF (under `sort_by`) and
    merges any not-yet-known ones into the existing snapshot. Existing entries
    are never removed (HF repos rarely disappear, and the snapshot is a growing
    curated set), so on a typical refresh the file may grow by 0..top_k models.
    """
    file = Path(file)
    existing = _load_existing_attrs(file)
    if existing:
        print(f"Loaded {len(existing)} existing entries from {file.relative_to(here.parent)}")

    top = list_local_chat_models(
        top_k=top_k,
        min_downloads=min_downloads,
        sort_by=sort_by,
        known_attrs=existing,
    )
    new_ids = [m["id"] for m in top if m["id"] not in existing]
    print(f"Top {len(top)} by {sort_by!r}: {len(new_ids)} new, {len(top) - len(new_ids)} already known")

    merged: dict[str, dict] = dict(existing)
    for m in top:
        if m["id"] not in merged:
            merged[m["id"]] = {"context_size": m["context_size"], "supports_tools": m["supports_tools"]}

    if not merged:
        raise RuntimeError("No models in result - check filters / network")

    sorted_ids = sorted(merged.keys(), key=str.lower)

    print(f"Writing {file.relative_to(here.parent)} with {len(sorted_ids)} models ({len(new_ids)} added this run)")
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
# NOTE: this list is NEITHER EXHAUSTIVE NOR GUARANTEED CURRENT. It is a curated
# snapshot of popular HuggingFace models that ship a tokenizer `chat_template`,
# accumulated across runs of the codegen script. Many other valid chat models
# exist on the Hub; the local backend accepts any HF repo id, so passing a
# string outside this Literal still works. The Literal is here purely for IDE
# autocomplete on the popular cases.

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
