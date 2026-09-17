"""
Internet access for the model: DuckDuckGo search and page fetching.

Stdlib only (urllib + html.parser), so no extra dependency or API key. Both
tools are read-only and available in every mode, but they send data off the
machine, so they can be switched off with `Permissions.web = False`.
"""

from __future__ import annotations

import html
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser

USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) toki-harness/1.0"
SEARCH_URL = "https://html.duckduckgo.com/html/"
MAX_PAGE_CHARS = 20_000
TIMEOUT = 20


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


class WebError(Exception):
    pass


def http_get(url: str, *, data: dict | None = None, timeout: int = TIMEOUT) -> tuple[str, str]:
    """Fetch `url` (POST if `data`). Returns (content_type, body_text)."""
    if not url.lower().startswith(("http://", "https://")):
        raise WebError(f"only http(s) URLs are allowed: {url}")
    body = urllib.parse.urlencode(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*;q=0.8"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            ctype = resp.headers.get("Content-Type", "")
            raw = resp.read(4_000_000)
    except urllib.error.HTTPError as e:
        raise WebError(f"HTTP {e.code} for {url}")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise WebError(f"could not fetch {url}: {getattr(e, 'reason', e)}")
    charset = "utf-8"
    m = re.search(r"charset=([\w-]+)", ctype)
    if m:
        charset = m.group(1)
    return ctype, raw.decode(charset, errors="replace")


# --- search -------------------------------------------------------------------

class _DDGParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[SearchResult] = []
        self._in_title = False
        self._in_snippet = False
        self._title: list[str] = []
        self._snippet: list[str] = []
        self._url = ""

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        cls = a.get("class", "") or ""
        if tag == "a" and "result__a" in cls:
            self._in_title = True
            self._title = []
            self._url = _unwrap_ddg(a.get("href", "") or "")
        elif tag == "a" and "result__snippet" in cls:
            self._in_snippet = True
            self._snippet = []

    def handle_endtag(self, tag):
        if tag == "a" and self._in_title:
            self._in_title = False
        elif tag == "a" and self._in_snippet:
            self._in_snippet = False
            if self._url:
                self.results.append(SearchResult("".join(self._title).strip(), self._url, "".join(self._snippet).strip()))
                self._url = ""

    def handle_data(self, data):
        if self._in_title:
            self._title.append(data)
        elif self._in_snippet:
            self._snippet.append(data)


def _unwrap_ddg(href: str) -> str:
    """DuckDuckGo wraps results as //duckduckgo.com/l/?uddg=<encoded url>&rut=..."""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urllib.parse.urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        q = urllib.parse.parse_qs(parsed.query)
        if "uddg" in q:
            return q["uddg"][0]
    return href


def parse_search_html(page: str) -> list[SearchResult]:
    p = _DDGParser()
    p.feed(page)
    return p.results


def search(query: str, *, max_results: int = 8) -> list[SearchResult]:
    _, page = http_get(SEARCH_URL, data={"q": query})
    return parse_search_html(page)[:max_results]


# --- page text ----------------------------------------------------------------

_SKIP = {"script", "style", "noscript", "svg", "head", "nav", "footer", "iframe", "template"}
_BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "pre", "section", "article", "blockquote", "hr", "table", "ul", "ol", "dd", "dt"}


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.title: list[str] = []
        self._skip = 0
        self._in_title = False
        self._in_pre = False

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        elif tag in _BLOCK:
            self.parts.append("\n")
            if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
                self.parts.append("#" * int(tag[1]) + " ")
            elif tag == "li":
                self.parts.append("- ")
        if tag == "pre":
            self._in_pre = True

    def handle_endtag(self, tag):
        if tag in _SKIP and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in _BLOCK and tag != "li":
            self.parts.append("\n")
        if tag == "pre":
            self._in_pre = False

    def handle_data(self, data):
        if self._in_title:
            self.title.append(data)
        elif not self._skip:
            self.parts.append(data if self._in_pre else re.sub(r"\s+", " ", data))


def html_to_text(page: str) -> tuple[str, str]:
    """Returns (title, readable text) for an HTML page."""
    p = _TextParser()
    p.feed(page)
    text = html.unescape("".join(p.parts))
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"^\s*(?:-|#+)\s*$\n?", "", text, flags=re.MULTILINE)   # markers from empty <li>/<hN>
    text = re.sub(r"(?:^|\s)(?:- ){2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return "".join(p.title).strip(), text


def fetch(url: str, *, max_chars: int = MAX_PAGE_CHARS) -> tuple[str, str]:
    ctype, body = http_get(url)
    if "html" in ctype or body.lstrip()[:200].lower().startswith(("<!doctype html", "<html")):
        title, text = html_to_text(body)
    else:
        title, text = "", body
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n\n... (truncated; {len(text) - max_chars} more chars)"
    return title, text
