"""
Shared fakes used by the 2FA test suite.

The goal is to exercise the runner's state machine and discovery logic
without requiring a real browser, OpenRouter key, or Google account.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional


class FakeElement:
    """Mimics enough of nodriver's Element for click/send_keys/select tests."""

    def __init__(self, page: "FakePage", selector: str,
                 text: str = "", tag: str = "div"):
        self.page = page
        self.selector = selector
        self._text = text
        self.tag = tag
        self.clicked = False
        self.sent_keys: List[str] = []

    async def click(self) -> None:
        self.clicked = True
        self.page.click_log[self.selector] = self.page.click_log.get(self.selector, 0) + 1
        # When the chooser entry is clicked, transition page state.
        await self.page._on_click(self.selector)

    async def send_keys(self, text: str) -> None:
        self.sent_keys.append(text)


class FakePage:
    """
    Mimics the nodriver Page surface used by runner.py and healer.py:
      - `url` (str attribute)
      - `await page.evaluate(js, return_by_value=True)`
      - `await page.select(selector, timeout=...)`
      - `await page.query_selector(selector)`
      - `await page.sleep(seconds)`
      - `await page.get_content()`

    Behaviour is driven by a controllable state model: you can change
    `url`, `body_text`, `selectors`, and `chooser_visible` to simulate
    navigation between password → chooser → prompt → authenticated pages.
    """

    def __init__(self, url: str = "about:blank", body_text: str = "",
                 selectors: Optional[Dict[str, str]] = None,
                 chooser_visible: bool = False):
        self.url = url
        self.body_text = body_text
        self.selectors: Dict[str, str] = dict(selectors or {})
        # sel -> visible-element flag. By default all registered selectors exist.
        self.selector_hidden: set = set()
        self.chooser_visible = chooser_visible
        self.click_log: Dict[str, int] = {}
        self.evaluate_calls: List[str] = []
        # Allows a test to inject arbitrary click handlers to simulate
        # page navigation (e.g. clicking the chooser entry moves URL
        # from challenge/selection → challenge/prompt).
        self._click_hook: Optional[Callable[[str], Awaitable[None]]] = None
        # Force this many retries before evaluate("location.href") reflects
        # the next url. Mirrors real-life navigation delay.
        self.url_transition_remaining: int = 0
        self._pending_url: Optional[str] = None

    # ----- state mutation helpers used by tests -----
    def set_url(self, url: str) -> None:
        self.url = url

    def set_body_text(self, text: str) -> None:
        self.body_text = text

    def set_chooser_visible(self, v: bool) -> None:
        self.chooser_visible = v

    def register_selector(self, selector: str, text: str = "",
                           hidden: bool = False) -> None:
        self.selectors[selector] = text
        if hidden:
            self.selector_hidden.add(selector)
        else:
            self.selector_hidden.discard(selector)

    def click_transitions_to(self, new_url: Optional[str] = None,
                             new_body: Optional[str] = None,
                             chooser: Optional[bool] = None,
                             after_n_polls: int = 0) -> None:
        """Make a click event update URL/body/chooser after `after_n_polls`
        `observe_auth_state` calls. Use this to simulate post-click navigation
        latency."""
        async def hook(_sel: str) -> None:
            self._pending_url = new_url
            self._pending_body = new_body
            self._pending_chooser = chooser
            self.url_transition_remaining = after_n_polls + 1

        self._click_hook = hook

    def commit_pending_state(self) -> None:
        """Apply any pending state. Tests may call this directly; the
        `observe_auth_state` call path also auto-decrements."""
        if self._pending_url is not None:
            self.url = self._pending_url
            self._pending_url = None
        if getattr(self, "_pending_body", None) is not None:
            self.body_text = self._pending_body
            self._pending_body = None
        if getattr(self, "_pending_chooser", None) is not None:
            self.chooser_visible = self._pending_chooser
            self._pending_chooser = None

    async def _on_click(self, selector: str) -> None:
        if self._click_hook is not None:
            await self._click_hook(selector)

    # ----- nodriver surface -----
    async def sleep(self, seconds: float) -> None:
        # Don't actually wait — tests should run fast. Tests that need
        # time-based behaviour should manipulate state directly.
        await asyncio.sleep(0)
        return None

    async def get_content(self) -> str:
        return f"<html><body>{self.body_text}</body></html>"

    @staticmethod
    def _normalize_selector(selector: str) -> str:
        """Canonicalise selector variants the fake should treat as equal.

        Browsers accept `li[data-challengetype='13']` and
        `[data-challengetype="13"]` interchangeably, so the fake must too —
        otherwise tests would over-specify selector string formatting.
        """
        if not selector:
            return ""
        out = selector.replace("'", '"').strip()
        # Strip a leading "li " or "li>" prefix from attribute selectors so
        # `li[data-challengetype="13"]` matches `[data-challengetype="13"]`.
        if out.startswith("li[") and "]" in out:
            out = out[2:]
        return out

    def _matched_selector(self, selector: str) -> Optional[str]:
        """Return the registered key that matches `selector`, ignoring
        quote-style and a leading `li` on attribute selectors."""
        if selector in self.selector_hidden:
            return None
        if selector in self.selectors:
            return selector
        target = self._normalize_selector(selector)
        for key in self.selectors.keys():
            if self._normalize_selector(key) == target:
                return key
        return None

    async def query_selector(self, selector: str) -> Optional[FakeElement]:
        matched = self._matched_selector(selector)
        if matched is None:
            return None
        return FakeElement(self, matched, self.selectors[matched])

    async def select(self, selector: str, timeout: float = 8) -> Optional[FakeElement]:
        # Tests don't actually wait — return immediately to keep them fast.
        return await self.query_selector(selector)

    async def evaluate(self, js: str, return_by_value: bool = True) -> Any:
        self.evaluate_calls.append(js)
        # Text-finder JS first — it contains `querySelectorAll(sel).length`
        # for uniqueness checks, so it MUST be detected before the generic
        # count-query branch below.
        if "querySelectorAll(sel)" in js and "innerText" in js:
            # The JS embeds `const TEXT = __TEXT__;` and `__TEXT__` was
            # substituted with a JSON-quoted string. Pull the value out.
            m = re.search(r"const TEXT = \"([^\"]*)\"", js)
            if m:
                target = m.group(1).lower()
                best_sel = None
                best_len = float("inf")
                for sel, txt in self.selectors.items():
                    if sel in self.selector_hidden:
                        continue
                    if target and target in txt.lower():
                        if len(txt) < best_len:
                            best_len = len(txt)
                            best_sel = sel
                return best_sel
            return None
        # Chooser probe — exact JS identifier.
        if "querySelectorAll('[data-challengetype]')" in js:
            return self.chooser_visible
        # location.href / location lookup
        if "location.href" in js or js.strip().startswith("location.href") \
                or "() => location.href" in js:
            return self.url
        # body innerText probe
        if "document.body.innerText" in js or "document.body &&" in js:
            return self.body_text[:4000]
        # querySelectorAll(<sel>).length
        m = re.search(r"document\.querySelectorAll\((.+?)\)\.length", js)
        if m:
            try:
                sel = json.loads(m.group(1))
            except Exception:
                return 0
            if self._matched_selector(sel) is None:
                return 0
            return 1
        # document.querySelector(<sel>) -> getElementText probe
        m = re.search(r'document\.querySelector\("(.+?)"\)', js)
        if m:
            sel = m.group(1)
            matched = self._matched_selector(sel)
            if matched is None:
                return ""
            return self.selectors[matched]
        return None
