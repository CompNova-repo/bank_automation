"""
healer.py — selector discovery and DOM inspection helpers.

Provides:
- `extract_interactive_dom` — DOM snapshot used as context for LLM selector repair.
- `selector_exists` / `get_element_text` / `get_page_url` — small DOM probes.
- `scroll_through_page` — forces lazy-loaded elements into the DOM.
- `find_selector_by_text` — deterministic text-target selector finder.
- `populate_and_save_selector` / `populate_and_save_text_target` — discovery
  pipelines that fall back to the LLM and only persist *validated* selectors.
- `ask_openrouter_candidates` / `ask_openrouter_candidates_text` — LLM callers.
- `heal_and_update_config` — selector-repair helper invoked by the runner.

Design intent:
    ACTION execution  (run by runner.py)
        vs.
    STATE observation  (run by runner.py via state_machine.py helpers)
        vs.
    SELECTOR healing   (this module).

Discovery distinguishes between three terminal outcomes that the runner
needs to disambiguate:

    FOUND             — a validated selector was returned.
    EMPTY_GENUINE     — a chooser is genuinely not present
                        (e.g. Google auto-sent the prompt, or auth
                        already completed).
    ERROR             — the LLM/network failed, or no candidate
                        validated against the live DOM. We do NOT
                        treat this as "nothing to click".

The previous implementation conflated ERROR with EMPTY_GENUINE, which is
the source of the observed `challenge/pwd → empty selector → timeout`
regression.
"""

import os
import json
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL_NAME = os.environ.get("HEALER_MODEL", "qwen/qwen3.7-flash")

# Bounded retry/backoff for transient OpenRouter failures (429 / 5xx /
# connection errors). The browser workflow does not depend on this — it
# prefers deterministic DOM discovery — but we still try a few times
# before giving up so a flaky key does not falsely look like "no candidates".
OPENROUTER_MAX_ATTEMPTS = int(os.environ.get("OPENROUTER_MAX_ATTEMPTS", "3"))
OPENROUTER_BACKOFF_SECONDS = float(os.environ.get("OPENROUTER_BACKOFF_SECONDS", "2.0"))

# Verbose / debug logging for the LLM helpers. When enabled, every raw LLM
# reply is dumped (untruncated) and `_diagnose_llm_response` runs sanity
# checks (refusal phrases, hallucinated method names, broken JSON syntax).
# Useful when chasing "automation silently ended on password page" type bugs.
HEALER_DEBUG = os.environ.get("HEALER_DEBUG", "").strip().lower() in (
    "1", "true", "yes", "on",
)

# Phrases that strongly suggest the model is refusing rather than returning
# data. Kept conservative — these are the patterns we have actually seen from
# qwen/qwen3.7-flash when it cannot decide what to click.
_REFUSAL_PHRASES = (
    "i cannot decide",
    "i can't decide",
    "i cannot choose",
    "i can't choose",
    "i am unable",
    "i'm unable",
    "as an ai",
    "i do not have enough information",
    "insufficient information",
)

# Method/action names that have appeared in hallucinated outputs from smaller
# models. None of these are valid CSS selectors — when one shows up the
# caller knows the LLM response is junk and should not be persisted.
_INVALID_METHOD_TOKENS = (
    "fix_it_with_magic",
    "do_magic",
    "use_magic",
    "auto_fix",
    "magic_select",
    "fix_all",
    "click_any",
)


class DiscoveryStatus(str, Enum):
    """Result categories returned by `populate_and_save_text_target`."""

    FOUND = "found"             # Validated selector returned and persisted.
    EMPTY_GENUINE = "empty"     # No chooser visible. LLM (if called) confirmed it.
    ERROR = "error"             # LLM/network/parse failure, or no candidate validated.
    NOT_READY = "not_ready"     # The page has not yet reached the expected state.


@dataclass
class DiscoveryResult:
    """Outcome of a discovery attempt.

    `selector` is only meaningful when `status == FOUND`.
    """

    status: DiscoveryStatus
    selector: Optional[str] = None
    detail: str = ""


# ---------------------------------------------------------------------------
# DOM inspection
# ---------------------------------------------------------------------------

async def extract_interactive_dom(page, limit: int = 200):
    """
    Extracts visible interactive elements with enough context to ground
    selector repair. Includes list items, buttons, links, and any element
    carrying stable identifying attributes (data-*, aria-*, role, jsname,
    name, id). No hardcoded domain knowledge — the LLM does the reasoning.
    """
    js_extract = """
    () => {
        const els = Array.from(document.querySelectorAll(
            'input, button, a, [role="button"], [role="listitem"], [role="link"], [role="menuitem"], [tabindex], div[id], span[id], li, [jsname], [data-testid], [data-challengetype], [data-id], [aria-label], [name]'
        ));
        const out = [];
        for (const el of els) {
            const r = el.getBoundingClientRect ? el.getBoundingClientRect() : {width: 0, height: 0, x: 0, y: 0};
            const style = window.getComputedStyle ? window.getComputedStyle(el) : null;
            const visible = r.width > 0 && r.height > 0 && (!style || (style.visibility !== 'hidden' && style.display !== 'none'));
            const parentWithId = el.closest ? (el.closest('div[id], span[id], form[id]') || null) : null;
            const attrs = {};
            if (el.attributes) {
                for (const a of Array.from(el.attributes)) {
                    if (!a || !a.name) continue;
                    if (a.name.indexOf('data-') === 0 || a.name.indexOf('aria-') === 0 ||
                        ['role','name','jsname','id','type','placeholder','href'].includes(a.name)) {
                        const v = a.value || '';
                        // Skip empty / giant hashed classes / styles.
                        if (!v) continue;
                        if (a.name === 'class' && /[A-Za-z0-9_-]{20,}/.test(v)) continue;
                        attrs[a.name] = v.length > 120 ? v.slice(0, 120) : v;
                    }
                }
            }
            out.push({
                tag: el.tagName ? el.tagName.toLowerCase() : '',
                id: el.id || '',
                parentId: (parentWithId && parentWithId !== el) ? parentWithId.id : '',
                innerText: ((el.innerText || el.textContent || '').trim()).slice(0, 160),
                visible: visible,
                attrs: attrs,
                x: Math.round(r.x || 0),
                y: Math.round(r.y || 0)
            });
        }
        out.sort((a, b) => (b.visible - a.visible));
        return out.filter(item =>
            item.visible && (
                item.id || item.parentId || item.innerText ||
                Object.keys(item.attrs).length > 0
            )
        ).slice(0, LIMIT);
    }
    """.replace("LIMIT", str(limit))
    try:
        elements_data = await page.evaluate(js_extract, return_by_value=True)
        # nodriver evaluate may return RemoteObject wrapper; normalize
        if isinstance(elements_data, dict) and "value" in elements_data:
            elements_data = elements_data["value"]
        if not isinstance(elements_data, list):
            elements_data = []
        return json.dumps(elements_data, indent=2)
    except Exception:
        try:
            html = await page.get_content()
            return html[:4000]
        except Exception:
            return "[]"


def build_deterministic_candidates(step_name: str, failed_selector: str):
    """
    Cheap local guesses tried BEFORE spending an LLM call.
    Covers the common Google-login case and generic Next/Submit buttons.
    Intentionally does NOT hardcode 2FA selectors — the LLM is responsible for
    figuring those out from the live DOM context.
    """
    name = (step_name or "").lower()
    cands = []
    if "password" in name and "next" in name:
        cands += ["#passwordNext", "#passwordNext button", "[jsname='V67aGc']"]
    if "identifier" in name or ("email" in name and "next" in name):
        cands += ["#identifierNext", "#identifierNext button"]
    if "next" in name:
        cands += ["#passwordNext", "#passwordNext button", "#identifierNext", "#identifierNext button"]
    if "submit" in failed_selector or "submit" in name:
        cands += ["button[type='submit']", "button", "[role='button']"]
    # de-dup, keep order
    seen, out = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


async def selector_exists(page, selector: str) -> bool:
    """Fast non-waiting existence check via query_selector."""
    try:
        el = await page.query_selector(selector)
        return el is not None
    except Exception:
        return False


def _parse_candidates(raw: str):
    """Accept either a JSON array or a lone selector string."""
    raw = (raw or "").strip().replace("`", "").strip()
    # Try JSON array first
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(s).strip().strip('"').strip("'") for s in data if str(s).strip()]
        if isinstance(data, dict):
            for key in ("selectors", "candidates"):
                if isinstance(data.get(key), list):
                    return [str(s).strip() for s in data[key] if str(s).strip()]
    except Exception:
        pass
    # Fall back: one selector per line, first non-empty wins as list
    lines = [ln.strip().strip('"').strip("'").rstrip(",") for ln in raw.splitlines() if ln.strip()]
    return [ln for ln in lines if ln][:5] or ([raw] if raw else [])


# ---------------------------------------------------------------------------
# Text-targeted helpers — used for the 2FA method chooser which has no stable id
# ---------------------------------------------------------------------------

def _build_text_finder_js(text: str, match_type: str, tag_hint: str) -> str:
    """JS that scans the DOM for a visible element whose text contains `text`.

    Returns a unique CSS selector for the best match, or null. Strategy:
    scan multiple tag types (li, div, span, button, a, role=button, etc.),
    pick the most specific match (shortest innerText containing the target),
    build a selector preferring #id / unique data-* / aria-* attrs, falling
    back to a :nth-of-type path. No hardcoded domain knowledge.
    """
    safe_text = json.dumps(text)
    safe_match = json.dumps(match_type or "contains")
    safe_tag = json.dumps(tag_hint or "")
    return """
    () => {
        const TEXT = __TEXT__;
        const MATCH = __MATCH__;
        const TAG = __TAG__;
        const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim().toLowerCase();
        const target = norm(TEXT);
        if (!target) return null;
        const isVisible = (el) => {
            const r = el.getBoundingClientRect();
            if (!r || r.width === 0 || r.height === 0) return false;
            const s = window.getComputedStyle(el);
            return s.visibility !== 'hidden' && s.display !== 'none';
        };
        // Scan broadly — Google uses li/div for 2FA items, and the tag_hint
        // is only a preference.
        const tagSet = TAG
            ? [TAG, 'li', 'div', 'span', 'button', 'a', '[role="button"]', '[role="listitem"]']
            : ['li', 'div', 'span', 'button', 'a', '[role="button"]', '[role="listitem"]'];
        const seen = new Set();
        let best = null;
        let bestLen = Infinity;
        for (const sel of tagSet) {
            const els = Array.from(document.querySelectorAll(sel));
            for (const el of els) {
                if (!isVisible(el)) continue;
                const tx = norm(el.innerText || el.textContent || '');
                const ok = MATCH === 'exact' ? (tx === target) : (tx.indexOf(target) !== -1);
                if (!ok) continue;
                if (tx.length >= bestLen) continue; // prefer smaller, more specific match
                best = el;
                bestLen = tx.length;
            }
        }
        if (!best) return null;
        const el = best;
        if (el.id) return '#' + el.id;
        // Prefer unique data-* / aria-* / role / name / jsname
        const attrPriority = ['data-challengetype', 'data-id', 'data-testid', 'aria-label', 'role', 'jsname', 'name'];
        for (const an of attrPriority) {
            const v = el.getAttribute && el.getAttribute(an);
            if (v) {
                const sel = '[' + an + '="' + v.replace(/"/g, '\\\\"') + '"]';
                try { if (document.querySelectorAll(sel).length === 1) return sel; } catch (e) {}
            }
        }
        // Fallback: nth-of-type path
        const parts = [];
        let cur = el;
        while (cur && cur.tagName && parts.length < 6) {
            let part = cur.tagName.toLowerCase();
            if (cur.id) { parts.unshift('#' + cur.id); break; }
            const parent = cur.parentElement;
            if (parent) {
                const sibs = Array.from(parent.children).filter(c => c.tagName === cur.tagName);
                if (sibs.length > 1) {
                    const idx = sibs.indexOf(cur) + 1;
                    part += ':nth-of-type(' + idx + ')';
                }
            }
            parts.unshift(part);
            cur = parent;
        }
        return parts.join(' > ');
    }
    """.replace("__TEXT__", safe_text).replace("__MATCH__", safe_match).replace("__TAG__", safe_tag)


async def find_selector_by_text(page, text: str, match_type: str = "contains",
                                tag_hint: str = "li") -> Optional[str]:
    """Run the text-finder JS and return the CSS selector it picks, or None."""
    if not text:
        return None
    js = _build_text_finder_js(text, match_type, tag_hint)
    try:
        result = await page.evaluate(js, return_by_value=True)
        if isinstance(result, dict) and "value" in result:
            result = result["value"]
        if isinstance(result, str) and result.strip():
            return result.strip()
    except Exception:
        return None
    return None


async def get_element_text(page, selector: str) -> str:
    """Return the visible innerText of the matched element (best-effort)."""
    if not selector:
        return ""
    safe = selector.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    js = (
        "() => {"
        "try {"
        "  const el = document.querySelector(\"" + safe + "\");"
        "  if (!el) return '';"
        "  return (el.innerText || el.textContent || '').trim();"
        "} catch (e) { return ''; }"
        "}"
    )
    try:
        result = await page.evaluate(js, return_by_value=True)
        if isinstance(result, dict) and "value" in result:
            result = result["value"]
        return str(result or "")
    except Exception:
        return ""


async def get_page_url(page) -> str:
    """Return the current page URL as a string (best-effort)."""
    try:
        url_attr = getattr(page, "url", None)
        if isinstance(url_attr, str):
            return url_attr
    except Exception:
        pass
    try:
        result = await page.evaluate("() => location.href", return_by_value=True)
        if isinstance(result, dict) and "value" in result:
            result = result["value"]
        return str(result or "")
    except Exception:
        return ""


async def scroll_through_page(page, step_px: int = 400, max_steps: int = 8) -> None:
    """
    Scroll the page top-to-bottom in `step_px` increments to force lazy-loaded
    elements (e.g. off-screen 2FA options) into the DOM. Then scroll back to top.
    """
    js = """
    () => new Promise(resolve => {
        try {
            let y = 0;
            const step = STEP;
            const max = MAX;
            let i = 0;
            function tick() {
                window.scrollTo(0, y);
                y += step;
                i += 1;
                if (i < max) {
                    setTimeout(tick, 80);
                } else {
                    window.scrollTo(0, 0);
                    setTimeout(resolve, 200);
                }
            }
            tick();
        } catch (e) { resolve(); }
    })
    """.replace("STEP", str(step_px)).replace("MAX", str(max_steps))
    try:
        await page.evaluate(js, return_by_value=True)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Chooser-visible probe — used to distinguish "chooser is here" from
# "prompt already auto-sent". Both look like an idle page to a naive caller.
# ---------------------------------------------------------------------------

CHOOSER_PROBE_JS = """
() => {
    // Google wraps 2FA method options as <li data-challengetype="..."> elements.
    // The chooser is the only place these appear together in meaningful numbers.
    const items = document.querySelectorAll('[data-challengetype]');
    if (items.length >= 2) {
        const visible = Array.from(items).filter(el => {
            const r = el.getBoundingClientRect();
            return r.width > 0 && r.height > 0;
        });
        if (visible.length >= 2) return true;
    }
    // Fallback: scan for a known chooser heading.
    const headings = Array.from(document.querySelectorAll('h1, h2, div'))
        .map(el => (el.innerText || '').trim().toLowerCase());
    if (headings.some(t => t.indexOf('2-step verification') !== -1 ||
                           t.indexOf('choose how') !== -1 ||
                           t.indexOf('pick a way') !== -1 ||
                           t.indexOf('select a method') !== -1)) {
        return true;
    }
    return false;
}
"""


async def is_chooser_visible(page) -> bool:
    """Return True if the 2FA method chooser is on screen right now."""
    try:
        result = await page.evaluate(CHOOSER_PROBE_JS, return_by_value=True)
        if isinstance(result, dict) and "value" in result:
            result = result["value"]
        return bool(result)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Validation helpers — shared by discovery + healing paths
# ---------------------------------------------------------------------------

async def _validate_selector_against_target(page, selector: str, text: str,
                                            match_type: str = "contains") -> bool:
    """True iff the selector matches exactly one visible element whose text
    contains the target (case-insensitive). When `text` is empty, only the
    one-match requirement is enforced."""
    if not selector:
        return False
    if not await selector_exists(page, selector):
        return False
    try:
        count_js = "() => document.querySelectorAll(" + json.dumps(selector) + ").length"
        n = await page.evaluate(count_js, return_by_value=True)
        if isinstance(n, dict) and "value" in n:
            n = n["value"]
        if isinstance(n, (int, float)) and n != 1:
            return False
    except Exception:
        pass
    if not text:
        return True
    elem_text = await get_element_text(page, selector)
    norm = lambda s: (s or "").replace("\s+", " ").strip().lower()
    if match_type == "exact":
        return norm(elem_text) == norm(text)
    return norm(text) in norm(elem_text)


# ---------------------------------------------------------------------------
# Text-targeted discovery pipeline
# ---------------------------------------------------------------------------

async def populate_and_save_text_target(
    page, config_path: str, step_index: int,
    text: Optional[str] = None, match_type: Optional[str] = None,
    tag_hint: Optional[str] = None,
) -> DiscoveryResult:
    """
    Discovers a text-targeted element (e.g. a 2FA method chooser entry),
    persists a CSS selector for it into config.json, and returns a
    DiscoveryResult describing the outcome.

    Pipeline:
      0. Scroll through the page so off-screen elements enter the DOM.
      1. JS text-finder scan (free, instant) — pick most-specific text match.
      2. LLM call with full DOM + intent + page URL.
      3. Validate each candidate against the live DOM (one element, text overlap).
      4. Empty-list from LLM + no chooser visible -> EMPTY_GENUINE (no persist).
      5. LLM failure / no validated candidate -> ERROR (no persist).

    Caller-supplied `text` / `match_type` / `tag_hint` take precedence over
    whatever is on the step in config.json. This fixes the propagation bug
    where `runner.py`'s `resolve_2fa_text()` computed values never reached
    this function.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    step = config["flow_steps"][step_index]
    intent = step.get("intent", step.get("step_name", ""))
    # Caller overrides win; fall back to step fields.
    if text is None:
        text = step.get("text", "") or ""
    if match_type is None:
        match_type = step.get("match_type", "contains") or "contains"
    if tag_hint is None or tag_hint == "":
        tag_hint = step.get("tag_hint", "") or ""

    def _persist(sel):
        step["selector"] = sel
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    url = await get_page_url(page)
    print(f"\n[AI Discovery] Page URL: {url}")
    print(f"[AI Discovery] Searching for element matching intent: '{intent[:120]}' "
          f"(text hint='{text}', tag_hint='{tag_hint}')...")

    # 0) Scroll so lazy-loaded / off-screen elements are rendered.
    await scroll_through_page(page)

    # 1) JS scanner (cheap path)
    discovered = await find_selector_by_text(page, text, match_type, tag_hint)
    if discovered and await _validate_selector_against_target(
            page, discovered, text, match_type):
        print(f"[AI Discovery] Text-scan match: '{discovered}'")
        _persist(discovered)
        return DiscoveryResult(DiscoveryStatus.FOUND, discovered)

    chooser_visible = await is_chooser_visible(page)

    # 2) LLM with full DOM + URL context. Distinguish error vs empty.
    candidates: List[str] = []
    llm_err: Optional[str] = None
    llm_called = False
    try:
        dom = await extract_interactive_dom(page)
        candidates = ask_openrouter_candidates_text(
            dom, intent, text, match_type, tag_hint, url)
        llm_called = True
    except DiscoveryError as e:
        llm_err = str(e)
        print(f"[AI Healer] LLM text-candidate call failed: {llm_err}")
    except Exception as e:  # defensive — never crash the runner on LLM hiccups
        llm_err = repr(e)
        print(f"[AI Healer] LLM text-candidate call raised: {llm_err}")

    print(f"[AI Healer] LLM text-candidates: {candidates}")

    for cand in candidates:
        if await _validate_selector_against_target(page, cand, text, match_type):
            print(f"[AI Healer] Repaired selector: '{cand}' (validated)")
            _persist(cand)
            return DiscoveryResult(DiscoveryStatus.FOUND, cand)

    # 3) Lenient: token overlap with target text.
    for cand in candidates:
        if await selector_exists(page, cand):
            elem_text = (await get_element_text(page, cand)).lower()
            target = (text or "").lower().strip()
            target_tokens = {t for t in target.split() if len(t) > 2}
            elem_tokens = {t for t in elem_text.split() if len(t) > 2}
            overlap = len(target_tokens & elem_tokens)
            if target_tokens and overlap / len(target_tokens) >= 0.5:
                print(f"[AI Healer] Lenient repair: '{cand}' "
                      f"(text overlap {overlap}/{len(target_tokens)})")
                _persist(cand)
                return DiscoveryResult(DiscoveryStatus.FOUND, cand)

    # 4) Empty-list from a successful LLM AND no chooser visible:
    #    genuinely nothing to click right now (e.g. auto-sent prompt).
    if not candidates and llm_called and not llm_err and not chooser_visible:
        print(f"[AI Discovery] LLM returned [] and chooser not visible — "
              f"no selection required. Not persisting.")
        return DiscoveryResult(DiscoveryStatus.EMPTY_GENUINE, detail="no chooser / no candidates")

    # 5) Chooser visible but we could not validate anything: that's an error,
    #    not a genuine "nothing to click" — preserve any stored selector and
    #    surface the failure to the caller.
    if chooser_visible and (not candidates or llm_err):
        detail = (f"chooser visible but discovery failed "
                  f"(llm_err={llm_err or 'no-candidate-validated'})")
        print(f"[AI Healer] {detail}; not persisting empty selector.")
        return DiscoveryResult(DiscoveryStatus.ERROR, detail=detail)

    # 6) No candidates, no chooser, and no LLM call happened — page state
    #    is just not ready yet. Tell the caller to keep waiting.
    if not candidates and not llm_called and not discovered:
        return DiscoveryResult(DiscoveryStatus.NOT_READY,
                               detail="page still transitioning")

    # 7) Last resort: if we got an unvalidated JS-scan guess, don't persist it
    #    unless it's the only thing we have AND the chooser is visible.
    if discovered and chooser_visible:
        print(f"[AI Healer] Persisting unvalidated JS-scan selector "
              f"'{discovered}' (chooser visible).")
        _persist(discovered)
        return DiscoveryResult(DiscoveryStatus.FOUND, discovered)

    if llm_err:
        return DiscoveryResult(DiscoveryStatus.ERROR, detail=llm_err)

    return DiscoveryResult(DiscoveryStatus.ERROR,
                           detail="no validated candidate")


# ---------------------------------------------------------------------------
# OpenRouter LLM callers — with bounded retry/backoff and explicit error type
# ---------------------------------------------------------------------------

class DiscoveryError(RuntimeError):
    """The LLM transport failed or returned an unparsable response.

    Distinct from a successful empty `[]` answer, which means the model
    believes there is nothing appropriate to click.
    """


def _post_with_retry(payload: dict, headers: dict, label: str) -> dict:
    """POST to OpenRouter with bounded retry on transient errors.

    Retries on HTTP 429 and 5xx, plus connection/timeout errors. Treats
    non-transient HTTP failures (400/401/403/404) as immediate DiscoveryError.
    Returns parsed JSON; raises DiscoveryError otherwise.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    last_err: Optional[Exception] = None
    for attempt in range(1, OPENROUTER_MAX_ATTEMPTS + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=30)
        except requests.exceptions.Timeout as e:
            last_err = e
            print(f"[AI Healer] {label}: timeout (attempt {attempt}/"
                  f"{OPENROUTER_MAX_ATTEMPTS})")
        except requests.exceptions.ConnectionError as e:
            last_err = e
            print(f"[AI Healer] {label}: connection error (attempt {attempt}/"
                  f"{OPENROUTER_MAX_ATTEMPTS})")
        else:
            if response.status_code == 429 or 500 <= response.status_code < 600:
                last_err = requests.exceptions.HTTPError(
                    f"{response.status_code} {response.text[:120]}"
                )
                print(f"[AI Healer] {label}: transient HTTP "
                      f"{response.status_code} (attempt {attempt}/"
                      f"{OPENROUTER_MAX_ATTEMPTS})")
            elif response.status_code >= 400:
                # Non-transient — don't retry.
                raise DiscoveryError(
                    f"{label} HTTP {response.status_code}: {response.text[:200]}"
                )
            else:
                try:
                    return response.json()
                except Exception as e:
                    raise DiscoveryError(f"{label} returned non-JSON response: {e}")

        if attempt < OPENROUTER_MAX_ATTEMPTS:
            # Exponential backoff with a small jitter.
            delay = OPENROUTER_BACKOFF_SECONDS * (2 ** (attempt - 1))
            time.sleep(delay)

    raise DiscoveryError(
        f"{label} failed after {OPENROUTER_MAX_ATTEMPTS} attempts: {last_err}"
    )


def _diagnose_llm_response(content: str, label: str) -> None:
    """Verbose dump + sanity checks for a raw LLM reply.

    Always silent when HEALER_DEBUG is off. When HEALER_DEBUG is on:
      * Prints the FULL untruncated response so you can see exactly what the
        model emitted (the per-caller "no parsable selectors" log line is
        truncated to 300/400 chars).
      * Flags well-known refusal patterns ("I cannot decide", ...).
      * Flags hallucinated method names ("method": "fix_it_with_magic", ...).
      * Flags outputs that look like JSON but fail to parse.

    This is the entry point to check when chasing "automation ends silently
    on the password page" type bugs — the raw LLM reply is dumped here.
    """
    if not HEALER_DEBUG:
        return
    if content is None:
        print(f"[AI Healer][DEBUG] === Raw LLM response ({label}) === "
              f"<NoneType> ===")
        return
    print(f"\n[AI Healer][DEBUG] === Raw LLM response ({label}) "
          f"len={len(content)} ===")
    print(content)
    print(f"[AI Healer][DEBUG] === End raw response ({label}) ===\n")

    lower = (content or "").lower()
    # Refusal detection.
    for phrase in _REFUSAL_PHRASES:
        if phrase in lower:
            print(f"[AI Healer][DEBUG] WARN: refusal phrase detected "
                  f"({phrase!r}). Model is declining rather than picking a "
                  f"selector. Prompt may need more grounding.")
            break
    # Hallucinated method/action names.
    for tok in _INVALID_METHOD_TOKENS:
        if tok in lower:
            print(f"[AI Healer][DEBUG] WARN: hallucinated token {tok!r} "
                  f"present in response. Not a valid CSS selector.")
    if '"method"' in lower and 'magic' in lower:
        print(f"[AI Healer][DEBUG] WARN: response contains a "
                  f"'method'-shaped JSON field that looks fabricated.")
    # JSON syntax sanity check. Only run when the output *looks* like JSON
    # so we do not nag at the (perfectly fine) bare-selector responses that
    # this codebase's prompts deliberately produce: [#id, #id, ...] with no
    # quoting. A response that contains `"key":` style fields is genuinely
    # trying to be JSON and should be parsed cleanly.
    stripped = content.strip().strip("`")
    looks_like_json = (
        stripped.startswith("{")
        or ('"' in stripped and (stripped.startswith("[") or '":' in stripped))
    )
    if looks_like_json:
        try:
            json.loads(stripped)
        except Exception as e:
            print(f"[AI Healer][DEBUG] WARN: output looks like JSON but "
                  f"failed to parse: {e}")


def ask_openrouter_candidates(dom_context: str, step_name: str, failed_selector: str,
                              page_url: str = "") -> List[str]:
    if not OPENROUTER_API_KEY:
        raise DiscoveryError("OPENROUTER_API_KEY environment variable is missing.")

    prompt = f"""You are an expert browser automation engineer repairing a broken CSS selector.

Page URL: {page_url or "(unknown)"}
Failed step intent: "{step_name}"
Broken selector: "{failed_selector}"

Live DOM (each entry shows tag, id, parentId, attrs dict with data-*, aria-*, role, jsname, name keys, and innerText snippet):
---
{dom_context}
---

Rules:
1. Target the element matching "{step_name}".
2. Ground EVERY selector in attributes you can see in the DOM list. Do NOT invent ids or hashed classes.
3. Prefer stable selectors in this order: #id, [data-testid="..."], [data-challengetype="..."], [data-id="..."], [aria-label="..."], [role="..."], [jsname="..."], [name="..."], then #parentId button, then :nth-of-type paths.
4. Never propose random hashed classes or unqualified "button" / "div" as first choice.
5. The selector must uniquely identify ONE element. If multiple elements share an attribute value, qualify with a parent selector or :nth-of-type.
6. Return a JSON array of 3-5 CSS selectors, most reliable first.
7. Output ONLY the JSON array. No markdown fences, no explanation, no prose."""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:9000",
        "X-Title": "Bank-RPA-Healer",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    data = _post_with_retry(payload, headers, label="repair-selector")
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise DiscoveryError(f"unexpected OpenRouter response shape: {e}")
    _diagnose_llm_response(content, label="repair-selector")
    cands = _parse_candidates(content)
    if not cands:
        print(f"[AI Healer] Raw LLM response (no parsable selectors): {content[:300]}")
    seen, out2 = set(), []
    for c in cands:
        c = c.strip()
        if c and c not in seen:
            seen.add(c)
            out2.append(c)
    return out2


def ask_openrouter_candidates_text(dom_context: str, intent: str, text: str,
                                   match_type: str, tag_hint: str,
                                   page_url: str = "") -> List[str]:
    """
    LLM variant for click_text steps. Asks the model for CSS selectors that
    would identify the element currently matching the user's intent.

    Returns a list of CSS selectors. An empty list means "nothing to click".
    Raises DiscoveryError on transport / non-transient HTTP failure or
    unparsable response — distinct from a successful empty answer.
    """
    if not OPENROUTER_API_KEY:
        raise DiscoveryError("OPENROUTER_API_KEY environment variable is missing.")

    target_line = (
        f'Hint text: "{text}"  (match: {match_type}, tag: {tag_hint or "any"})\n'
        if text else ""
    )

    prompt = f"""You are an expert browser automation engineer. The user wants to click an element on the current page.

Page URL: {page_url or "(unknown)"}
Step intent: "{intent}"
{target_line}
The visible text on the actual element MAY be phrased differently from the intent. Match by SEMANTIC meaning.

Live DOM (each entry shows tag, id, parentId, attrs dict with data-*, aria-*, role, jsname, name keys, and innerText snippet):
---
{dom_context}
---

Task: identify the clickable element matching the intent and return CSS selectors that uniquely point to it.

CRITICAL rules:
1. Ground EVERY selector in attributes you can see in the DOM list. Do NOT invent ids or data-* values.
2. The selector must uniquely identify ONE element. If two elements share an attribute value, qualify with a parent selector or :nth-of-type.
3. If the page does NOT show a clickable method chooser (e.g. the prompt was already auto-sent, or the page is waiting for approval), return an EMPTY array [].
4. If a clickable element IS present and matches the intent (by visible text OR by semantic role/data attribute), return 1-3 CSS selectors.
5. Examples of semantic flexibility:
   - intent "Google Prompt" matches elements containing "Google Prompt", "Get a Google prompt", "Send a prompt to your phone", "Tap Yes on your phone"
   - intent "SMS code" matches elements containing "Text message", "Get a code by text"
   - intent "Authenticator" matches elements containing "Authenticator app", "Get a verification code"
6. Output ONLY a JSON array. Empty array [] is a valid answer. No markdown fences, no explanation, no prose."""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:9000",
        "X-Title": "Bank-RPA-Healer-Text",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    data = _post_with_retry(payload, headers, label="repair-text-target")
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise DiscoveryError(f"unexpected OpenRouter response shape: {e}")
    _diagnose_llm_response(content, label="repair-text-target")
    parsed = _parse_candidates(content)
    if not parsed:
        print(f"[AI Healer] Raw LLM response (no parsable selectors): {content[:400]}")
    return parsed


# ---------------------------------------------------------------------------
# Healing — invoked by the runner's retry loop
# ---------------------------------------------------------------------------

async def heal_and_update_config(page, config_path: str, failed_step_index: int,
                                 action_hint: Optional[str] = None) -> str:
    """
    Returns the best VALIDATED selector for the failing step, or "" if no
    candidate could be validated. action_hint="click_text" routes through the
    text-targeted path; otherwise the original selector-based path is used.

    The caller is expected to also inspect live browser state — returning ""
    here does NOT necessarily mean there is nothing to click; it just means
    this helper could not validate a selector against the current DOM.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    step = config["flow_steps"][failed_step_index]
    step_name = step["step_name"]
    failed_selector = step.get("selector", "")

    def _persist(sel):
        step["selector"] = sel
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    # Text-targeted path: re-discover by text and validate.
    if action_hint == "click_text" or step.get("action") == "click_text":
        text = step.get("text", "")
        match_type = step.get("match_type", "contains")
        tag_hint = step.get("tag_hint", "")
        intent = step.get("intent", step_name)

        url = await get_page_url(page)
        print(f"\n[AI Healer] Text-targeted repair for '{step_name}' "
              f"(text='{text}', url={url})...")

        # Scroll so off-screen elements are rendered.
        await scroll_through_page(page)

        # 1) JS scanner first (cheap path).
        discovered = await find_selector_by_text(page, text, match_type, tag_hint)
        if discovered and await _validate_selector_against_target(
                page, discovered, text, match_type):
            print(f"[AI Healer] Repaired selector: '{failed_selector}' -> "
                  f"'{discovered}' (validated)")
            _persist(discovered)
            return discovered

        # 2) LLM with full DOM + URL + intent. Error vs empty are distinct.
        candidates: List[str] = []
        try:
            dom = await extract_interactive_dom(page)
            candidates = ask_openrouter_candidates_text(
                dom, intent, text, match_type, tag_hint, url)
        except DiscoveryError as e:
            print(f"[AI Healer] LLM text-candidate call failed: {e}")
            candidates = []

        print(f"[AI Healer] LLM text-candidates: {candidates}")
        for cand in candidates:
            if await _validate_selector_against_target(page, cand, text, match_type):
                print(f"[AI Healer] Repaired selector: '{failed_selector}' -> "
                      f"'{cand}' (validated)")
                _persist(cand)
                return cand

        # 3) Lenient fallback: token overlap.
        for cand in candidates:
            if await selector_exists(page, cand):
                elem_text = (await get_element_text(page, cand)).lower()
                target = (text or "").lower().strip()
                target_tokens = {t for t in target.split() if len(t) > 2}
                elem_tokens = {t for t in elem_text.split() if len(t) > 2}
                overlap = len(target_tokens & elem_tokens)
                if target_tokens and overlap / len(target_tokens) >= 0.5:
                    print(f"[AI Healer] Lenient repair: '{cand}' "
                          f"(text overlap {overlap}/{len(target_tokens)})")
                    _persist(cand)
                    return cand

        # 4) Nothing validated — preserve any stored selector rather than
        #    blanking it out due to a transient condition. The caller will
        #    decide what to do based on live state.
        return failed_selector or ""

    # Selector-based path (original behaviour).
    print(f"\n[AI Healer] Initiating diagnostic with {MODEL_NAME} for step '{step_name}'...")
    dom_context = await extract_interactive_dom(page)

    # 1) Deterministic fast-path (no LLM cost) — fixes div#passwordNext case
    for cand in build_deterministic_candidates(step_name, failed_selector):
        if await selector_exists(page, cand):
            print(f"[AI Healer] Fast-path match (no LLM needed): '{cand}'")
            config["flow_steps"][failed_step_index]["selector"] = cand
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            return cand

    # 2) LLM repair with ranked candidates
    try:
        candidates = ask_openrouter_candidates(dom_context, step_name, failed_selector)
    except DiscoveryError as e:
        print(f"[AI Healer] LLM call failed: {e}")
        candidates = []

    print(f"[AI Healer] Candidates: {candidates}")
    for cand in candidates:
        if await selector_exists(page, cand):
            print(f"[AI Healer] Repaired selector: '{failed_selector}' -> "
                  f"'{cand}' (validated)")
            config["flow_steps"][failed_step_index]["selector"] = cand
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            return cand

    # 3) Nothing validated — return what we had so the caller can decide.
    if candidates:
        print(f"[AI Healer] WARNING: no candidate validated live; "
              f"returning first guess '{candidates[0]}'.")
        return candidates[0]

    print("[AI Healer] No candidates produced; caller will decide what to do.")
    return failed_selector or ""


# Backwards-compat alias (old runner imported this name)
async def heal_selector(page, config_path: str, failed_step_index: int) -> str:
    return await heal_and_update_config(page, config_path, failed_step_index)


# ---------------------------------------------------------------------------
# Generic selector-based discovery (used for non-2FA steps)
# ---------------------------------------------------------------------------

def discover_selector(dom_context: str, intent: str) -> str:
    """
    Queries the LLM to locate the single best CSS selector matching a high-level
    semantic intent. Used on first run when config.json has empty selectors.
    """
    if not OPENROUTER_API_KEY:
        raise DiscoveryError("OPENROUTER_API_KEY is not set.")

    prompt = f"""You are an expert browser automation engine.
Your goal is to inspect the current page's interactive elements and find the single best CSS selector matching this intent:

Target Intent: "{intent}"

Live Visible DOM Elements:
---
{dom_context}
---

Rules:
1. Return ONLY the most robust, unique CSS selector matching the intent.
2. Prefer unique attributes in order: #id, [name='...'], [placeholder='...'], [aria-label='...'], or '#containerId button'.
3. Do not return markdown ticks or explanations. Output the raw CSS selector only."""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:9000",
        "X-Title": "Bank-RPA-Discovery"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0
    }

    data = _post_with_retry(payload, headers, label="discover-selector")
    try:
        raw_content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise DiscoveryError(f"unexpected OpenRouter response shape: {e}")
    _diagnose_llm_response(raw_content, label="discover-selector")
    res = raw_content.strip().replace("`", "").strip()
    return res.split("\n")[0].strip('"').strip("'")


async def populate_and_save_selector(page, config_path: str, step_index: int) -> str:
    """
    Discovers the element matching a step's semantic intent, persists it into
    config.json, and returns the new selector. Used on first run when a step
    has an empty selector.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    step = config["flow_steps"][step_index]
    intent = step.get("intent", step["step_name"])

    print(f"\n[AI Discovery] Searching for element matching intent: '{intent}'...")
    dom = await extract_interactive_dom(page)
    new_selector = discover_selector(dom, intent)
    print(f"[AI Discovery] Discovered and mapped: '{new_selector}'")

    config["flow_steps"][step_index]["selector"] = new_selector
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return new_selector


__all__ = [
    "DiscoveryError",
    "DiscoveryStatus",
    "DiscoveryResult",
    "build_deterministic_candidates",
    "discover_selector",
    "extract_interactive_dom",
    "find_selector_by_text",
    "get_element_text",
    "get_page_url",
    "heal_and_update_config",
    "heal_selector",
    "is_chooser_visible",
    "populate_and_save_selector",
    "populate_and_save_text_target",
    "scroll_through_page",
    "selector_exists",
]
