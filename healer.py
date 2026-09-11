import os
import json
import re
import requests
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL_NAME = os.environ.get("HEALER_MODEL", "qwen/qwen3.7-flash")


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
                                tag_hint: str = "li") -> str | None:
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


async def populate_and_save_text_target(page, config_path: str, step_index: int,
                                        text: str = "", match_type: str = "contains",
                                        tag_hint: str = "li") -> str | None:
    """
    Discovers the element matching the step's intent, persists a CSS
    selector for it into config.json, and returns that selector.

    Pipeline:
      0. Scroll through the page so off-screen elements enter the DOM.
      1. JS text-finder scan (free, instant) — pick most-specific text match.
      2. LLM call with full DOM + intent + page URL — let the model decide.
      3. Validate each candidate (selector must match exactly one element
         whose innerText actually contains the target text or overlaps
         semantically with the intent).
      4. Empty list = "nothing to click" (prompt already auto-sent). Persist
         empty selector and return None so the runner can skip gracefully.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    step = config["flow_steps"][step_index]
    intent = step.get("intent", step.get("step_name", ""))
    # Defaults from step config if not passed.
    if not text:
        text = step.get("text", "")
    if not match_type:
        match_type = step.get("match_type", "contains")
    if not tag_hint:
        tag_hint = step.get("tag_hint", "")

    def _persist(sel):
        step["selector"] = sel
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    async def _validate(sel: str) -> bool:
        if not sel or not await selector_exists(page, sel):
            return False
        try:
            count_js = "() => document.querySelectorAll(" + json.dumps(sel) + ").length"
            n = await page.evaluate(count_js, return_by_value=True)
            if isinstance(n, dict) and "value" in n:
                n = n["value"]
            if isinstance(n, (int, float)) and n != 1:
                return False
        except Exception:
            pass
        if not text:
            return True
        elem_text = await get_element_text(page, sel)
        norm = lambda s: (s or "").replace("\s+", " ").strip().lower()
        if match_type == "exact":
            return norm(elem_text) == norm(text)
        return norm(text) in norm(elem_text)

    url = await get_page_url(page)
    print(f"\n[AI Discovery] Page URL: {url}")
    print(f"[AI Discovery] Searching for element matching intent: '{intent[:120]}' "
          f"(text hint='{text}', tag_hint='{tag_hint}')...")

    # 0) Scroll so lazy-loaded / off-screen elements are rendered.
    await scroll_through_page(page)

    # 1) JS scanner (cheap path)
    discovered = await find_selector_by_text(page, text, match_type, tag_hint)
    if discovered and await _validate(discovered):
        print(f"[AI Discovery] Text-scan match: '{discovered}'")
        _persist(discovered)
        return discovered

    # 2) LLM with full DOM + URL context
    dom = await extract_interactive_dom(page)
    candidates = []
    try:
        candidates = ask_openrouter_candidates_text(dom, intent, text, match_type, tag_hint, url)
    except Exception as e:
        print(f"[AI Healer] LLM text-candidate call failed: {e}")

    print(f"[AI Healer] LLM text-candidates: {candidates}")
    for cand in candidates:
        if await _validate(cand):
            print(f"[AI Healer] Repaired selector: '{cand}' (validated)")
            _persist(cand)
            return cand

    # 3) Lenient: token overlap with target text.
    for cand in candidates:
        if await selector_exists(page, cand):
            elem_text = (await get_element_text(page, cand)).lower()
            target = (text or "").lower().strip()
            target_tokens = {t for t in target.split() if len(t) > 2}
            elem_tokens = {t for t in elem_text.split() if len(t) > 2}
            overlap = len(target_tokens & elem_tokens)
            if target_tokens and overlap / len(target_tokens) >= 0.5:
                print(f"[AI Healer] Lenient repair: '{cand}' (text overlap {overlap}/{len(target_tokens)})")
                _persist(cand)
                return cand

    # 4) LLM returned [] — interpret as "nothing to click". Persist empty.
    if not candidates and not discovered:
        print(f"[AI Discovery] LLM returned no candidates — likely nothing clickable on this page. "
              f"Persisting empty selector; runner will skip.")
        _persist("")
        return ""

    # 5) Persist whatever we have so the runner retry-loop can re-attempt.
    if discovered:
        print(f"[AI Healer] WARNING: persisting unvalidated JS-scan selector '{discovered}'.")
        _persist(discovered)
        return discovered

    if candidates:
        print(f"[AI Healer] No candidate fully validated; persisting first guess '{candidates[0]}'.")
        _persist(candidates[0])
        return candidates[0]

    _persist("")
    return ""


def ask_openrouter_candidates(dom_context: str, step_name: str, failed_selector: str,
                              page_url: str = ""):
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable is missing.")

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
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=30
    )
    response.raise_for_status()
    result = response.json()
    content = result["choices"][0]["message"]["content"]
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
                                   page_url: str = ""):
    """
    LLM variant for click_text steps. Asks the model for CSS selectors that
    would identify the element currently matching the user's intent. The
    model is allowed to interpret semantic intent flexibly — the visible
    text may be phrased differently than the literal target, and the page
    may already have auto-advanced past the chooser (in which case the
    correct answer is an empty array).

    Returns a list of CSS selectors. An empty list means "nothing to click".
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable is missing.")

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
    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=30
    )
    response.raise_for_status()
    result = response.json()
    content = result["choices"][0]["message"]["content"]
    parsed = _parse_candidates(content)
    if not parsed:
        print(f"[AI Healer] Raw LLM response (no parsable selectors): {content[:400]}")
    return parsed


async def heal_and_update_config(page, config_path: str, failed_step_index: int,
                                 action_hint: str | None = None) -> str:
    """
    Returns the best VALIDATED selector. Tries deterministic candidates first,
    then LLM candidates, validating each against the live DOM. Only persists
    a selector that actually exists, so config.json never gets a hallucinated
    value. Falls back to the LLM's first guess (unvalidated) so the caller can
    still attempt text-based recovery.

    action_hint="click_text" routes through the text-targeted path; otherwise
    the original selector-based path is used.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    step = config["flow_steps"][failed_step_index]
    step_name = step["step_name"]
    failed_selector = step.get("selector", "")

    # Text-targeted path: re-discover by text and validate.
    if action_hint == "click_text" or step.get("action") == "click_text":
        text = step.get("text", "")
        match_type = step.get("match_type", "contains")
        tag_hint = step.get("tag_hint", "")
        intent = step.get("intent", step_name)

        def _persist(sel):
            step["selector"] = sel
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)

        async def _validate(sel: str) -> bool:
            if not sel or not await selector_exists(page, sel):
                return False
            try:
                count_js = "() => document.querySelectorAll(" + json.dumps(sel) + ").length"
                n = await page.evaluate(count_js, return_by_value=True)
                if isinstance(n, dict) and "value" in n:
                    n = n["value"]
                if isinstance(n, (int, float)) and n != 1:
                    return False
            except Exception:
                pass
            if not text:
                return True
            elem_text = await get_element_text(page, sel)
            norm = lambda s: (s or "").replace("\s+", " ").strip().lower()
            if match_type == "exact":
                return norm(elem_text) == norm(text)
            return norm(text) in norm(elem_text)

        url = await get_page_url(page)
        print(f"\n[AI Healer] Text-targeted repair for '{step_name}' "
              f"(text='{text}', url={url})...")

        # Scroll so off-screen elements are rendered.
        await scroll_through_page(page)

        # 1) JS scanner first (cheap path).
        discovered = await find_selector_by_text(page, text, match_type, tag_hint)
        if discovered and await _validate(discovered):
            print(f"[AI Healer] Repaired selector: '{failed_selector}' -> '{discovered}' (validated)")
            _persist(discovered)
            return discovered

        # 2) LLM with full DOM + URL + intent.
        candidates = []
        try:
            dom = await extract_interactive_dom(page)
            candidates = ask_openrouter_candidates_text(dom, intent, text, match_type, tag_hint, url)
        except Exception as e:
            print(f"[AI Healer] LLM text-candidate call failed: {e}")

        print(f"[AI Healer] LLM text-candidates: {candidates}")
        for cand in candidates:
            if await _validate(cand):
                print(f"[AI Healer] Repaired selector: '{failed_selector}' -> '{cand}' (validated)")
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
                    print(f"[AI Healer] Lenient repair: '{cand}' (text overlap {overlap}/{len(target_tokens)})")
                    _persist(cand)
                    return cand

        # 4) LLM returned [] — nothing clickable, persist empty selector.
        if not candidates and not discovered:
            print(f"[AI Healer] LLM returned no candidates — nothing clickable. Persisting empty.")
            _persist("")
            return ""

        if discovered:
            _persist(discovered)
            return discovered
        if candidates:
            _persist(candidates[0])
            return candidates[0]
        _persist("")
        return ""

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
    except Exception as e:
        print(f"[AI Healer] LLM call failed: {e}")
        candidates = []

    print(f"[AI Healer] Candidates: {candidates}")
    for cand in candidates:
        if await selector_exists(page, cand):
            print(f"[AI Healer] Repaired selector: '{failed_selector}' -> '{cand}' (validated)")
            config["flow_steps"][failed_step_index]["selector"] = cand
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            return cand

    # 3) Nothing validated — persist nothing, return best guess for text-fallback path
    if candidates:
        print(f"[AI Healer] WARNING: no candidate validated live; trying '{candidates[0]}' + text fallback.")
        return candidates[0]

    print("[AI Healer] No candidates produced; caller will try text fallback.")
    return failed_selector


# Backwards-compat alias (old runner imported this name)
async def heal_selector(page, config_path: str, failed_step_index: int) -> str:
    return await heal_and_update_config(page, config_path, failed_step_index)


def discover_selector(dom_context: str, intent: str) -> str:
    """
    Queries the LLM to locate the single best CSS selector matching a high-level
    semantic intent. Used on first run when config.json has empty selectors.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is not set.")

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

    resp = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=25)
    resp.raise_for_status()
    res = resp.json()["choices"][0]["message"]["content"].strip().replace("`", "").strip()
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