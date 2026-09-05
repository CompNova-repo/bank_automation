import os
import json
import re
import requests

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL_NAME = os.environ.get("HEALER_MODEL", "qwen/qwen3.7-flash")


async def extract_interactive_dom(page, limit: int = 80):
    """
    Extracts visible interactive elements with enough context to ground
    selector repair. Crucially includes div/span wrappers that carry the id
    (e.g. Google's div#passwordNext > button), which the old extractor missed
    because it only queried input/button/a/[role=button]/[tabindex].
    """
    js_extract = """
    () => {
        const els = Array.from(document.querySelectorAll(
            'input, button, a, [role="button"], [tabindex], div[id], span[id], [jsname], [data-testid], [aria-label], [name]'
        ));
        const out = [];
        for (const el of els) {
            const r = el.getBoundingClientRect ? el.getBoundingClientRect() : {width: 0, height: 0, x: 0, y: 0};
            const style = window.getComputedStyle ? window.getComputedStyle(el) : null;
            const visible = r.width > 0 && r.height > 0 && (!style || (style.visibility !== 'hidden' && style.display !== 'none'));
            const parentWithId = el.closest ? (el.closest('div[id], span[id], form[id]') || null) : null;
            out.push({
                tag: el.tagName ? el.tagName.toLowerCase() : '',
                id: el.id || '',
                parentId: (parentWithId && parentWithId !== el) ? parentWithId.id : '',
                name: el.getAttribute ? (el.getAttribute('name') || '') : '',
                type: el.getAttribute ? (el.getAttribute('type') || '') : '',
                ariaLabel: el.getAttribute ? (el.getAttribute('aria-label') || '') : '',
                innerText: ((el.innerText || el.textContent || '').trim()).slice(0, 80),
                jsname: el.getAttribute ? (el.getAttribute('jsname') || '') : '',
                dataTestId: el.getAttribute ? (el.getAttribute('data-testid') || '') : '',
                className: (el.className && typeof el.className === 'string') ? el.className.slice(0, 60) : '',
                visible: visible,
                x: Math.round(r.x || 0),
                y: Math.round(r.y || 0)
            });
        }
        // Visible elements first, then anything with an id/text/label
        out.sort((a, b) => (b.visible - a.visible));
        return out.filter(item => item.id || item.parentId || item.name || item.innerText || item.ariaLabel || item.jsname || item.dataTestId)
                  .slice(0, LIMIT);
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


def ask_openrouter_candidates(dom_context: str, step_name: str, failed_selector: str):
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable is missing.")

    prompt = f"""You are an expert browser automation engineer repairing a broken CSS selector.
Failed step intent: "{step_name}"
Broken selector: "{failed_selector}"

Live DOM (visible first, truncated innerText). Ground every guess in this list — do NOT invent ids not present unless it is a well-known Google login id AND no better grounded selector exists:
---
{dom_context}
---

Rules:
1. Target the element matching "{step_name}" (e.g. click_next_password = the visible "Next"/"Weiter" button on the password screen).
2. If a button has no id but sits inside a parent with an id (see parentId), propose "#parentId button" and "#parentId".
3. Prefer stable selectors: #id, #parentId button, [name="..."], [jsname="..."], [data-testid="..."], [aria-label="..."].
4. Never propose random hashed classes (e.g. button.J7pUA) or generic unqualified "button" as first choice.
5. Return a JSON array of 3-5 CSS selectors, most reliable first. Example: ["#passwordNext button", "#passwordNext", "[jsname='V67aGc']"]
6. Output ONLY the JSON array, no markdown, no explanation."""

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
    # Always keep the failed selector out, de-dup
    seen, out = set(), []
    for c in cands:
        c = c.strip()
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


async def heal_and_update_config(page, config_path: str, failed_step_index: int) -> str:
    """
    Returns the best VALIDATED selector. Tries deterministic candidates first,
    then LLM candidates, validating each against the live DOM. Only persists
    a selector that actually exists, so config.json never gets a hallucinated
    value. Falls back to the LLM's first guess (unvalidated) so the caller can
    still attempt text-based recovery.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    step = config["flow_steps"][failed_step_index]
    step_name = step["step_name"]
    failed_selector = step["selector"]

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
