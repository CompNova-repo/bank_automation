import os
import json
import asyncio
import nodriver as uc
from dotenv import load_dotenv
from healer import (
    heal_and_update_config,
    populate_and_save_selector,
    populate_and_save_text_target,
    find_selector_by_text,
)

load_dotenv()

USER_DATA_DIR = os.path.join(os.environ.get("LOCALAPPDATA", "C:\\Temp"), "Test_RPA_Profile")
CONFIG_PATH = "config.json"

# Map a semantic 2FA method name -> the visible label Google renders on the
# 2FA method chooser page. Used when a step doesn't carry its own `text`.
METHOD_TEXT_MAP = {
    "google_prompt": "Google Prompt",
    "sms": "Text message",
    "voice": "Phone call",
    "totp": "Authenticator app",
    "security_key": "Security key",
}

# Fallback order when the preferred tag yields nothing. Picked to match
# Google's typical 2FA method chooser rendering (li > div > button > ...).
TEXT_TAG_FALLBACK = ["li", "div", "button", "a", "span"]


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def resolve_2fa_text(step: dict, config: dict) -> str:
    """
    Decide which visible text to click for a click_text step.

    Priority: step["text"] > METHOD_TEXT_MAP[2fa_settings.method] > step["text"] fallback "".
    Environment variable GMAIL_2FA_METHOD overrides 2fa_settings.method when set.
    """
    if step.get("text"):
        return step["text"]
    method = os.environ.get("GMAIL_2FA_METHOD") or config.get("2fa_settings", {}).get("method", "")
    return METHOD_TEXT_MAP.get(method, method or step.get("text", ""))


async def reliable_type(page, selector: str, text: str, timeout: float = 8):
    """
    Clicks the element to focus Google's dynamic input, clears existing text, and types.
    """
    if not text:
        raise ValueError("Environment variable value for typing is empty!")

    el = await page.select(selector, timeout=timeout)
    if not el:
        raise TimeoutError(f"Input element '{selector}' not found.")

    # 1. Focus input
    await el.click()
    await page.sleep(0.3)

    # 2. Send keys directly
    await el.send_keys(text)
    await page.sleep(0.5)


async def reliable_click(page, selector: str, timeout: float = 8):
    """
    Scrolls to and clicks the target element.
    """
    el = await page.select(selector, timeout=timeout)
    if not el:
        raise TimeoutError(f"Button element '{selector}' not found.")

    await el.click()
    await page.sleep(1.5)


# ---------------------------------------------------------------------------
# Text-based discovery (used for the 2FA method chooser, which has no stable id)
# The DOM-scanning JS + selector builder live in healer.py — see
# find_selector_by_text / _build_text_finder_js there.
# ---------------------------------------------------------------------------

async def click_by_text(page, text: str, match_type: str = "contains", tag_hint: str = "li",
                        timeout: float = 8) -> str | None:
    """
    Locate an element by visible text and click it. Returns the selector used
    so the caller can persist it into config.json for next-run speed.
    Returns None if nothing matches (caller decides whether to fail).
    """
    if not text:
        raise ValueError("click_by_text requires a non-empty 'text' value.")

    deadline = asyncio.get_event_loop().time() + timeout
    last_selector = None
    while asyncio.get_event_loop().time() < deadline:
        selector = await find_selector_by_text(page, text, match_type, tag_hint)
        if selector:
            last_selector = selector
            try:
                el = await page.select(selector, timeout=2)
                if el:
                    await el.click()
                    await page.sleep(1.0)
                    return selector
            except Exception:
                # Selector stale — let next iteration re-discover.
                pass
        await page.sleep(0.6)

    # Fallback: try each tag hint sequentially in case tag_hint is wrong.
    if tag_hint:
        for t in TEXT_TAG_FALLBACK:
            if t == tag_hint:
                continue
            selector = await find_selector_by_text(page, text, match_type, t)
            if selector:
                last_selector = selector
                try:
                    el = await page.select(selector, timeout=2)
                    if el:
                        await el.click()
                        await page.sleep(1.0)
                        return selector
                except Exception:
                    continue

    return last_selector  # may be None — signals "not found"


async def wait_for_url_change(page, contains: list, not_contains: list,
                              timeout: float = 60, poll: float = 2) -> str:
    """
    Poll page.url until it matches the criteria:
      - url contains any of `contains` (OR)
      - url contains none of `not_contains` (AND)

    Returns the final URL on success. Raises TimeoutError otherwise.
    """
    contains = contains or [""]
    not_contains = not_contains or []
    deadline = asyncio.get_event_loop().time() + timeout
    last_url = ""
    while asyncio.get_event_loop().time() < deadline:
        try:
            # nodriver's Tab.url is a sync property; fall back to JS if absent.
            url_attr = getattr(page, "url", None)
            if isinstance(url_attr, str):
                url = url_attr
            else:
                url = await page.evaluate("location.href", return_by_value=True)
            if isinstance(url, dict) and "value" in url:
                url = url["value"]
            last_url = url or ""
        except Exception:
            last_url = ""

        ok = any((c in last_url) for c in contains) if contains else True
        bad = any((nc in last_url) for nc in not_contains)
        if ok and not bad:
            return last_url

        await page.sleep(poll)

    raise TimeoutError(f"URL did not satisfy contains={contains} not_contains={not_contains} within {timeout}s. Last url: {last_url}")


async def run_automation():
    config = load_config()
    print(f"[*] Starting Flow: {config['bank_name']}")

    # Ensure env vars exist before starting
    email_val = os.environ.get("TEST_GMAIL_USER")
    pass_val = os.environ.get("TEST_GMAIL_PASS")
    if not email_val:
        print("[!] ERROR: TEST_GMAIL_USER environment variable is not set!")
        return

    twofa_enabled = config.get("2fa_settings", {}).get("enabled", False)
    if twofa_enabled:
        method = os.environ.get("GMAIL_2FA_METHOD") or config["2fa_settings"].get("method", "google_prompt")
        print(f"[*] 2FA enabled. Method: {method}. "
              f"Approval timeout: {config['2fa_settings'].get('approval_timeout_seconds', 60)}s.")

    browser = await uc.start(
        headless=False,
        user_data_dir=USER_DATA_DIR,
        browser_args=["--start-maximized"]
    )

    try:
        print(f"[*] Navigating to: {config['login_url']}")
        page = await browser.get(config["login_url"])
        await page.sleep(3)

        steps = config["flow_steps"]
        for idx, step in enumerate(steps):
            step_name = step["step_name"]
            action = step["action"]
            timeout = config.get("timeout_seconds", 8)
            optional = bool(step.get("optional", False))

            if action == "sleep":
                print(f"[>] Waiting {step.get('duration', 2)}s...")
                await page.sleep(step.get("duration", 2))
                continue

            # ---- wait_for_url_change: no selector needed ---------------------
            if action == "wait_for_url_change":
                contains = step.get("contains", [])
                not_contains = step.get("not_contains", [])
                poll = config.get("2fa_settings", {}).get("poll_interval_seconds", 2)
                wtimeout = step.get("timeout_seconds",
                                    config.get("2fa_settings", {}).get("approval_timeout_seconds", 60))
                print(f"[>] Executing step: '{step_name}' waiting for URL change "
                      f"(contains={contains}, not_contains={not_contains}, timeout={wtimeout}s)...")
                try:
                    final = await wait_for_url_change(page, contains, not_contains, wtimeout, poll)
                    print(f"[+] URL advanced to: {final}")
                except TimeoutError as e:
                    if optional:
                        print(f"[~] Step '{step_name}' timed out but is optional; continuing. {e}")
                        continue
                    raise
                continue

            # ---- click_text: text-targeted discovery / heal -----------------
            if action == "click_text":
                # The intent is the primary signal. `text` is just a hint.
                intent = step.get("intent", step_name)
                text = step.get("text", "") or resolve_2fa_text(step, config)
                match_type = step.get("match_type", "contains")
                tag_hint = step.get("tag_hint", "")
                selector = step.get("selector", "").strip()
                print(f"[>] Executing step: '{step_name}' (click_text) "
                      f"intent='{intent[:80]}...' text='{text}' optional={optional}...")

                # First-run discovery if selector missing.
                if not selector:
                    print(f"[?] Step '{step_name}' has no selector. Auto-discovering by intent...")
                    selector = await populate_and_save_text_target(
                        page, CONFIG_PATH, idx
                    )
                    step["selector"] = selector or ""

                # If discovery returned empty (LLM said nothing to click),
                # gracefully continue when optional.
                if not selector:
                    if optional:
                        print(f"[~] Step '{step_name}' has nothing to click (prompt may already be sent). Continuing.")
                        continue
                    else:
                        raise RuntimeError(f"Step '{step_name}' could not discover a selector.")

                for attempt in range(2):
                    try:
                        el = await page.select(selector, timeout=timeout)
                        if el:
                            await el.click()
                            await page.sleep(1.0)
                            print(f"[+] Successfully executed '{step_name}' via selector '{selector}'.")
                            break
                        raise TimeoutError(f"click_text target not found via '{selector}'")
                    except Exception as e:
                        if attempt == 0:
                            print(f"[!] Step '{step_name}' failed with selector '{selector}'. "
                                  f"Triggering Healer (text path)...")
                            selector = await heal_and_update_config(
                                page, CONFIG_PATH, idx, action_hint="click_text"
                            ) or ""
                            step["selector"] = selector
                            print(f"[↺] Retrying with repaired selector '{selector}'...")
                        else:
                            if optional:
                                print(f"[~] Step '{step_name}' failed after heal but is optional; continuing. {e}")
                                break
                            raise RuntimeError(f"Step '{step_name}' failed completely: {e}")
                continue

            # ---- click / type: original path --------------------------------
            selector = step["selector"]

            # 1. First-run discovery: empty selector -> ask AI to populate from intent
            if not selector.strip():
                print(f"[?] Step '{step_name}' has no selector. Auto-discovering from intent...")
                selector = await populate_and_save_selector(page, CONFIG_PATH, idx)
                step["selector"] = selector

            print(f"[>] Executing step: '{step_name}' ({action}) targeting '{selector}'...")

            success = False
            for attempt in range(2):
                try:
                    if action == "type":
                        val = os.environ.get(step.get("value_env", ""), "")
                        await reliable_type(page, selector, val, timeout=timeout)
                    elif action == "click":
                        await reliable_click(page, selector, timeout=timeout)
                    success = True
                    print(f"[+] Successfully executed '{step_name}'.")
                    break
                except Exception as e:
                    if attempt == 0:
                        print(f"[!] Step '{step_name}' failed with selector '{selector}'. Triggering Healer...")
                        selector = await heal_and_update_config(page, CONFIG_PATH, idx)
                        print(f"[↺] Retrying with repaired selector '{selector}'...")
                    else:
                        raise RuntimeError(f"Step '{step_name}' failed completely: {e}")

        print("[OK] Flow completed successfully.")
        await page.sleep(5)

    finally:
        browser.stop()

if __name__ == "__main__":
    uc.loop().run_until_complete(run_automation())