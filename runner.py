"""
runner.py — orchestrates the configured authentication flow.

Architecture
------------
This module deliberately separates three concerns:

  * ACTION execution — the step loop that performs clicks / types / sleeps.
  * STATE observation — the small auth state machine in `_auth_state.py`.
  * SELECTOR healing — delegated to `healer.py`.

The state machine is what fixes the `challenge/pwd → challenge/selection`
race that previously caused the runner to skip method selection too early.
After password submission the runner now polls the page until it reaches a
recognizable state (authenticated, chooser, prompt-auto-sent, challenge-active,
transitioning) and only then decides what to do.

Public entry point: `run_automation()`.
"""

import asyncio
import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Awaitable, Callable, List, Optional

import nodriver as uc
from dotenv import load_dotenv

from healer import (
    DiscoveryError,
    DiscoveryResult,
    DiscoveryStatus,
    find_selector_by_text,
    get_page_url,
    heal_and_update_config,
    is_chooser_visible,
    populate_and_save_selector,
    populate_and_save_text_target,
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


# ---------------------------------------------------------------------------
# Auth state machine
# ---------------------------------------------------------------------------

class AuthState(str, Enum):
    """Observable page state during post-password navigation."""

    TRANSITIONING = "transitioning"   # still navigating, nothing settled yet
    AUTHENTICATED = "authenticated"   # landed on the post-auth domain
    CHOOSER = "chooser"               # 2FA method chooser is visible
    PROMPT_SENT = "prompt_sent"       # prompt was auto-sent (no chooser)
    CHALLENGE_ACTIVE = "challenge_active"  # some other 2FA challenge is showing
    ERROR = "error"                   # unexpected error state
    UNKNOWN = "unknown"


@dataclass
class StateProvider:
    """Provider-specific state classification rules.

    Defaults are tuned for Google accounts, but the structure is reusable:
    any site exposing a chooser-style 2FA screen can be plugged in by
    extending this dataclass. The runner does NOT special-case URLs beyond
    the markers supplied here.
    """

    authenticated_url_markers: List[str]
    challenge_url_markers: List[str]
    chooser_url_markers: List[str]
    error_url_markers: List[str]
    prompt_sent_indicators: List[str]  # visible-text substrings
    error_indicators: List[str]        # visible-text substrings
    password_challenge_markers: List[str] = None  # URLs indicating still on password step

    def __post_init__(self):
        if self.password_challenge_markers is None:
            self.password_challenge_markers = []

    @classmethod
    def from_config(cls, cfg: dict) -> "StateProvider":
        sp = cfg.get("2fa_settings", {}).get("state_provider", {}) or {}
        return cls(
            authenticated_url_markers=list(sp.get(
                "authenticated_url_markers",
                ["myaccount.google.com", "mail.google.com", "oauth",
                 "google.com/u/", "accounts.google.com/b/"])),
            challenge_url_markers=list(sp.get(
                "challenge_url_markers",
                ["signin/v2/challenge", "v3/signin/challenge"])),
            chooser_url_markers=list(sp.get(
                "chooser_url_markers", ["challenge/selection"])),
            error_url_markers=list(sp.get(
                "error_url_markers",
                ["accounts.google.com/v3/signin/rejected"])),
            prompt_sent_indicators=list(sp.get(
                "prompt_sent_indicators",
                ["tap yes", "tapping", "approve", "tried to sign in",
                 "did you just sign in", "notification on your",
                 "check your phone"])),
            error_indicators=list(sp.get(
                "error_indicators",
                ["couldn't sign you in", "wrong password",
                 "this browser or app may not be secure"])),
            password_challenge_markers=list(sp.get(
                "password_challenge_markers", ["challenge/pwd"])),
        )

    def classify(self, url: str, page_text: str, chooser: bool) -> AuthState:
        u = (url or "").lower()
        text = (page_text or "").lower()
        if any(m in u for m in self.authenticated_url_markers):
            # Don't fire AUTHENTICATED just because an oauth callback
            # *contains* myaccount — but if we got past challenge/selection
            # and we're on an auth-success domain, treat it as success.
            if not any(m in u for m in self.challenge_url_markers):
                return AuthState.AUTHENTICATED
        if any(m in u for m in self.error_url_markers):
            return AuthState.ERROR
        if chooser:
            return AuthState.CHOOSER
        if any(t in text for t in self.prompt_sent_indicators):
            return AuthState.PROMPT_SENT
        if any(t in text for t in self.error_indicators):
            return AuthState.ERROR
        if any(m in u for m in self.chooser_url_markers):
            # URL says we're on the chooser but the chooser probe couldn't
            # confirm it. Could be a Google quirk where data-* attrs haven't
            # painted yet — classify as CHOOSER and let the runner poll.
            return AuthState.CHOOSER
        # Check for password challenge URLs - these indicate we're still
        # on the password step, not yet at 2FA stage.
        if any(m in u for m in self.password_challenge_markers):
            return AuthState.TRANSITIONING
        # Check for other challenge types that indicate an active 2FA challenge
        # (TOTP, security key, etc.) but NOT password challenges.
        if any(m in u for m in self.challenge_url_markers):
            return AuthState.CHALLENGE_ACTIVE
        return AuthState.UNKNOWN


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


def resolve_2fa_method(config: dict) -> str:
    """The configured 2FA method, allowing env override."""
    return (os.environ.get("GMAIL_2FA_METHOD")
            or config.get("2fa_settings", {}).get("method", "")
            or "google_prompt")


def resolve_2fa_text(step: dict, config: dict) -> str:
    """Decide which visible text to click for a click_text step.

    Priority:
      1. step["text"] if present.
      2. METHOD_TEXT_MAP[2fa_settings.method] mapped to a Google-style label.
      3. The configured method name as a last resort.
    """
    if step.get("text"):
        return step["text"]
    method = resolve_2fa_method(config)
    return METHOD_TEXT_MAP.get(method, method or step.get("text", ""))


# ---------------------------------------------------------------------------
# Browser helpers
# ---------------------------------------------------------------------------

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
# Auth state observation
# ---------------------------------------------------------------------------

async def _get_visible_text(page) -> str:
    """Return the visible body text (truncated) for state classification."""
    js = "() => (document.body && (document.body.innerText || document.body.textContent) || '').slice(0, 4000)"
    try:
        result = await page.evaluate(js, return_by_value=True)
        if isinstance(result, dict) and "value" in result:
            result = result["value"]
        return str(result or "")
    except Exception:
        return ""


async def observe_auth_state(page, provider: StateProvider) -> AuthState:
    """One-shot classification of the current page."""
    url = await get_page_url(page)
    chooser = await is_chooser_visible(page)
    text = await _get_visible_text(page)
    return provider.classify(url, text, chooser)


async def wait_for_post_password_state(
    page, provider: StateProvider,
    timeout: float = 15, poll: float = 0.5,
) -> AuthState:
    """
    Poll until the page reaches a state other than TRANSITIONING / UNKNOWN
    that the runner can act on. Returns the final state.

    Unlike the previous "click passwordNext then immediately call
    select_2fa_method" pattern, this waits for Google to actually finish
    moving to `challenge/selection` (or further) before any 2FA work begins.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    last = AuthState.TRANSITIONING
    while asyncio.get_event_loop().time() < deadline:
        last = await observe_auth_state(page, provider)
        if last not in (AuthState.TRANSITIONING, AuthState.UNKNOWN):
            return last
        await page.sleep(poll)
    return last


async def wait_for_state(
    page, provider: StateProvider,
    accept: List[AuthState], timeout: float, poll: float = 1.0,
) -> AuthState:
    """Poll until one of `accept` is observed or timeout."""
    deadline = asyncio.get_event_loop().time() + timeout
    last = await observe_auth_state(page, provider)
    acceptable = set(accept)
    while asyncio.get_event_loop().time() < deadline:
        if last in acceptable:
            return last
        await page.sleep(poll)
        last = await observe_auth_state(page, provider)
    return last


# ---------------------------------------------------------------------------
# Text-targeted click (used for the 2FA chooser)
# ---------------------------------------------------------------------------

async def click_by_text(page, text: str, match_type: str = "contains",
                        tag_hint: str = "li",
                        timeout: float = 8) -> Optional[str]:
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
                pass
        await page.sleep(0.6)

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
    return last_selector


# ---------------------------------------------------------------------------
# URL-based waiting — used for the final approval wait
# ---------------------------------------------------------------------------

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

    raise TimeoutError(
        f"URL did not satisfy contains={contains} not_contains={not_contains} "
        f"within {timeout}s. Last url: {last_url}"
    )


# ---------------------------------------------------------------------------
# 2FA decision logic — drives the post-password portion of the flow
# ---------------------------------------------------------------------------

async def handle_select_2fa_step(page, step: dict, idx: int, config: dict,
                                 provider: StateProvider,
                                 config_path: str = CONFIG_PATH) -> bool:
    """
    Implements the `select_2fa_method` step in a state-aware way.

    Returns True if the method was selected (or selection was correctly
    skipped because the chooser is genuinely not present). Returns False
    if the runner should treat this as a hard failure (e.g. chooser is
    visible and discovery could not find the configured method).
    """
    optional = bool(step.get("optional", False))

    # First, observe the state — what does the page actually show?
    state = await observe_auth_state(page, provider)
    print(f"[*] Post-password state observed: {state.value}")

    if state == AuthState.AUTHENTICATED:
        print("[+] Authentication already complete — no 2FA needed.")
        return True

    if state == AuthState.PROMPT_SENT:
        print("[*] Google Prompt appears to have been auto-sent "
              "(no chooser visible, prompt indicator present). "
              "Skipping method selection.")
        return True

    if state == AuthState.CHALLENGE_ACTIVE:
        print("[*] An active challenge is showing without a chooser "
              "(e.g. TOTP code, security key). Skipping method selection; "
              "final approval wait will block until auth completes.")
        return True

    if state == AuthState.ERROR:
        if optional:
            print("[!] Page is in an error state; treating as optional skip.")
            return True
        raise RuntimeError("Post-password page is in an error state.")

    # If we're not yet on a stable state, wait for one before deciding.
    if state in (AuthState.TRANSITIONING, AuthState.UNKNOWN):
        print("[*] Page still transitioning — waiting for a stable state...")
        state = await wait_for_post_password_state(
            page, provider,
            timeout=config.get("2fa_settings", {}).get(
                "transition_grace_seconds", 8),
            poll=config.get("2fa_settings", {}).get("poll_interval_seconds", 0.5),
        )
        print(f"[*] Stable state reached: {state.value}")
        if state == AuthState.AUTHENTICATED:
            print("[+] Authentication already complete.")
            return True
        if state == AuthState.PROMPT_SENT:
            print("[*] Google Prompt auto-sent during transition.")
            return True
        if state == AuthState.CHALLENGE_ACTIVE:
            print("[*] Active challenge (no chooser) — skipping selection.")
            return True
        if state == AuthState.ERROR:
            if optional:
                print("[!] Error state after wait; optional skip.")
                return True
            raise RuntimeError("Post-password page is in an error state.")

    # CHOOSER: actually do method selection.
    return await _select_from_chooser(page, step, idx, config, provider,
                                     optional, config_path=config_path)


async def _select_from_chooser(page, step: dict, idx: int, config: dict,
                               provider: StateProvider, optional: bool,
                               config_path: str = CONFIG_PATH) -> bool:
    """Chooser IS present. Discover and click the configured method."""
    intent = step.get("intent", step.get("step_name", ""))
    text = resolve_2fa_text(step, config)
    match_type = step.get("match_type", "contains")
    tag_hint = step.get("tag_hint", "")
    stored_selector = (step.get("selector") or "").strip()

    print(f"[*] 2FA chooser visible. Preferred method text: '{text}'.")
    print(f"[*] Stored selector: '{stored_selector or '(none)'}'.")

    # 1) Try stored selector first (cheap; no LLM).
    if stored_selector:
        try:
            el = await page.select(stored_selector, timeout=3)
            if el:
                await el.click()
                await page.sleep(1.0)
                print(f"[+] Clicked preferred 2FA method via stored selector "
                      f"'{stored_selector}'.")
                return True
            print(f"[!] Stored selector '{stored_selector}' did not resolve.")
        except Exception as e:
            print(f"[!] Stored selector '{stored_selector}' failed: {e}")

    # 2) Discovery — propagates resolved text/match_type/tag_hint.
    print("[?] Discovering selector for configured 2FA method...")
    discovery: DiscoveryResult = await populate_and_save_text_target(
        page, config_path, idx,
        text=text, match_type=match_type, tag_hint=tag_hint,
    )

    if discovery.status == DiscoveryStatus.FOUND and discovery.selector:
        try:
            el = await page.select(discovery.selector, timeout=4)
            if el:
                await el.click()
                await page.sleep(1.0)
                print(f"[+] Clicked preferred 2FA method via "
                      f"'{discovery.selector}'.")
                return True
        except Exception as e:
            print(f"[!] Newly discovered selector '{discovery.selector}' "
                  f"failed to click: {e}")

    if discovery.status == DiscoveryStatus.EMPTY_GENUINE:
        # Independent confirmation that no chooser is showing right now —
        # safe to skip (the LLM said so AND the chooser probe agreed).
        print("[+] Discovery returned EMPTY_GENUINE; no chooser visible. Skipping.")
        return True

    if discovery.status == DiscoveryStatus.ERROR:
        # Chooser visible but we could not validate a selector — that's a
        # real failure, not an empty result.
        if optional:
            print(f"[!] Discovery error but step is optional: "
                  f"{discovery.detail}. Continuing.")
            return True
        raise RuntimeError(
            f"2FA chooser is visible but no method selector could be "
            f"validated: {discovery.detail}"
        )

    if discovery.status == DiscoveryStatus.NOT_READY:
        if optional:
            print("[~] Page still transitioning; skipping optional selection.")
            return True
        raise RuntimeError(
            "Page not yet ready for 2FA method selection "
            "(still transitioning).")

    # Fallback — no clear status from discovery.
    if optional:
        print("[~] No selector obtained; optional step; continuing.")
        return True
    raise RuntimeError("Could not select the preferred 2FA method.")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run_automation():
    config = load_config()
    print(f"[*] Starting Flow: {config['bank_name']}")

    email_val = os.environ.get("TEST_GMAIL_USER")
    pass_val = os.environ.get("TEST_GMAIL_PASS")
    if not email_val:
        print("[!] ERROR: TEST_GMAIL_USER environment variable is not set!")
        return

    twofa_enabled = config.get("2fa_settings", {}).get("enabled", False)
    if twofa_enabled:
        method = resolve_2fa_method(config)
        print(f"[*] 2FA enabled. Method: {method}. "
              f"Approval timeout: "
              f"{config['2fa_settings'].get('approval_timeout_seconds', 60)}s.")

    provider = StateProvider.from_config(config)

    browser = await uc.start(
        headless=False,
        user_data_dir=USER_DATA_DIR,
        browser_args=["--start-maximized"]
    )

    twofa_state = AuthState.UNKNOWN

    try:
        print(f"[*] Navigating to: {config['login_url']}")
        page = await browser.get(config["login_url"])
        await page.sleep(3)

        steps = config["flow_steps"]
        # We process the 2FA portion specially so we can drive it via the
        # state machine instead of treating the two steps as independent
        # action executions.
        pre_2fa_steps: List[dict] = []
        twofa_select_step: Optional[dict] = None
        twofa_wait_step: Optional[dict] = None
        post_2fa_steps: List[dict] = []
        section = "pre"
        for step in steps:
            sname = step.get("step_name", "")
            if section == "pre":
                if step.get("action") == "click_text" and \
                        "2fa" in sname and "select" in sname:
                    twofa_select_step = step
                    section = "select"
                    continue
                if step.get("action") == "wait_for_url_change" and \
                        "2fa" in sname:
                    twofa_wait_step = step
                    section = "wait"
                    continue
                pre_2fa_steps.append(step)
            elif section == "select":
                if step.get("action") == "wait_for_url_change" and \
                        "2fa" in sname:
                    twofa_wait_step = step
                    section = "wait"
                    continue
                else:
                    post_2fa_steps.append(step)
            else:  # wait
                post_2fa_steps.append(step)

        timeout = config.get("timeout_seconds", 8)

        # ---------- pre-2FA action steps (email + password) ----------
        for idx, step in enumerate(pre_2fa_steps):
            step_name = step["step_name"]
            action = step["action"]
            optional = bool(step.get("optional", False))

            if action == "sleep":
                print(f"[>] Waiting {step.get('duration', 2)}s...")
                await page.sleep(step.get("duration", 2))
                continue

            if action == "wait_for_url_change":
                # Should not appear pre-2FA in the test config, but handle it.
                contains = step.get("contains", [])
                not_contains = step.get("not_contains", [])
                poll = config.get("2fa_settings", {}).get(
                    "poll_interval_seconds", 2)
                wtimeout = step.get(
                    "timeout_seconds",
                    config.get("2fa_settings", {}).get(
                        "approval_timeout_seconds", 60))
                try:
                    final = await wait_for_url_change(
                        page, contains, not_contains, wtimeout, poll)
                    print(f"[+] URL advanced to: {final}")
                except TimeoutError as e:
                    if optional:
                        print(f"[~] Step '{step_name}' timed out but is "
                              f"optional; continuing. {e}")
                        continue
                    raise
                continue

            selector = step["selector"]

            if not selector.strip():
                print(f"[?] Step '{step_name}' has no selector. "
                      f"Auto-discovering from intent...")
                selector = await populate_and_save_selector(
                    page, CONFIG_PATH, idx)
                step["selector"] = selector

            print(f"[>] Executing step: '{step_name}' ({action}) "
                  f"targeting '{selector}'...")

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
                        print(f"[!] Step '{step_name}' failed with "
                              f"selector '{selector}'. Triggering Healer...")
                        selector = await heal_and_update_config(
                            page, CONFIG_PATH, idx)
                        print(f"[↺] Retrying with repaired selector "
                              f"'{selector}'...")
                    else:
                        raise RuntimeError(f"Step '{step_name}' failed "
                                           f"completely: {e}")

        # ---------- state-aware 2FA handling ----------
        if twofa_enabled and (twofa_select_step or twofa_wait_step):
            print("\n[*] Entering state-aware 2FA handling...")
            # 1) Wait for the page to leave the password state.
            initial = await wait_for_post_password_state(
                page, provider,
                timeout=config.get("2fa_settings", {}).get(
                    "transition_grace_seconds", 8),
                poll=config.get("2fa_settings", {}).get(
                    "poll_interval_seconds", 0.5),
            )
            print(f"[*] Post-password state observed: {initial.value}")

            if twofa_select_step is not None:
                # 2) Choose what to do based on state.
                # Locate the index in the original steps list for persistence.
                select_idx = steps.index(twofa_select_step)
                ok = await handle_select_2fa_step(
                    page, twofa_select_step, select_idx, config, provider)
                if not ok and not bool(twofa_select_step.get("optional", False)):
                    raise RuntimeError("2FA method selection failed.")

            # 3) After selection (or auto-prompt), wait for an active
            #    approval / challenge state before going into the long
            #    URL-based wait. Skip this when we already authenticated.
            post = await observe_auth_state(page, provider)
            print(f"[*] State after method selection: {post.value}")
            if post != AuthState.AUTHENTICATED and \
                    twofa_wait_step is not None:
                # Wait briefly for the active-challenge/prompt state to
                # render (or for authentication to complete immediately).
                wait_cfg = config.get("2fa_settings", {})
                post = await wait_for_state(
                    page, provider,
                    accept=[AuthState.CHALLENGE_ACTIVE,
                            AuthState.PROMPT_SENT,
                            AuthState.AUTHENTICATED],
                    timeout=wait_cfg.get("active_challenge_timeout_seconds", 15),
                    poll=wait_cfg.get("poll_interval_seconds", 1.0),
                )
                print(f"[*] Active-state reached: {post.value}")

            if twofa_wait_step is not None and \
                    post != AuthState.AUTHENTICATED:
                # 4) Final URL-based approval wait.
                contains = twofa_wait_step.get("contains", [])
                not_contains = twofa_wait_step.get("not_contains", [])
                poll = config.get("2fa_settings", {}).get(
                    "poll_interval_seconds", 2)
                wtimeout = twofa_wait_step.get(
                    "timeout_seconds",
                    config.get("2fa_settings", {}).get(
                        "approval_timeout_seconds", 60))
                print(f"[*] Waiting for authentication to complete "
                      f"(timeout={wtimeout}s)...")
                try:
                    final = await wait_for_url_change(
                        page, contains, not_contains, wtimeout, poll)
                    print(f"[+] Authentication complete. Final URL: {final}")
                except TimeoutError as e:
                    print(f"[!] Final approval wait timed out: {e}")
                    raise
            twofa_state = await observe_auth_state(page, provider)
            print(f"[+] 2FA phase complete. State: {twofa_state.value}")

        # ---------- any post-2FA action steps (unused for test config) ----
        for step in post_2fa_steps:
            step_name = step["step_name"]
            action = step["action"]
            optional = bool(step.get("optional", False))

            if action == "sleep":
                print(f"[>] Waiting {step.get('duration', 2)}s...")
                await page.sleep(step.get("duration", 2))
                continue

            if action == "wait_for_url_change":
                contains = step.get("contains", [])
                not_contains = step.get("not_contains", [])
                poll = config.get("2fa_settings", {}).get(
                    "poll_interval_seconds", 2)
                wtimeout = step.get(
                    "timeout_seconds",
                    config.get("2fa_settings", {}).get(
                        "approval_timeout_seconds", 60))
                try:
                    final = await wait_for_url_change(
                        page, contains, not_contains, wtimeout, poll)
                    print(f"[+] URL advanced to: {final}")
                except TimeoutError as e:
                    if optional:
                        print(f"[~] Step '{step_name}' timed out but is "
                              f"optional; continuing. {e}")
                        continue
                    raise
                continue

            selector = step["selector"]
            print(f"[>] Executing step: '{step_name}' ({action}) "
                  f"targeting '{selector}'...")
            for attempt in range(2):
                try:
                    if action == "type":
                        val = os.environ.get(step.get("value_env", ""), "")
                        await reliable_type(page, selector, val, timeout=timeout)
                    elif action == "click":
                        await reliable_click(page, selector, timeout=timeout)
                    print(f"[+] Successfully executed '{step_name}'.")
                    break
                except Exception as e:
                    if attempt == 0:
                        selector = await heal_and_update_config(
                            page, CONFIG_PATH, steps.index(step))
                    else:
                        raise RuntimeError(f"Step '{step_name}' failed "
                                           f"completely: {e}")

        print("[OK] Flow completed successfully.")
        await page.sleep(5)

    finally:
        browser.stop()


if __name__ == "__main__":
    uc.loop().run_until_complete(run_automation())
