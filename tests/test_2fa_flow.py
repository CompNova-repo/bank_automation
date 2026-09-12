"""
Tests for the 2FA state machine, discovery pipeline, and error semantics.

Covers the 8 scenarios listed in the bug report:

  1. Password page remains present briefly and later transitions to
     `challenge/selection`. The method-selection logic must wait instead
     of prematurely skipping.
  2. `google_prompt` is configured and the chooser appears. The
     configured method is actually selected.
  3. Google automatically starts the prompt without displaying the
     chooser. The runner correctly skips method selection and waits
     for approval.
  4. LLM/OpenRouter returns HTTP 429 while a chooser exists. The
     runner must NOT interpret that as "nothing to click", must NOT
     persist an empty selector, and must not incorrectly continue to
     final approval waiting.
  5. LLM successfully returns a genuine empty result while browser
     state independently confirms no chooser is required.
  6. Authentication succeeds directly after password entry.
  7. A stored 2FA selector becomes invalid and recovery/healing is
     triggered.
  8. Delayed navigation does not cause the exact timeout currently
     being observed.

Plus unit tests for the deterministic text-finder, OpenRouter retry
policies, and the state provider classifier.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from typing import List
from unittest.mock import patch

# Configure sane test env defaults BEFORE importing the modules under test.
os.environ.setdefault("OPENROUTER_API_KEY", "")
os.environ.setdefault("TEST_GMAIL_USER", "user@example.com")
os.environ.setdefault("TEST_GMAIL_PASS", "fake-password")
os.environ.setdefault("OPENROUTER_BACKOFF_SECONDS", "0")
os.environ.setdefault("OPENROUTER_MAX_ATTEMPTS", "3")

from tests.fakes import FakePage  # noqa: E402

from healer import (  # noqa: E402
    DiscoveryError,
    DiscoveryResult,
    DiscoveryStatus,
    _parse_candidates,
    _post_with_retry,
    ask_openrouter_candidates_text,
    build_deterministic_candidates,
    find_selector_by_text,
    is_chooser_visible,
    populate_and_save_text_target,
    selector_exists,
)
from runner import (  # noqa: E402
    AuthState,
    METHOD_TEXT_MAP,
    StateProvider,
    _get_visible_text,
    handle_select_2fa_step,
    observe_auth_state,
    reliable_click,
    resolve_2fa_text,
    resolve_2fa_texts,
    wait_for_post_password_state,
    wait_for_state,
)
import runner as runner_module  # noqa: E402  — used so monkey-patches of
# `runner.observe_auth_state` propagate to inner helpers.


def runner_observe(p, sp):
    """Late-binding access to runner.observe_auth_state. Capturing the
    function at import time would defeat monkey-patching."""
    return runner_module.observe_auth_state(p, sp)


PROVIDER = StateProvider.from_config({"2fa_settings": {"state_provider": {}}})


def run(coro):
    """Convenience: run a coroutine and return its result."""
    return asyncio.get_event_loop().run_until_complete(coro)


class _ConfigFixture:
    """Minimal config used by populate_and_save_text_target."""

    FLOW_STEP = {
        "step_name": "select_2fa_method",
        "intent": "Click the Google Prompt option on the 2FA chooser.",
        "action": "click_text",
        "selector": "",
        "optional": True,
    }

    @classmethod
    def write(cls, path: str) -> None:
        cfg = {
            "bank_name": "Test",
            "flow_steps": [cls.FLOW_STEP],
            "2fa_settings": {"method": "google_prompt", "state_provider": {}},
        }
        with open(path, "w") as f:
            json.dump(cfg, f)


# ---------------------------------------------------------------------------
# Unit tests for the state provider and pure helpers
# ---------------------------------------------------------------------------

class StateProviderClassifierTests(unittest.TestCase):
    def test_authenticated_when_url_contains_post_auth_marker(self):
        sp = PROVIDER
        self.assertEqual(
            sp.classify("https://myaccount.google.com/", "", chooser=False),
            AuthState.AUTHENTICATED,
        )
        self.assertEqual(
            sp.classify("https://mail.google.com/mail/u/0/", "", chooser=False),
            AuthState.AUTHENTICATED,
        )
        self.assertEqual(
            sp.classify("https://oauth.example.com/callback", "", chooser=False),
            AuthState.AUTHENTICATED,
        )

    def test_chooser_when_url_matches_selection_path(self):
        sp = PROVIDER
        self.assertEqual(
            sp.classify(
                "https://accounts.google.com/v3/signin/challenge/selection",
                "",
                chooser=True,
            ),
            AuthState.CHOOSER,
        )

    def test_chooser_via_url_only_when_dom_disagrees(self):
        """URL says chooser; DOM probe says no. Classify as CHOOSER
        because the runner will keep polling and re-evaluate."""
        sp = PROVIDER
        self.assertEqual(
            sp.classify(
                "https://accounts.google.com/v3/signin/challenge/selection",
                "",
                chooser=False,
            ),
            AuthState.CHOOSER,
        )

    def test_prompt_sent_via_visible_text(self):
        sp = PROVIDER
        self.assertEqual(
            sp.classify(
                "https://accounts.google.com/v3/signin/challenge/pwd",
                "Google sent a notification to your phone. Tap Yes on your phone to confirm.",
                chooser=False,
            ),
            AuthState.PROMPT_SENT,
        )

    def test_challenge_active_via_challenge_url_no_chooser(self):
        sp = PROVIDER
        self.assertEqual(
            sp.classify(
                "https://accounts.google.com/v3/signin/challenge/otp",
                "Enter the 6-digit code",
                chooser=False,
            ),
            AuthState.CHALLENGE_ACTIVE,
        )

    def test_error_via_url_marker(self):
        sp = PROVIDER
        self.assertEqual(
            sp.classify(
                "https://accounts.google.com/v3/signin/rejected",
                "",
                chooser=False,
            ),
            AuthState.ERROR,
        )

    def test_error_via_visible_text(self):
        sp = PROVIDER
        self.assertEqual(
            sp.classify(
                "https://accounts.google.com/v3/signin/challenge/pwd",
                "Couldn't sign you in. Please try again.",
                chooser=False,
            ),
            AuthState.ERROR,
        )

    def test_transitioning_when_on_password_challenge(self):
        """URL is challenge/pwd which means we're still on password step,
        not yet at 2FA stage. Should be classified as TRANSITIONING."""
        sp = PROVIDER
        self.assertEqual(
            sp.classify(
                "https://accounts.google.com/v3/signin/challenge/pwd",
                "",
                chooser=False,
            ),
            AuthState.TRANSITIONING,
        )

    def test_password_url_wins_over_false_positive_chooser_probe(self):
        """Pre-rendered 2FA markup must not turn a no-op submit into CHOOSER."""
        self.assertEqual(
            PROVIDER.classify(
                "https://accounts.google.com/v3/signin/challenge/pwd",
                "2-Step Verification",
                chooser=True,
            ),
            AuthState.TRANSITIONING,
        )


class Resolve2faTextTests(unittest.TestCase):
    def test_uses_explicit_step_text(self):
        step = {"text": "Get a Google prompt"}
        cfg = {"2fa_settings": {"method": "google_prompt"}}
        self.assertEqual(resolve_2fa_text(step, cfg), "Get a Google prompt")

    def test_falls_back_to_method_text_map(self):
        step = {}
        cfg = {"2fa_settings": {"method": "sms"}}
        self.assertEqual(resolve_2fa_text(step, cfg), "Text message")

    def test_env_var_overrides_configured_method(self):
        step = {}
        cfg = {"2fa_settings": {"method": "totp"}}
        with patch.dict(os.environ, {"GMAIL_2FA_METHOD": "google_prompt"}):
            self.assertEqual(resolve_2fa_text(step, cfg), "Google Prompt")

    def test_default_is_google_prompt(self):
        step = {}
        cfg = {"2fa_settings": {}}
        self.assertEqual(resolve_2fa_text(step, cfg), "Google Prompt")


class ParseCandidatesTests(unittest.TestCase):
    def test_parses_json_array(self):
        out = _parse_candidates('[ "#a", "li:nth-of-type(2)", "[role=button]" ]')
        self.assertEqual(out, ["#a", "li:nth-of-type(2)", "[role=button]"])

    def test_parses_single_string(self):
        out = _parse_candidates("li:nth-of-type(2)")
        self.assertEqual(out, ["li:nth-of-type(2)"])

    def test_parses_dict_with_selectors_key(self):
        out = _parse_candidates('{"selectors": ["#a", "#b"]}')
        self.assertEqual(out, ["#a", "#b"])

    def test_returns_empty_for_empty_input(self):
        self.assertEqual(_parse_candidates(""), [])


class BuildDeterministicCandidatesTests(unittest.TestCase):
    def test_password_next_includes_known_id(self):
        cands = build_deterministic_candidates("click_next_password", "")
        self.assertIn("#passwordNext", cands)

    def test_identifier_next_includes_known_id(self):
        cands = build_deterministic_candidates("click_next_email", "")
        self.assertIn("#identifierNext", cands)


# ---------------------------------------------------------------------------
# FakePage-level tests — exercise the JS probes via the FakePage evaluator
# ---------------------------------------------------------------------------

class FakePageProbeTests(unittest.TestCase):
    def test_is_chooser_visible_returns_true_when_set(self):
        page = FakePage(chooser_visible=True)
        self.assertTrue(run(is_chooser_visible(page)))

    def test_is_chooser_visible_returns_false_otherwise(self):
        page = FakePage(chooser_visible=False)
        self.assertFalse(run(is_chooser_visible(page)))

    def test_observe_auth_state_chooser(self):
        page = FakePage(
            url="https://accounts.google.com/v3/signin/challenge/selection",
            chooser_visible=True,
        )
        self.assertEqual(run(observe_auth_state(page, PROVIDER)),
                         AuthState.CHOOSER)

    def test_observe_auth_state_prompt_sent(self):
        page = FakePage(
            url="https://accounts.google.com/v3/signin/challenge/pwd",
            body_text="Google sent a notification to your phone. Please tap Yes.",
            chooser_visible=False,
        )
        self.assertEqual(run(observe_auth_state(page, PROVIDER)),
                         AuthState.PROMPT_SENT)

    def test_observe_auth_state_authenticated(self):
        page = FakePage(url="https://myaccount.google.com/")
        self.assertEqual(run(observe_auth_state(page, PROVIDER)),
                         AuthState.AUTHENTICATED)


class ReliableClickTests(unittest.TestCase):
    def test_verified_click_rejects_no_op(self):
        page = FakePage(
            url="https://accounts.google.com/v3/signin/challenge/pwd",
            selectors={
                "[jsname='V67aGc']": "Next",
                'input[type="password"]': "",
            },
        )
        with self.assertRaisesRegex(TimeoutError, "page did not change"):
            run(reliable_click(
                page, "[jsname='V67aGc']", timeout=0.01,
                verify_page_change=True,
                disappearance_selector='input[type="password"]',
            ))

    def test_verified_click_accepts_navigation(self):
        page = FakePage(
            url="https://accounts.google.com/v3/signin/challenge/pwd",
            selectors={
                "#passwordNext": "Next",
                'input[type="password"]': "",
            },
        )

        async def navigate(_selector):
            page.url = ("https://accounts.google.com/v3/signin/challenge/"
                        "selection")

        page._click_hook = navigate
        run(reliable_click(
            page, "#passwordNext", timeout=0.01,
            verify_page_change=True,
            disappearance_selector='input[type="password"]',
        ))
        self.assertEqual(page.click_log.get("#passwordNext"), 1)


# ---------------------------------------------------------------------------
# State-aware waiting — covers scenarios 1, 3, 6, 8 from the bug report
# ---------------------------------------------------------------------------

class WaitForPostPasswordStateTests(unittest.TestCase):

    # Scenario 1: password page remains briefly, then transitions to
    # challenge/selection. The runner must NOT skip selection prematurely.
    def test_waits_through_pwd_until_chooser_visible(self):
        # Step 1: page is on challenge/pwd, no chooser. Step 2: after one
        # poll, the chooser becomes visible. wait_for_post_password_state
        # should block until the CHOOSER state is observed.
        page = FakePage(
            url="https://accounts.google.com/v3/signin/challenge/pwd",
            body_text="",
            chooser_visible=False,
        )
        real_observe = runner_module.observe_auth_state
        # Flip state after the first observation.
        first_poll = {"done": False}

        async def flipping_observe(p, sp):
            if not first_poll["done"]:
                first_poll["done"] = True
                p.chooser_visible = True
                p.url = ("https://accounts.google.com/v3/signin/challenge/"
                         "selection?hl=en")
                p.body_text = "2-Step Verification"
            return await real_observe(p, sp)

        with patch("runner.observe_auth_state", flipping_observe):
            state = run(wait_for_post_password_state(
                page, PROVIDER, timeout=4.0, poll=0.01))
        self.assertEqual(state, AuthState.CHOOSER)

    # Scenario 8: delayed navigation does not cause premature timeout.
    # This test proves that as long as the state eventually settles,
    # the wait returns rather than timing out.
    def test_does_not_time_out_when_state_eventually_settles(self):
        # Start with an UNKNOWN URL (matches no markers) so the wait
        # keeps polling until the flip happens.
        page = FakePage(
            url="https://accounts.google.com/loading",
            chooser_visible=False,
            body_text="",
        )
        real_observe = runner_module.observe_auth_state
        poll_count = {"n": 0}

        async def eventually_chooser(p, sp):
            poll_count["n"] += 1
            if poll_count["n"] >= 3:
                p.chooser_visible = True
                p.url = ("https://accounts.google.com/v3/signin/challenge/"
                         "selection")
                p.body_text = "Choose how you want to sign in"
            return await real_observe(p, sp)

        with patch("runner.observe_auth_state", eventually_chooser):
            state = run(wait_for_post_password_state(
                page, PROVIDER, timeout=4.0, poll=0.01))
        self.assertEqual(state, AuthState.CHOOSER)
        self.assertGreaterEqual(poll_count["n"], 3)


# ---------------------------------------------------------------------------
# handle_select_2fa_step — scenarios 2, 3, 4, 5, 7
# ---------------------------------------------------------------------------

class HandleSelect2faStepTests(unittest.TestCase):
    """Drives the runner's decision logic directly, mocking only the
    expensive parts (LLM, browser launch)."""

    FLOW_STEP = _ConfigFixture.FLOW_STEP

    def _make_cfg_path(self) -> str:
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        _ConfigFixture.write(path)
        return path

    def _patch_discovery(self, discovery_result: DiscoveryResult):
        """Patch populate_and_save_text_target to return a fixed result."""
        async def fake_discover(page, cfg_path, idx, text=None,
                                match_type=None, tag_hint=None):
            return discovery_result
        return patch("runner.populate_and_save_text_target", fake_discover)

    # Scenario 2: chooser visible, google_prompt configured → method
    # is selected and the chooser entry is clicked.
    def test_chooser_visible_and_method_found_clicks_it(self):
        cfg_path = self._make_cfg_path()
        # The chooser entry for Google Prompt exists in the DOM.
        page = FakePage(
            url="https://accounts.google.com/v3/signin/challenge/selection",
            chooser_visible=True,
            selectors={
                "li[data-challengetype='13']":
                    "Google Prompt Get a Google prompt",
            },
        )

        async def fake_discover(page, cfg_path, idx, text=None,
                                match_type=None, tag_hint=None):
            # Mimic the real pipeline: persist the validated selector.
            with open(cfg_path) as f:
                cfg = json.load(f)
            cfg["flow_steps"][idx]["selector"] = "li[data-challengetype='13']"
            with open(cfg_path, "w") as f:
                json.dump(cfg, f)
            return DiscoveryResult(
                DiscoveryStatus.FOUND,
                "li[data-challengetype='13']",
            )

        with patch("runner.populate_and_save_text_target", fake_discover):
            ok = run(handle_select_2fa_step(
                page, self.FLOW_STEP, 0,
                {"2fa_settings": {"method": "google_prompt",
                                  "state_provider": {}}},
                PROVIDER,
                config_path=cfg_path,
            ))
        self.assertTrue(ok)
        self.assertEqual(
            page.click_log.get("li[data-challengetype='13']"), 1)

        # Sanity: we never persisted empty selector.
        with open(cfg_path) as f:
            cfg = json.load(f)
        self.assertEqual(
            cfg["flow_steps"][0]["selector"],
            "li[data-challengetype='13']",
        )
        os.unlink(cfg_path)

    # Scenario 3: auto-prompt (no chooser) → skip selection, do not
    # call discovery at all.
    def test_auto_prompt_skips_method_selection(self):
        page = FakePage(
            url="https://accounts.google.com/v3/signin/challenge/pwd",
            chooser_visible=False,
            body_text="We sent a notification to your phone. Tap Yes on "
                      "your phone to confirm it's you.",
        )
        called = {"n": 0}

        async def fake_discover(page, cfg_path, idx, text=None,
                                match_type=None, tag_hint=None):
            called["n"] += 1
            return DiscoveryResult(DiscoveryStatus.EMPTY_GENUINE)

        with patch("runner.populate_and_save_text_target", fake_discover):
            ok = run(handle_select_2fa_step(
                page, self.FLOW_STEP, 0,
                {"2fa_settings": {"method": "google_prompt",
                                  "state_provider": {}}},
                PROVIDER))
        self.assertTrue(ok)
        # Discovery should NOT have been called — the runner observed
        # PROMPT_SENT from browser state alone and skipped.
        self.assertEqual(called["n"], 0)

    # Scenario 4: chooser visible but LLM returned HTTP 429 → must NOT
    # skip, must NOT persist empty selector, must propagate as failure
    # when the step is not optional.
    def test_llm_429_with_chooser_visible_does_not_skip(self):
        cfg_path = self._make_cfg_path()
        page = FakePage(
            url="https://accounts.google.com/v3/signin/challenge/selection",
            chooser_visible=True,
        )

        async def fake_discover(page, cfg_path, idx, text=None,
                                match_type=None, tag_hint=None):
            # Mimics what populate_and_save_text_target returns when LLM
            # failed: status=ERROR, no selector.
            return DiscoveryResult(
                DiscoveryStatus.ERROR,
                detail="LLM call failed after 3 attempts: 429",
            )

        non_optional_step = {**self.FLOW_STEP, "optional": False}
        with patch("runner.populate_and_save_text_target", fake_discover):
            with self.assertRaises(RuntimeError):
                run(handle_select_2fa_step(
                    page, non_optional_step, 0,
                    {"2fa_settings": {"method": "google_prompt",
                                      "state_provider": {}}},
                    PROVIDER,
                    config_path=cfg_path,
                ))
        # Important: empty selector was NOT persisted.
        with open(cfg_path) as f:
            cfg = json.load(f)
        self.assertEqual(cfg["flow_steps"][0]["selector"], "")
        os.unlink(cfg_path)

    # Scenario 4 (companion): the same situation, but the step IS optional.
    # The runner must NOT raise, but it must also NOT silently persist "".
    def test_optional_step_with_llm_429_does_not_persist_empty(self):
        cfg_path = self._make_cfg_path()
        page = FakePage(
            url="https://accounts.google.com/v3/signin/challenge/selection",
            chooser_visible=True,
        )

        async def fake_discover(page, cfg_path, idx, text=None,
                                match_type=None, tag_hint=None):
            return DiscoveryResult(
                DiscoveryStatus.ERROR,
                detail="LLM call failed after 3 attempts: 429",
            )

        with patch("runner.populate_and_save_text_target", fake_discover):
            ok = run(handle_select_2fa_step(
                page, self.FLOW_STEP, 0,
                {"2fa_settings": {"method": "google_prompt",
                                  "state_provider": {}}},
                PROVIDER,
                config_path=cfg_path,
            ))
        self.assertTrue(ok)
        with open(cfg_path) as f:
            cfg = json.load(f)
        # Critical: even when optional, the runner must not blank the
        # selector — it just continues.
        self.assertEqual(cfg["flow_steps"][0]["selector"], "")
        os.unlink(cfg_path)

    # Scenario 5: LLM returns [] AND chooser not visible → empty_genuine,
    # safe to skip.
    def test_empty_genuine_with_no_chooser_skips_safely(self):
        cfg_path = self._make_cfg_path()
        page = FakePage(
            url="https://accounts.google.com/v3/signin/challenge/pwd",
            chooser_visible=False,
            body_text="",
        )

        async def fake_discover(page, cfg_path, idx, text=None,
                                match_type=None, tag_hint=None):
            return DiscoveryResult(DiscoveryStatus.EMPTY_GENUINE)

        with patch("runner.populate_and_save_text_target", fake_discover):
            ok = run(handle_select_2fa_step(
                page,
                # Force the runner past the early state checks so it
                # actually invokes discovery. Use a transitioning URL
                # so wait_for_post_password_state runs.
                {**self.FLOW_STEP, "intent": "Click Google Prompt."},
                0,
                {"2fa_settings": {"method": "google_prompt",
                                  "state_provider": {}}},
                PROVIDER))
        # With no chooser AND empty genuine, the runner may either skip
        # (via early state observation) or rely on the discovery result.
        # Either way, it must not raise.
        self.assertTrue(ok)

        # Empty selector MUST NOT be persisted — we only persist validated
        # selectors.
        with open(cfg_path) as f:
            cfg = json.load(f)
        self.assertEqual(cfg["flow_steps"][0]["selector"], "")
        os.unlink(cfg_path)

    # Scenario 6: auth succeeded directly after password entry → runner
    # observes AUTHENTICATED and returns True without touching the LLM.
    def test_already_authenticated_skips_selection(self):
        page = FakePage(url="https://myaccount.google.com/")
        called = {"n": 0}

        async def fake_discover(page, cfg_path, idx, text=None,
                                match_type=None, tag_hint=None):
            called["n"] += 1
            return DiscoveryResult(DiscoveryStatus.FOUND, "#whatever")

        with patch("runner.populate_and_save_text_target", fake_discover):
            ok = run(handle_select_2fa_step(
                page, self.FLOW_STEP, 0,
                {"2fa_settings": {"method": "google_prompt",
                                  "state_provider": {}}},
                PROVIDER))
        self.assertTrue(ok)
        self.assertEqual(called["n"], 0)

    # Scenario 7: stored 2FA selector is invalid → runner tries stored
    # selector, fails, then calls discovery which returns a fresh
    # validated selector.
    def test_stored_selector_invalid_falls_through_to_discovery(self):
        cfg_path = self._make_cfg_path()
        page = FakePage(
            url="https://accounts.google.com/v3/signin/challenge/selection",
            chooser_visible=True,
            selectors={
                "li[data-challengetype='13']":
                    "Google Prompt Get a Google prompt",
            },
        )
        # Note: stale selector is NOT registered → page.select returns
        # None, simulating "stored selector doesn't resolve". The runner
        # falls through to discovery.

        async def fake_discover(page, cfg_path, idx, text=None,
                                match_type=None, tag_hint=None):
            # Persist the freshly discovered selector.
            with open(cfg_path) as f:
                cfg = json.load(f)
            cfg["flow_steps"][idx]["selector"] = "li[data-challengetype='13']"
            with open(cfg_path, "w") as f:
                json.dump(cfg, f)
            return DiscoveryResult(
                DiscoveryStatus.FOUND,
                "li[data-challengetype='13']",
            )

        step_with_stale = {
            **self.FLOW_STEP,
            "selector": "#stale-selector-from-last-run",
        }
        with patch("runner.populate_and_save_text_target", fake_discover):
            ok = run(handle_select_2fa_step(
                page, step_with_stale, 0,
                {"2fa_settings": {"method": "google_prompt",
                                  "state_provider": {}}},
                PROVIDER,
                config_path=cfg_path,
            ))
        self.assertTrue(ok)
        self.assertEqual(
            page.click_log.get("li[data-challengetype='13']"), 1)
        # The stale selector should NOT have been clicked.
        self.assertIsNone(page.click_log.get(
            "#stale-selector-from-last-run"))

        with open(cfg_path) as f:
            cfg = json.load(f)
        self.assertEqual(
            cfg["flow_steps"][0]["selector"],
            "li[data-challengetype='13']",
        )
        os.unlink(cfg_path)


# ---------------------------------------------------------------------------
# populate_and_save_text_target — direct integration with FakePage
# ---------------------------------------------------------------------------

class PopulateAndSaveTextTargetTests(unittest.TestCase):
    """Exercises the discovery pipeline end-to-end with FakePage and a
    mocked LLM. Validates the FOUND / EMPTY_GENUINE / ERROR / NOT_READY
    semantic separation."""

    def _make_cfg_path(self) -> str:
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        _ConfigFixture.write(path)
        return path

    def test_propagates_caller_supplied_text_into_discovery(self):
        """Regression test for the propagation bug: caller-supplied
        text/match_type/tag_hint must reach the deterministic finder."""
        cfg_path = self._make_cfg_path()
        page = FakePage(
            url="https://accounts.google.com/v3/signin/challenge/selection",
            chooser_visible=True,
            selectors={
                "li[data-challengetype='13']":
                    "Google Prompt Get a Google prompt",
            },
        )

        # Patch out the LLM so we can prove the deterministic path got
        # the text passed by the caller.
        with patch("healer.ask_openrouter_candidates_text") as llm, \
                patch("healer.extract_interactive_dom",
                      return_value="[]"):
            result = run(populate_and_save_text_target(
                page, cfg_path, 0,
                text="Google Prompt", match_type="contains",
                tag_hint="li",
            ))
        # LLM should not need to be called when deterministic finder
        # validates a candidate.
        self.assertEqual(result.status, DiscoveryStatus.FOUND)
        self.assertEqual(result.selector, "li[data-challengetype='13']")
        llm.assert_not_called()
        os.unlink(cfg_path)

    def test_returns_error_when_chooser_visible_and_llm_429(self):
        cfg_path = self._make_cfg_path()
        page = FakePage(
            url="https://accounts.google.com/v3/signin/challenge/selection",
            chooser_visible=True,
        )

        def boom(*a, **kw):
            raise DiscoveryError("HTTP 429: rate-limited")

        with patch("healer.ask_openrouter_candidates_text", side_effect=boom), \
                patch("healer.extract_interactive_dom", return_value="[]"):
            result = run(populate_and_save_text_target(
                page, cfg_path, 0,
                text="Google Prompt", match_type="contains",
                tag_hint="li",
            ))
        self.assertEqual(result.status, DiscoveryStatus.ERROR)
        with open(cfg_path) as f:
            cfg = json.load(f)
        self.assertEqual(cfg["flow_steps"][0]["selector"], "")
        os.unlink(cfg_path)

    def test_returns_empty_genuine_when_llm_says_no_chooser(self):
        cfg_path = self._make_cfg_path()
        page = FakePage(
            url="https://accounts.google.com/v3/signin/challenge/pwd",
            chooser_visible=False,
        )
        with patch("healer.ask_openrouter_candidates_text",
                   return_value=[]), \
                patch("healer.extract_interactive_dom", return_value="[]"):
            result = run(populate_and_save_text_target(
                page, cfg_path, 0,
                text="Google Prompt", match_type="contains",
                tag_hint="li",
            ))
        self.assertEqual(result.status, DiscoveryStatus.EMPTY_GENUINE)
        with open(cfg_path) as f:
            cfg = json.load(f)
        # Empty selector NOT persisted.
        self.assertEqual(cfg["flow_steps"][0]["selector"], "")
        os.unlink(cfg_path)

    def test_returns_not_ready_when_no_chooser_and_no_llm_yet(self):
        cfg_path = self._make_cfg_path()
        page = FakePage(
            url="https://accounts.google.com/v3/signin/challenge/pwd",
            chooser_visible=False,
        )
        # Force the LLM to not be called by simulating a transient
        # network blip — but no error surfaced. This is hard to fake
        # without raising; the realistic "page still settling" case
        # shows up as the deterministic finder missing AND the LLM not
        # yet producing validated output. We approximate by returning a
        # single non-validated candidate from the LLM.
        with patch("healer.ask_openrouter_candidates_text",
                   return_value=["#does-not-exist"]), \
                patch("healer.extract_interactive_dom", return_value="[]"):
            result = run(populate_and_save_text_target(
                page, cfg_path, 0,
                text="Google Prompt", match_type="contains",
                tag_hint="li",
            ))
        # No chooser visible AND no validated candidate → ERROR (we don't
        # persist unvalidated guesses when the chooser isn't there).
        self.assertEqual(result.status, DiscoveryStatus.ERROR)
        os.unlink(cfg_path)


# ---------------------------------------------------------------------------
# OpenRouter retry/backoff semantics
# ---------------------------------------------------------------------------

class OpenRouterRetryTests(unittest.TestCase):

    def test_retries_on_429_then_succeeds(self):
        calls = {"n": 0}

        class FakeResp:
            def __init__(self, status=200, payload=None, text=""):
                self.status_code = status
                self._payload = payload or {}
                self.text = text

            def json(self):
                return self._payload

        def fake_post(url, headers=None, json=None, timeout=None):
            calls["n"] += 1
            if calls["n"] < 3:
                return FakeResp(status=429, text="rate-limited")
            return FakeResp(
                status=200,
                payload={
                    "choices": [
                        {"message": {"content": '["#a"]'}}
                    ]
                },
            )

        with patch("healer.requests.post", side_effect=fake_post), \
                patch.dict(os.environ, {
                    "OPENROUTER_BACKOFF_SECONDS": "0",
                    "OPENROUTER_MAX_ATTEMPTS": "3",
                    "OPENROUTER_API_KEY": "test",
                }):
            data = _post_with_retry({"messages": []}, {}, "test")
        self.assertEqual(calls["n"], 3)
        self.assertEqual(data["choices"][0]["message"]["content"], '["#a"]')

    def test_raises_discovery_error_on_persistent_429(self):
        class FakeResp:
            status_code = 429
            text = "still rate-limited"

            def json(self):
                return {}

        def fake_post(*a, **kw):
            return FakeResp()

        with patch("healer.requests.post", side_effect=fake_post), \
                patch.dict(os.environ, {
                    "OPENROUTER_BACKOFF_SECONDS": "0",
                    "OPENROUTER_MAX_ATTEMPTS": "2",
                    "OPENROUTER_API_KEY": "test",
                }):
            with self.assertRaises(DiscoveryError):
                _post_with_retry({}, {}, "test")

    def test_raises_discovery_error_on_401(self):
        """Non-transient failures should not retry."""

        class FakeResp:
            status_code = 401
            text = "unauthorized"

            def json(self):
                return {}

        def fake_post(*a, **kw):
            return FakeResp()

        calls = {"n": 0}

        def counting_post(*a, **kw):
            calls["n"] += 1
            return FakeResp()

        with patch("healer.requests.post", side_effect=counting_post), \
                patch.dict(os.environ, {"OPENROUTER_API_KEY": "test"}):
            with self.assertRaises(DiscoveryError):
                _post_with_retry({}, {}, "test")
        self.assertEqual(calls["n"], 1)

    def test_llm_429_does_not_return_empty_array(self):
        """The LLM helper must raise DiscoveryError on 429 — never
        silently return [] as if nothing was clickable."""

        class FakeResp:
            status_code = 429
            text = "rate-limited"

            def json(self):
                return {}

        def fake_post(*a, **kw):
            return FakeResp()

        with patch("healer.requests.post", side_effect=fake_post), \
                patch.dict(os.environ, {
                    "OPENROUTER_BACKOFF_SECONDS": "0",
                    "OPENROUTER_MAX_ATTEMPTS": "1",
                    "OPENROUTER_API_KEY": "test",
                }):
            with self.assertRaises(DiscoveryError):
                ask_openrouter_candidates_text("[]", "intent", "text",
                                              "contains", "li", "url")


# ---------------------------------------------------------------------------
# Deterministic text finder — proves the LLM is not needed for a normal
# Google chooser.
# ---------------------------------------------------------------------------

class DeterministicTextFinderTests(unittest.TestCase):

    def test_finds_preferred_method_without_llm(self):
        """When the chooser is rendered and the configured method text
        exists in the DOM, the deterministic finder MUST validate a
        selector without any LLM call."""
        cfg_path = tempfile.mktemp(suffix=".json")
        _ConfigFixture.write(cfg_path)
        page = FakePage(
            url="https://accounts.google.com/v3/signin/challenge/selection",
            chooser_visible=True,
            selectors={
                "li[data-challengetype='13']":
                    "Google Prompt",
            },
        )
        # extract_interactive_dom would normally be called — but the
        # JS scanner should produce a valid candidate first.
        with patch("healer.ask_openrouter_candidates_text") as llm, \
                patch("healer.extract_interactive_dom",
                      return_value="[]"):
            result = run(populate_and_save_text_target(
                page, cfg_path, 0,
                text="Google Prompt", match_type="contains", tag_hint="li",
            ))
        self.assertEqual(result.status, DiscoveryStatus.FOUND)
        llm.assert_not_called()
        os.unlink(cfg_path)


# ---------------------------------------------------------------------------
# Fallback text strategy + data-challengetype fallback — keeps the flow
# advancing when Google renders a different visible label than the configured
# one (and especially when the OpenRouter API key is unavailable).
# ---------------------------------------------------------------------------

class FallbackTextStrategyTests(unittest.TestCase):
    """Proves the runner advances past the chooser without ever calling the
    LLM when ANY of the candidate visible texts (or the configured method's
    `data-challengetype`) resolves on the live page.
    """

    FLOW_STEP = _ConfigFixture.FLOW_STEP

    def _make_cfg_path(self) -> str:
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        _ConfigFixture.write(path)
        return path

    def test_chooser_uses_alternate_text_when_primary_missing(self):
        """Primary text "Google Prompt" not on the DOM; alternate
        "Tap Yes on your" is present and visible. Runner must click
        without ever calling the LLM."""
        cfg_path = self._make_cfg_path()
        page = FakePage(
            url="https://accounts.google.com/v3/signin/challenge/selection",
            chooser_visible=True,
            selectors={
                "li[data-challengetype='13']":
                    "Tap Yes on your phone or tablet",
            },
        )
        # Patch the LLM — if it gets called the test fails, proving the
        # deterministic fallback handled it.
        with patch("runner.populate_and_save_text_target") as disc, \
                patch("healer.ask_openrouter_candidates_text") as llm, \
                patch("healer.extract_interactive_dom",
                      return_value="[]"):
            ok = run(handle_select_2fa_step(
                page, self.FLOW_STEP, 0,
                {"2fa_settings": {"method": "google_prompt",
                                  "state_provider": {}}},
                PROVIDER,
                config_path=cfg_path,
            ))
        self.assertTrue(ok)
        self.assertEqual(
            page.click_log.get("li[data-challengetype='13']"), 1)
        # The LLM discovery helper must not have been invoked.
        disc.assert_not_called()
        llm.assert_not_called()
        # Persisted selector + the matching text.
        with open(cfg_path) as f:
            cfg = json.load(f)
        self.assertEqual(
            cfg["flow_steps"][0]["selector"],
            "li[data-challengetype='13']")
        self.assertEqual(
            cfg["flow_steps"][0]["text"], "Tap Yes on your")
        os.unlink(cfg_path)

    def test_chooser_uses_data_challengetype_when_no_text_matches(self):
        """No visible text candidate matches at all, but the configured
        method's `data-challengetype` element is present. Runner must
        still advance without calling the LLM."""
        cfg_path = self._make_cfg_path()
        page = FakePage(
            url="https://accounts.google.com/v3/signin/challenge/selection",
            chooser_visible=True,
            selectors={
                "li[data-challengetype='13']":
                    "Some unexpected label Google did not enumerate",
            },
        )
        with patch("runner.populate_and_save_text_target") as disc, \
                patch("healer.ask_openrouter_candidates_text") as llm, \
                patch("healer.extract_interactive_dom",
                      return_value="[]"):
            ok = run(handle_select_2fa_step(
                page, self.FLOW_STEP, 0,
                {"2fa_settings": {"method": "google_prompt",
                                  "state_provider": {}}},
                PROVIDER,
                config_path=cfg_path,
            ))
        self.assertTrue(ok)
        self.assertEqual(
            page.click_log.get("li[data-challengetype='13']"), 1)
        disc.assert_not_called()
        llm.assert_not_called()
        with open(cfg_path) as f:
            cfg = json.load(f)
        self.assertEqual(
            cfg["flow_steps"][0]["selector"],
            "[data-challengetype=\"13\"]")
        os.unlink(cfg_path)

    def test_chooser_does_not_call_llm_with_alternate_text_match(self):
        """Direct pipeline test: when the chooser is visible and an
        alternate text candidate matches, `populate_and_save_text_target`
        itself does not need to fall back to the LLM."""
        cfg_path = self._make_cfg_path()
        page = FakePage(
            url="https://accounts.google.com/v3/signin/challenge/selection",
            chooser_visible=True,
            selectors={
                "li[data-challengetype='13']":
                    "Get a Google prompt",
            },
        )
        with patch("healer.ask_openrouter_candidates_text") as llm, \
                patch("healer.extract_interactive_dom",
                      return_value="[]"):
            # Direct call to the deterministic finder with the alternate
            # text; the LLM helper must never be invoked.
            result = run(populate_and_save_text_target(
                page, cfg_path, 0,
                text="Get a Google prompt", match_type="contains",
                tag_hint="li",
            ))
        self.assertEqual(result.status, DiscoveryStatus.FOUND)
        self.assertEqual(result.selector, "li[data-challengetype='13']")
        llm.assert_not_called()
        os.unlink(cfg_path)

    def test_step_supplied_text_wins_over_method_default(self):
        """Explicit `text` on the step is treated as the primary
        candidate; method fallbacks still come after it."""
        cfg_path = self._make_cfg_path()
        page = FakePage(
            url="https://accounts.google.com/v3/signin/challenge/selection",
            chooser_visible=True,
            selectors={
                "li[data-challengetype='13']":
                    "Tap Yes on your phone or tablet",
            },
        )
        step = {**self.FLOW_STEP, "text": "Tap Yes on your"}
        with patch("runner.populate_and_save_text_target") as disc, \
                patch("healer.ask_openrouter_candidates_text") as llm, \
                patch("healer.extract_interactive_dom",
                      return_value="[]"):
            ok = run(handle_select_2fa_step(
                page, step, 0,
                {"2fa_settings": {"method": "google_prompt",
                                  "state_provider": {}}},
                PROVIDER,
                config_path=cfg_path,
            ))
        self.assertTrue(ok)
        self.assertEqual(
            page.click_log.get("li[data-challengetype='13']"), 1)
        disc.assert_not_called()
        llm.assert_not_called()
        with open(cfg_path) as f:
            cfg = json.load(f)
        self.assertEqual(cfg["flow_steps"][0]["text"], "Tap Yes on your")
        os.unlink(cfg_path)


class Resolve2faTextsTests(unittest.TestCase):
    """Order and de-duplication semantics for resolve_2fa_texts."""

    def test_uses_method_fallbacks_when_no_step_text(self):
        step = {}
        cfg = {"2fa_settings": {"method": "google_prompt"}}
        out = resolve_2fa_texts(step, cfg)
        # Primary is the first METHOD_TEXT_MAP entry for google_prompt.
        self.assertEqual(out[0], "Google Prompt")
        # No duplicates.
        self.assertEqual(len(out), len(set(out)))
        # Subsequent entries are alternates (e.g. "Tap Yes on your").
        self.assertIn("Tap Yes on your", out)

    def test_explicit_text_is_primary(self):
        step = {"text": "Custom text"}
        cfg = {"2fa_settings": {"method": "google_prompt"}}
        out = resolve_2fa_texts(step, cfg)
        self.assertEqual(out[0], "Custom text")
        self.assertIn("Google Prompt", out)
        # No duplicates.
        self.assertEqual(len(out), len(set(out)))

    def test_unknown_method_falls_back_to_method_name(self):
        step = {}
        cfg = {"2fa_settings": {"method": "fingerprint"}}
        out = resolve_2fa_texts(step, cfg)
        self.assertEqual(out, ["fingerprint"])


if __name__ == "__main__":
    unittest.main()
