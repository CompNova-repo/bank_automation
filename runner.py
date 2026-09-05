import os
import json
import asyncio
import nodriver as uc
from healer import heal_and_update_config, populate_and_save_selector

USER_DATA_DIR = os.path.join(os.environ.get("LOCALAPPDATA", "C:\\Temp"), "Test_RPA_Profile")
CONFIG_PATH = "config.json"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

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

async def run_automation():
    config = load_config()
    print(f"[*] Starting Flow: {config['bank_name']}")

    # Ensure env vars exist before starting
    email_val = os.environ.get("TEST_GMAIL_USER")
    pass_val = os.environ.get("TEST_GMAIL_PASS")
    if not email_val:
        print("[!] ERROR: TEST_GMAIL_USER environment variable is not set!")
        return

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

            if action == "sleep":
                print(f"[>] Waiting {step.get('duration', 2)}s...")
                await page.sleep(step.get("duration", 2))
                continue

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