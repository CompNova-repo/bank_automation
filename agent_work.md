Yes, you can initialize your configuration with **semantic intent placeholders** (empty selectors) and let the AI populate the technical details at runtime on the first run.

Instead of hardcoding brittle CSS, you define *what* you want to accomplish (e.g., `"Enter bank username"`). On run 1, the AI agent inspects the live page DOM, discovers the appropriate elements, fills the JSON, and executes the actions. From run 2 onwards, the runner executes the saved JSON deterministically without calling the model.

---

### Step 1: The "Intent-Only" JSON (`config.json`)

You only supply the high-level intent, action, and any environment variable keys:

```json
{
  "bank_name": "Google Account Test Flow",
  "login_url": "https://accounts.google.com/signin/v2/identifier?flowName=GlifWebSignIn&flowEntry=ServiceLogin",
  "download_dir": "C:\\Temp",
  "timeout_seconds": 15,
  "flow_steps": [
    {
      "step_name": "enter_email",
      "intent": "The input field where the user enters their email or phone username",
      "action": "type",
      "selector": "",
      "value_env": "TEST_GMAIL_USER"
    },
    {
      "step_name": "click_next_email",
      "intent": "The primary Next or Continue button after entering the email address",
      "action": "click",
      "selector": ""
    },
    {
      "step_name": "wait_for_password_screen",
      "action": "sleep",
      "duration": 4
    },
    {
      "step_name": "enter_password",
      "intent": "The password input field where the user enters their password",
      "action": "type",
      "selector": "",
      "value_env": "TEST_GMAIL_PASS"
    },
    {
      "step_name": "click_next_password",
      "intent": "The primary Next or Sign in button after entering the password",
      "action": "click",
      "selector": ""
    }
  ]
}

```

---

### Step 2: Update `healer.py` to Resolve from Intent

Make the AI query depend on the high-level `intent` rather than a broken selector:

```python
import os
import json
import requests

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL_NAME = "qwen/qwen3.7-flash"

async def extract_interactive_dom(page):
    """
    Extracts structured interactive element information with visible text and container IDs.
    """
    js_extract = """
    () => {
        const interactive = Array.from(document.querySelectorAll('input, button, a, [role="button"], [tabindex], div[id], span[id]'));
        return interactive.map(el => {
            const r = el.getBoundingClientRect();
            const visible = r.width > 0 && r.height > 0;
            return {
                tag: el.tagName.toLowerCase(),
                id: el.id || '',
                name: el.getAttribute('name') || '',
                type: el.getAttribute('type') || '',
                placeholder: el.getAttribute('placeholder') || '',
                ariaLabel: el.getAttribute('aria-label') || '',
                innerText: (el.innerText || el.textContent || '').trim().slice(0, 60),
                className: el.className ? String(el.className).slice(0, 40) : '',
                visible: visible
            };
        }).filter(item => item.visible && (item.id || item.name || item.innerText || item.ariaLabel || item.placeholder));
    }
    """
    try:
        data = await page.evaluate(js_extract, return_by_value=True)
        if isinstance(data, dict) and "value" in data:
            data = data["value"]
        return json.dumps(data[:50], indent=2)
    except Exception:
        html = await page.get_content()
        return html[:3000]

def discover_selector(dom_context: str, intent: str) -> str:
    """
    Queries the LLM to locate the single best CSS selector for an empty/missing step intent.
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
    Discovers element matching the step's intent, writes it to the JSON file, and returns it.
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    step = config["flow_steps"][step_index]
    intent = step.get("intent", step["step_name"])

    print(f"\n[🧠 AI Discovery] Searching for element matching intent: '{intent}'...")
    dom = await extract_interactive_dom(page)
    new_selector = discover_selector(dom, intent)
    print(f"[🧠 AI Discovery] Discovered and mapped: '{new_selector}'")

    # Persist the discovered selector directly into config.json
    config["flow_steps"][step_index]["selector"] = new_selector
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return new_selector

```

---

### Step 3: Self-Populating `runner.py`

Modify `runner.py` to check if `selector` is empty or fails, automatically invoking discovery:

```python
import os
import json
import asyncio
import nodriver as uc
from healer import populate_and_save_selector

USER_DATA_DIR = os.path.join(os.environ.get("LOCALAPPDATA", "C:\\Temp"), "Test_RPA_Profile")
CONFIG_PATH = "config.json"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

async def run_automation():
    config = load_config()
    print(f"[*] Starting Flow: {config['bank_name']}")

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
            selector = step.get("selector", "").strip()

            # 1. Self-Discovery on First Run (Empty Selector)
            if not selector:
                print(f"[?] Step '{step_name}' has no selector. Auto-discovering...")
                selector = await populate_and_save_selector(page, CONFIG_PATH, idx)
                step["selector"] = selector

            # 2. Execution with Automatic Retry/Healing
            element = None
            for attempt in range(2):
                try:
                    element = await page.select(selector, timeout=6)
                    if not element:
                        raise TimeoutError(f"Selector '{selector}' timed out.")
                    break
                except Exception:
                    if attempt == 0:
                        print(f"[!] Step '{step_name}' selector failed. Re-discovering...")
                        selector = await populate_and_save_selector(page, CONFIG_PATH, idx)
                        step["selector"] = selector
                    else:
                        raise RuntimeError(f"Step '{step_name}' could not be resolved.")

            # 3. Action Execution
            if action == "click":
                await element.click()
                await page.sleep(2)

            elif action == "type":
                val = os.environ.get(step.get("value_env", ""), "")
                await element.click()
                await page.sleep(0.3)
                await element.send_keys(val)
                await page.sleep(0.5)

        print("[OK] Automation sequence complete. All selectors stored in config.json.")
        await page.sleep(5)

    finally:
        browser.stop()

if __name__ == "__main__":
    uc.loop().run_until_complete(run_automation())

```

---

### How This Works in Practice

1. **You provide an empty JSON** containing only intent descriptions (`"intent": "The input field where username is entered"`).
2. **On Run 1 (Discovery Mode):** The runner sees `""`, inspects the bank's live DOM, queries the AI for the precise selector, fills in `"selector": "#real-bank-user-id"`, saves the file, and executes the step.


3. **On Run 2+ (Deterministic Mode):** The runner reads the populated `config.json` directly, executing at full speed with **0 tokens and 0 AI overhead**.


4. **On Website Change (Healing Mode):** If a bank redesign breaks a stored selector, the exception handler triggers `populate_and_save_selector` to repair it dynamically.