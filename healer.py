import json
from lavague.core import ActionEngine, WorldModel
from lavague.core.agents import WebAgent
from lavague.drivers.playwright import PlaywrightDriver

def repair_selector(page, logical_key: str, user_intent: str) -> str:
    print(f"\n[!] Element for '{logical_key}' failed. Initiating AI Diagnostic...")
    
    # 1. Initialize LaVague Driver on current active page
    driver = PlaywrightDriver(page=page)
    world_model = WorldModel()
    action_engine = ActionEngine(driver=driver)
    agent = WebAgent(world_model=world_model, action_engine=action_engine)
    
    # 2. Query LaVague to identify the element for the given intent
    action_result = agent.run_step(user_intent)
    
    # 3. Extract the new CSS selector from the generated action code
    generated_code = action_result.code
    # Example heuristic extraction from generated Playwright locator
    new_selector = generated_code.split('locator("')[-1].split('")')[0] if 'locator("' in generated_code else ""
    
    if new_selector:
        # 4. Update the JSON registry for all future runs
        with open("selectors.json", "r") as f:
            selectors = json.load(f)
        selectors[logical_key] = new_selector
        with open("selectors.json", "w") as f:
            json.dump(selectors, f, indent=2)
        print(f"[+] Successfully healed selector for '{logical_key}' -> '{new_selector}'")
        return new_selector
    else:
        raise RuntimeError(f"Self-healing could not determine element for {logical_key}")