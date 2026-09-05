import json
import time
from playwright.sync_api import sync_playwright, TimeoutError
from browser_setup import get_stealth_browser
from healer import repair_selector

def load_selectors():
    with open("selectors.json", "r") as f:
        return json.load(f)

def run_workflow():
    selectors = load_selectors()
    
    with sync_playwright() as p:
        context, page = get_stealth_browser(p)
        
        try:
            # 1. Deterministic Navigation
            page.goto(selectors["login_url"])
            
            # 2. Click Statements Tab with Exception Fallback
            try:
                page.locator(selectors["statements_tab"]).click(timeout=3000)
            except TimeoutError:
                # Triggers healing and retries with new selector
                repaired = repair_selector(page, "statements_tab", "Click the Statements or Documents link")
                page.locator(repaired).click()
                selectors = load_selectors()  # Refresh local memory

            # 3. Download Statement
            try:
                with page.expect_download(timeout=5000) as download_info:
                    page.locator(selectors["download_btn"]).click(timeout=3000)
                download = download_info.value
                download.save_as("C:\\BankRPA_Engine\\statement.pdf")
                print("[✓] Process completed deterministically.")
            except TimeoutError:
                repaired = repair_selector(page, "download_btn", "Click the Download Statement PDF button")
                with page.expect_download() as download_info:
                    page.locator(repaired).click()
                download = download_info.value
                download.save_as("C:\\BankRPA_Engine\\statement.pdf")
                print("[✓] Process completed via healed selector.")

        finally:
            context.close()

if __name__ == "__main__":
    run_workflow()