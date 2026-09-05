import os
from playwright.sync_api import sync_playwright
from playwright_stealth import stealth_sync

USER_DATA_DIR = os.path.join(os.environ["LOCALAPPDATA"], "RPA_Persistent_Profile")

def get_stealth_browser(playwright_instance):
    """
    Launches a headed Chromium instance using a persistent local profile.
    Strips CDP automation artifacts to bypass perimeter bot gates.
    """
    context = playwright_instance.chromium.launch_persistent_context(
        user_data_dir=USER_DATA_DIR,
        headless=False,
        channel="chrome",  # Uses local Chrome binary if available
        args=[
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-infobars",
            "--start-maximized"
        ],
        viewport=None
    )
    page = context.pages[0] if context.pages else context.new_page()
    stealth_sync(page)
    return context, page