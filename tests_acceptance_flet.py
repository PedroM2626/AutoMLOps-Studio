import pytest
from playwright.sync_api import sync_playwright
import subprocess
import time
import os
import signal

@pytest.fixture(scope="module")
def flet_server():
    """Starts the flet app as a web server for testing."""
    # Start the flet app in web mode
    # We use a specific port to avoid conflicts
    process = subprocess.Popen(
        ["python", "simple_flet_app.py"],
        env={**os.environ, "FLET_SERVER_PORT": "8550", "FLET_VIEW": "web_browser"},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True if os.name == 'nt' else False
    )
    
    # Wait for the server to start
    time.sleep(5) 
    
    yield "http://localhost:8550"
    
    # Cleanup: kill the process
    if os.name == 'nt':
        subprocess.call(['taskkill', '/F', '/T', '/PID', str(process.pid)])
    else:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)

def test_acceptance_simple_flet_app(flet_server):
    """Acceptance test using Playwright to interact with the Flet web app."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        try:
            page.goto(flet_server)
            
            # Check if title exists
            # Flet web rendering uses a canvas or complex DOM, 
            # so we look for text content or specific aria-labels
            page.wait_for_selector("text=Interface de Teste Flet")
            
            # Click the button
            page.click("text=Verificar Status do Sistema")
            
            # Verify the status message changes or appears
            # Since mlruns likely exists in the environment:
            page.wait_for_selector("text=Sistema AutoML: Ativo")
            
            print("Acceptance test passed!")
        except Exception as e:
            pytest.fail(f"Acceptance test failed: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    # Note: Requires playwright to be installed: playwright install chromium
    pytest.main([__file__])
