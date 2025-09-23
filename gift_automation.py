import cv2
import numpy as np
import os
import time
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import threading
import argparse

# --- Configuration ---
BUTTON_TEMPLATES = {
    "CLAIM": "claim_button_template.png",      # استلم
    "AGREE": "mouwafeq_template.png",          # موافق
    "BACK": "return_grey_template.png"         # عودة
}

ZONE_OF_INTEREST = {
    "x1": 1205, "y1": 375,
    "x2": 1480, "y2": 560
}

THRESHOLD = 0.7
DRAG_SPEED = 0.2
CLICK_COOLDOWN = 10


class BalootSmartAutomation:
    def __init__(self):
        self.setup_chrome()
        self.load_templates()
        self.last_claim_time = 0
        self.running = False
        self.debug_folder = "final_debug_screenshots"
        os.makedirs(self.debug_folder, exist_ok=True)
        self.canvas = None
        self.panel_injected = False

    def setup_chrome(self):
        options = Options()
        options.add_argument("--start-maximized")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-infobars")
        options.add_experimental_option("useAutomationExtension", False)
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => false});")

    def load_templates(self):
        self.templates = {}
        for key, path in BUTTON_TEMPLATES.items():
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is not None:
                    self.templates[key] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    print(f"✅ Loaded template: {key}")
                else:
                    print(f"❌ Failed to load image: {path}")
            else:
                print(f"⚠️ Template missing: {path}")

    def take_screenshot(self):
        path = f"{self.debug_folder}/screen_{int(time.time())}.png"
        try:
            self.driver.save_screenshot(path)
            return path
        except Exception as e:
            print(f"Screenshot failed: {e}")
            return None

    def detect_button(self, screenshot_path, btn_name):
        if btn_name not in self.templates:
            return None
        if not os.path.exists(screenshot_path):
            return None

        screenshot = cv2.imread(screenshot_path)
        if screenshot is None:
            return None

        gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        template = self.templates[btn_name]
        h, w = template.shape

        result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= THRESHOLD:
            center_x = max_loc[0] + w // 2
            center_y = max_loc[1] + h // 2
            return {"x": center_x, "y": center_y, "confidence": max_val}
        return None

    def perform_drag(self):
        """Simulate drag using JavaScript on canvas"""
        screenshot_path = self.take_screenshot()
        if not screenshot_path:
            return False

        screenshot = cv2.imread(screenshot_path)
        if screenshot is None:
            return False

        x1, y1, x2, y2 = [ZONE_OF_INTEREST[k] for k in ("x1", "y1", "x2", "y2")]
        roi = screenshot[y1:y2, x1:x2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            return False

        contour = max(contours, key=cv2.contourArea)
        points = sorted([tuple(pt[0]) for pt in contour], key=lambda p: p[0])
        global_points = [(p[0] + x1, p[1] + y1) for p in points[::15]]

        if len(global_points) < 2:
            return False

        start_pt, end_pt = global_points[0], global_points[-1]

        # Perform drag via JS on Unity canvas
        try:
            script = """
            var canvas = document.getElementById('unity-canvas');
            var evt;

            // Mouse down
            evt = new PointerEvent("pointerdown", {
                bubbles: true, cancelable: true, view: window,
                clientX: %d, clientY: %d
            });
            canvas.dispatchEvent(evt);

            // Mouse move (drag)
            setTimeout(function() {
                evt = new PointerEvent("pointermove", {
                    bubbles: true, cancelable: true, view: window,
                    clientX: %d, clientY: %d
                });
                canvas.dispatchEvent(evt);
            }, 100);

            // Mouse up
            setTimeout(function() {
                evt = new PointerEvent("pointerup", {
                    bubbles: true, cancelable: true, view: window,
                    clientX: %d, clientY: %d
                });
                canvas.dispatchEvent(evt);
            }, 150);
            """ % (start_pt[0], start_pt[1], end_pt[0], end_pt[1], end_pt[0], end_pt[1])

            self.driver.execute_script(script)
            print("🖱️ Drag simulated on canvas.")
            return True
        except Exception as e:
            print(f"Drag error: {e}")
            return False

    def click_at(self, x, y):
        """Click using PointerEvent at canvas-relative coordinates"""
        try:
            script = """
            var canvas = document.getElementById('unity-canvas');
            var evt = new PointerEvent('click', {
                bubbles: true,
                cancelable: true,
                view: window,
                clientX: %d,
                clientY: %d
            });
            canvas.dispatchEvent(evt);
            """ % (x, y)
            self.driver.execute_script(script)
            print(f"🖱️ Clicked at ({x}, {y})")
            time.sleep(0.5)
        except Exception as e:
            print(f"Click failed: {e}")

    def create_control_panel(self):
        """Inject persistent control panel into the page"""
        script = """
        (() => {
            // Remove old panel if exists
            const oldPanel = document.getElementById('auto_control_panel');
            if (oldPanel) oldPanel.remove();

            // Create new panel
            const panel = document.createElement('div');
            panel.id = 'auto_control_panel';
            panel.style.cssText = `
                position: fixed;
                top: 50%;
                right: 20px;
                transform: translateY(-50%);
                background: #1a1c2c;
                border: 2px solid #00bfff;
                border-radius: 12px;
                width: 220px;
                padding: 16px;
                z-index: 9999999;
                color: white;
                font-family: 'Segoe UI', Arial, sans-serif;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.6);
            `;

            // Title
            const title = document.createElement('div');
            title.textContent = '🎮 BALOOT BOT';
            title.style.cssText = 'text-align:center; font-weight:bold; color:#00bfff; margin-bottom:12px; font-size:16px;';
            panel.appendChild(title);

            // Status
            const status = document.createElement('div');
            status.id = 'bot_status';
            status.textContent = 'STOPPED';
            status.style.cssText = 'color:yellow; text-align:center; margin-bottom:16px; font-weight:bold;';
            panel.appendChild(status);

            // Buttons
            const startBtn = document.createElement('button');
            startBtn.textContent = '▶️ START';
            startBtn.style.cssText = 'width:100%; background:#00aa00; color:white; border:none; padding:10px; margin-bottom:10px; border-radius:6px; cursor:pointer; font-size:14px;';
            startBtn.onclick = () => { window.botCommand = 'START'; };

            const stopBtn = document.createElement('button');
            stopBtn.textContent = '⏹️ STOP';
            stopBtn.style.cssText = 'width:100%; background:#cc0000; color:white; border:none; padding:10px; border-radius:6px; cursor:pointer; font-size:14px;';
            stopBtn.onclick = () => { window.botCommand = 'STOP'; };

            panel.appendChild(startBtn);
            panel.appendChild(stopBtn);
            document.body.appendChild(panel);

            console.log('✅ Baloot Bot Panel Injected');
        })();
        """

        try:
            self.driver.execute_script(script)
            print("✅ Control panel injected successfully.")
            self.panel_injected = True
        except Exception as e:
            print(f"❌ Panel injection failed: {e}")
            self.panel_injected = False

    def check_command(self):
        """Safely read and clear command"""
        try:
            cmd = self.driver.execute_script("return window.botCommand;")
            if cmd:
                self.driver.execute_script("window.botCommand = undefined;")
                return cmd
            return None
        except:
            return None

    def update_status(self, status):
        """Update panel status text and color"""
        colors = {"RUNNING": "#00ff00", "STOPPED": "yellow", "ERROR": "orange"}
        color = colors.get(status, "white")
        script = f"""
        ((status, color) => {{
            const el = document.getElementById('bot_status');
            if (el) {{
                el.textContent = status;
                el.style.color = color;
            }}
        }})('{status}', '{color}');
        """
        try:
            self.driver.execute_script(script)
        except:
            pass

    def repair_panel_if_needed(self):
        """Re-inject panel if removed by game logic"""
        try:
            exists = self.driver.execute_script("""
                return !!document.getElementById('auto_control_panel');
            """)
            if not exists:
                print("🔁 Panel missing! Re-injecting...")
                self.create_control_panel()
        except:
            pass

    def run_automation(self):
        print("🤖 Waiting for START command...")

        while True:
            # Repair panel every few seconds
            if self.running or not hasattr(self, '_last_repair') or time.time() - self._last_repair > 5:
                self.repair_panel_if_needed()
                self._last_repair = time.time()

            cmd = self.check_command()
            if cmd == "START" and not self.running:
                self.running = True
                self.update_status("RUNNING")
                print("▶️ Automation started!")
            elif cmd == "STOP":
                self.running = False
                self.update_status("STOPPED")
                print("⏹️ Automation stopped.")

            if not self.running:
                time.sleep(0.5)
                continue

            try:
                screenshot_path = self.take_screenshot()
                if not screenshot_path:
                    time.sleep(1)
                    continue

                # Perform drag action
                self.perform_drag()

                current_time = time.time()

                # Handle CLAIM button with cooldown
                claim_btn = self.detect_button(screenshot_path, "CLAIM")
                if claim_btn and (current_time - self.last_claim_time) > CLICK_COOLDOWN:
                    print("🎯 Found 'استلم' button! Clicking...")
                    self.click_at(claim_btn["x"], claim_btn["y"])
                    self.last_claim_time = current_time
                    time.sleep(1)

                # Check other buttons
                for btn_name, label in [("AGREE", "موافق"), ("BACK", "عودة")]:
                    btn = self.detect_button(screenshot_path, btn_name)
                    if btn:
                        print(f"✅ Found '{label}' button!")
                        self.click_at(btn["x"], btn["y"])
                        time.sleep(1)

                time.sleep(1.5)

            except Exception as e:
                print(f"❌ Loop error: {e}")
                time.sleep(2)

    def start(self):
        print("🚀 Starting Baloot Automation...")
        
        # Open correct URL (fixed trailing space)
        self.driver.get("https://kammelna.com/baloot/")  # <-- Removed extra spaces

        try:
            # Wait for canvas to ensure Unity loaded
            self.canvas = WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.ID, "unity-canvas"))
            )
            print("🎨 Canvas found.")
        except Exception as e:
            print("❌ Canvas not found:", e)
            return

        # Inject panel after page loads
        print("🔧 Injecting control panel...")
        self.create_control_panel()
        time.sleep(1)

        # Run automation loop in background
        thread = threading.Thread(target=self.run_automation, daemon=True)
        thread.start()

        print("🎛️ Control panel active. Press START to begin automation.")

        try:
            while thread.is_alive():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n👋 Stopping bot gracefully...")
            self.running = False
            self.driver.quit()


def main():
    parser = argparse.ArgumentParser(description="Baloot Automation with Reliable UI Controls")
    parser.add_argument("--claim", default="claim_button_template.png", help="Claim button template")
    parser.add_argument("--agree", default="mouwafeq_template.png", help="Agree button template")
    parser.add_argument("--back", default="return_grey_template.png", help="Back button template")
    args = parser.parse_args()

    BUTTON_TEMPLATES["CLAIM"] = args.claim
    BUTTON_TEMPLATES["AGREE"] = args.agree
    BUTTON_TEMPLATES["BACK"] = args.back

    bot = BalootSmartAutomation()
    bot.start()


if __name__ == "__main__":
    main()