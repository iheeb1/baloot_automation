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

BUTTON_TEMPLATES = {
    "CLAIM": "claim_button_template.png",
    "AGREE": "mouwafeq_template.png",
    "BACK": "return_grey_template.png",
    "GIFTBOX": "gift_box_template.png"
}

GIFTBOX_THRESHOLD = 0.5
BUTTON_THRESHOLD = 0.7
CLICK_COOLDOWN = 10
DRAG_DELAY = 0.1


class BalootGiftBoxAutomation:
    def __init__(self):
        self.setup_chrome()
        self.load_templates()
        self.last_claim_time = 0
        self.running = False
        self.debug_folder = "giftbox_debug"
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
    
    def cleanup_screenshot(self, screenshot_path):
        """Delete screenshot after use"""
        try:
            if screenshot_path and os.path.exists(screenshot_path):
                os.remove(screenshot_path)
        except Exception as e:
            print(f"⚠️ Failed to delete screenshot: {e}")

    def detect_button(self, screenshot_path, btn_name):
        """Detect button using template matching"""
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

        if max_val >= BUTTON_THRESHOLD:
            center_x = max_loc[0] + w // 2
            center_y = max_loc[1] + h // 2
            return {"x": center_x, "y": center_y, "confidence": max_val}
        return None

    def detect_giftbox(self, screenshot_path):
        """
        Detect gift box using multi-scale template matching
        Returns dict with location info or None
        """
        if "GIFTBOX" not in self.templates:
            print("⚠️ Gift box template not loaded")
            return None

        screenshot = cv2.imread(screenshot_path)
        if screenshot is None:
            return None

        gray_screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
        gray_template = self.templates["GIFTBOX"]
        
        template_h, template_w = gray_template.shape
        screen_height, screen_width = gray_screenshot.shape

        # Try multiple scales
        scales = [0.8, 0.9, 1.0, 1.1, 1.2]
        all_matches = []

        for scale in scales:
            if scale != 1.0:
                new_w = int(template_w * scale)
                new_h = int(template_h * scale)
                resized_template = cv2.resize(gray_template, (new_w, new_h))
            else:
                resized_template = gray_template
                new_w, new_h = template_w, template_h

            # Skip if template is larger than screenshot
            if new_h > screen_height or new_w > screen_width:
                continue

            # Template matching
            result = cv2.matchTemplate(gray_screenshot, resized_template, cv2.TM_CCOEFF_NORMED)
            threshold_map = result >= (GIFTBOX_THRESHOLD * 0.8)
            locations = np.where(threshold_map)

            for pt in zip(*locations[::-1]):
                confidence = result[pt[1], pt[0]]
                all_matches.append({
                    'location': pt,
                    'confidence': confidence,
                    'scale': scale,
                    'width': new_w,
                    'height': new_h
                })

        if not all_matches:
            return None

        # Sort by confidence
        all_matches.sort(key=lambda x: x['confidence'], reverse=True)

        # Filter: must be on right side of screen
        right_side_matches = [m for m in all_matches if m['location'][0] > screen_width * 0.6]

        if not right_side_matches:
            right_side_matches = all_matches

        best_match = right_side_matches[0]

        if best_match['confidence'] >= GIFTBOX_THRESHOLD:
            loc = best_match['location']
            w = best_match['width']
            h = best_match['height']

            top_left = loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            center = (top_left[0] + w // 2, top_left[1] + h // 2)

            print(f"🎁 Gift box found! Confidence: {best_match['confidence']:.3f}, Scale: {best_match['scale']:.1f}x")
            
            return {
                "found": True,
                "confidence": best_match['confidence'],
                "top_left": top_left,
                "bottom_right": bottom_right,
                "center": center,
                "width": w,
                "height": h,
                "scale": best_match['scale']
            }
        
        return None

    def detect_path_in_giftbox(self, screenshot_path, giftbox_info):
        """
        Detect the path inside the gift box area
        Returns list of path points for dragging
        """
        if not giftbox_info or not giftbox_info["found"]:
            return None

        screenshot = cv2.imread(screenshot_path)
        if screenshot is None:
            return None

        # Get original boundaries
        orig_x1, orig_y1 = giftbox_info["top_left"]
        orig_x2, orig_y2 = giftbox_info["bottom_right"]

        # Adjust boundaries: move down 100px, up 50px
        x1 = orig_x1
        y1 = orig_y1 + 100
        x2 = orig_x2
        y2 = orig_y2 - 50

        # Validate boundaries
        y1 = max(0, y1)
        y2 = min(screenshot.shape[0], y2)

        if y2 <= y1:
            print("⚠️ Invalid adjusted boundaries")
            return None

        # Extract ROI
        roi = screenshot[y1:y2, x1:x2].copy()
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Binary threshold to detect dark path
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

        # Clean up mask
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            print("⚠️ No path found, trying HSV fallback...")
            return self._detect_path_hsv_fallback(roi, x1, y1)

        # Get largest contour
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)

        if area < 100:  # Path too small
            print("⚠️ Path too small, trying HSV fallback...")
            return self._detect_path_hsv_fallback(roi, x1, y1)

        # Extract and sort points
        points = [tuple(pt[0]) for pt in contour]
        points_sorted = sorted(points, key=lambda p: p[0])

        # Sample points for smooth dragging
        sample_rate = max(1, len(points_sorted) // 20)
        sampled_points = points_sorted[::sample_rate]

        # Convert to global coordinates
        global_points = [(p[0] + x1, p[1] + y1) for p in sampled_points]

        print(f"🛤️ Path detected: {len(global_points)} points")
        print(f"   Start: {global_points[0]}, End: {global_points[-1]}")

        return global_points

    def _detect_path_hsv_fallback(self, roi, offset_x, offset_y):
        """Fallback method using HSV color detection"""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Brown/tan color range
        lower_brown = np.array([10, 20, 100])
        upper_brown = np.array([30, 150, 200])

        mask = cv2.inRange(hsv, lower_brown, upper_brown)

        # Clean up
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            print("❌ HSV fallback also failed")
            return None

        contour = max(contours, key=cv2.contourArea)
        points = [tuple(pt[0]) for pt in contour]
        points_sorted = sorted(points, key=lambda p: p[0])

        sample_rate = max(1, len(points_sorted) // 20)
        sampled_points = points_sorted[::sample_rate]
        global_points = [(p[0] + offset_x, p[1] + offset_y) for p in sampled_points]

        print(f"✅ HSV fallback found {len(global_points)} points")
        return global_points

    def perform_drag_on_path(self, path_points):
        """Drag mouse along the detected path"""
        if not path_points or len(path_points) < 2:
            print("⚠️ Not enough path points to drag")
            return False

        start_pt = path_points[0]
        end_pt = path_points[-1]

        try:
            # Create smooth drag path using intermediate points
            script = """
            var canvas = document.getElementById('unity-canvas');
            var points = %s;
            var currentIndex = 0;
            
            // Start drag
            var evt = new PointerEvent("pointerdown", {
                bubbles: true, cancelable: true, view: window,
                clientX: points[0][0], clientY: points[0][1]
            });
            canvas.dispatchEvent(evt);
            
            // Move through points
            var interval = setInterval(function() {
                currentIndex++;
                if (currentIndex >= points.length) {
                    clearInterval(interval);
                    
                    // End drag
                    var evtUp = new PointerEvent("pointerup", {
                        bubbles: true, cancelable: true, view: window,
                        clientX: points[points.length-1][0], 
                        clientY: points[points.length-1][1]
                    });
                    canvas.dispatchEvent(evtUp);
                    return;
                }
                
                var evtMove = new PointerEvent("pointermove", {
                    bubbles: true, cancelable: true, view: window,
                    clientX: points[currentIndex][0], 
                    clientY: points[currentIndex][1]
                });
                canvas.dispatchEvent(evtMove);
            }, %d);
            """ % (str(path_points), int(DRAG_DELAY * 1000))

            self.driver.execute_script(script)
            print(f"🖱️ Dragged along path from {start_pt} to {end_pt}")
            return True

        except Exception as e:
            print(f"❌ Drag error: {e}")
            return False

    def click_at(self, x, y):
        """Click at coordinates using PointerEvent"""
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
            print(f"❌ Click failed: {e}")

    def create_control_panel(self):
        """Inject control panel at TOP LEFT"""
        script = """
        (() => {
            const oldPanel = document.getElementById('auto_control_panel');
            if (oldPanel) oldPanel.remove();

            const panel = document.createElement('div');
            panel.id = 'auto_control_panel';
            panel.style.cssText = `
                position: fixed;
                top: 20px;
                left: 20px;
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

            const title = document.createElement('div');
            title.textContent = '🎁 BALOOT BOT';
            title.style.cssText = 'text-align:center; font-weight:bold; color:#00bfff; margin-bottom:12px; font-size:16px;';
            panel.appendChild(title);

            const status = document.createElement('div');
            status.id = 'bot_status';
            status.textContent = 'STOPPED';
            status.style.cssText = 'color:yellow; text-align:center; margin-bottom:16px; font-weight:bold;';
            panel.appendChild(status);

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

            console.log('✅ Panel Injected (Top Left)');
        })();
        """

        try:
            self.driver.execute_script(script)
            print("✅ Control panel injected at TOP LEFT")
            self.panel_injected = True
        except Exception as e:
            print(f"❌ Panel injection failed: {e}")

    def check_command(self):
        """Read and clear command"""
        try:
            cmd = self.driver.execute_script("return window.botCommand;")
            if cmd:
                self.driver.execute_script("window.botCommand = undefined;")
                return cmd
            return None
        except:
            return None

    def update_status(self, status):
        """Update panel status"""
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
        """Re-inject panel if missing"""
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
        """Main automation loop"""
        print("🤖 Waiting for START command...")
        last_repair = 0
        last_screenshot_time = 0

        while True:
            # Repair panel periodically
            if time.time() - last_repair > 5:
                self.repair_panel_if_needed()
                last_repair = time.time()

            # Check commands
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
                current_time = time.time()
                
                # Take screenshot every 1 second
                if current_time - last_screenshot_time < 1.0:
                    time.sleep(0.1)
                    continue
                
                screenshot_path = self.take_screenshot()
                last_screenshot_time = current_time
                
                if not screenshot_path:
                    time.sleep(1)
                    continue

                # STEP 1: Look for CLAIM button
                claim_btn = self.detect_button(screenshot_path, "CLAIM")
                if claim_btn and (current_time - self.last_claim_time) > CLICK_COOLDOWN:
                    print("🎯 Found CLAIM button! Clicking...")
                    self.click_at(claim_btn["x"], claim_btn["y"])
                    self.last_claim_time = current_time
                    time.sleep(1)

                    # STEP 2: After CLAIM, look for GIFT BOX (PRIORITY)
                    print("🔍 Searching for gift box...")
                    giftbox_info = self.detect_giftbox(screenshot_path)
                    
                    if giftbox_info and giftbox_info["found"]:
                        # STEP 3: Detect path inside gift box
                        path_points = self.detect_path_in_giftbox(screenshot_path, giftbox_info)
                        
                        if path_points:
                            # STEP 4: Drag along path
                            self.perform_drag_on_path(path_points)
                            time.sleep(1)
                        else:
                            print("⚠️ No path detected in gift box")
                    else:
                        print("⚠️ Gift box not found after CLAIM")

                # STEP 5: Handle popups (AGREE/BACK)
                for btn_name, label in [("AGREE", "موافق"), ("BACK", "عودة")]:
                    btn = self.detect_button(screenshot_path, btn_name)
                    if btn:
                        print(f"✅ Found '{label}' button!")
                        self.click_at(btn["x"], btn["y"])
                        time.sleep(1)

                # Delete screenshot after processing
                self.cleanup_screenshot(screenshot_path)

            except Exception as e:
                print(f"❌ Loop error: {e}")
                time.sleep(2)

    def start(self):
        """Start the bot"""
        print("🚀 Starting Baloot Gift Box Automation...")
        
        self.driver.get("https://kammelna.com/baloot/")

        try:
            self.canvas = WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.ID, "unity-canvas"))
            )
            print("🎨 Canvas found.")
        except Exception as e:
            print("❌ Canvas not found:", e)
            return

        print("🔧 Injecting control panel...")
        self.create_control_panel()
        time.sleep(1)

        thread = threading.Thread(target=self.run_automation, daemon=True)
        thread.start()

        print("🎛️ Control panel active (TOP LEFT). Press START to begin.")

        try:
            while thread.is_alive():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n👋 Stopping bot...")
            self.running = False
            self.driver.quit()


def main():
    parser = argparse.ArgumentParser(description="Baloot Automation with Gift Box Path Detection")
    parser.add_argument("--claim", default="claim_button_template.png")
    parser.add_argument("--agree", default="mouwafeq_template.png")
    parser.add_argument("--back", default="return_grey_template.png")
    parser.add_argument("--giftbox", default="gift_box_template.png")
    args = parser.parse_args()

    BUTTON_TEMPLATES["CLAIM"] = args.claim
    BUTTON_TEMPLATES["AGREE"] = args.agree
    BUTTON_TEMPLATES["BACK"] = args.back
    BUTTON_TEMPLATES["GIFTBOX"] = args.giftbox

    bot = BalootGiftBoxAutomation()
    bot.start()


if __name__ == "__main__":
    main()