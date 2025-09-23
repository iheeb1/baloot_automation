import cv2
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import time
import os
import re
import pytesseract
from PIL import Image
from datetime import datetime
import pyautogui
import shutil

class RobustBalootAutomation:
    def __init__(self):
        self.chrome_options = Options()
        self.chrome_options.add_argument("--start-maximized")
        self.chrome_options.add_argument("--disable-web-security")
        self.chrome_options.add_argument("--disable-features=VizDisplayCompositor")
        self.chrome_options.add_argument("--force-device-scale-factor=1")
        self.chrome_options.add_argument("--disable-extensions")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=self.chrome_options
        )
        self.canvas = None
        self.canvas_rect = None
        self.debug_overlay_id = "baloot_debug_overlay"
        self.automation_running = False

        self.debug_folder = "debug_screenshots"
        if not os.path.exists(self.debug_folder):
            os.makedirs(self.debug_folder)

        self.ocr_available = self.test_ocr()

        self.templates = self.load_templates()


    def cleanup_debug_folder(self):
        """Delete all files in debug_screenshots folder"""
        try:
            for filename in os.listdir(self.debug_folder):
                file_path = os.path.join(self.debug_folder, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            self.update_debug_overlay("🗑️ Debug folder cleared.")
        except Exception as e:
            self.update_debug_overlay(f"❌ Failed to clear debug folder: {e}")
    def load_templates(self):
        """Load template images for button matching."""
        template_files = {
            "PLAY_BALOOT": "play_baloot_template.png",
            "RETURN_GREEN": "return_template.png",      
            "RETURN_GREY": "return_grey_template.png",  
            "LEAVE_GAME": "leave_game_template.png",
            "GREEN_PARTICIPATE": "green_participate_button.png"  
        }
        templates = {}
        for state, filename in template_files.items():
            if os.path.exists(filename):
                template = cv2.imread(filename, cv2.IMREAD_COLOR)
                if template is not None:
                    templates[state] = template
                    print(f"✅ Loaded template for {state}: {filename}")
                else:
                    print(f"❌ Failed to load image: {filename}")
            else:
                print(f"⚠️ Template file not found: {filename}")
        if not templates:
            print("⚠️ No templates loaded. Falling back to OCR/Visual methods.")
        return templates

    def test_ocr(self):
        """Test if OCR is available and working"""
        try:
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False

    def get_timestamp(self):
        """Get formatted timestamp for files and logs"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_debug_screenshot(self, action, detection_result=None):
        """Save screenshot with detection annotations for debugging"""
        timestamp = self.get_timestamp()
        filename = f"{self.debug_folder}/{timestamp}_{action}.png"
        try:
            self.driver.save_screenshot(filename)
            if detection_result and detection_result.get("button_location"):
                self.annotate_screenshot(filename, detection_result)
            self.update_debug_overlay(f"Debug screenshot: {filename}")
            return filename
        except Exception as e:
            self.update_debug_overlay(f"Screenshot error: {e}")
            return None

    def annotate_screenshot(self, screenshot_path, detection_result):
        """Add annotations to show what was detected"""
        try:
            img = cv2.imread(screenshot_path)
            if img is None:
                return
            x, y = detection_result["button_location"]
            state = detection_result.get("state", "UNKNOWN")
            confidence = detection_result.get("confidence", 0)

            cv2.circle(img, (x, y), 15, (0, 0, 255), -1)
            cv2.circle(img, (x, y), 25, (255, 255, 255), 3)

            label = f"{state} ({confidence:.2f})"
            cv2.putText(img, label, (x - 50, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            timestamp = datetime.now().strftime("%H:%M:%S")
            cv2.putText(img, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imwrite(screenshot_path, img)
        except Exception as e:
            self.update_debug_overlay(f"Annotation error: {e}")

    def click_button_at_position(self, x, y, detection_result):
        """Click using PyAutoGUI with screen offset correction"""
        self.update_debug_overlay(f"Clicking at canvas coordinates ({x}, {y})")
        self.update_automation_status("CLICKING")

        self.show_click_indicator(x, y, "red", 3000)

        self.save_debug_screenshot("before_click", detection_result)

        try:
            window_pos = self.driver.get_window_position()
            canvas_x = self.canvas_rect['x']
            canvas_y = self.canvas_rect['y']

            screen_x = window_pos['x'] + canvas_x + x
            screen_y = window_pos['y'] + canvas_y + y

            chrome_top_bar_height = 130
            screen_y += chrome_top_bar_height

            self.update_debug_overlay(f"Absolute screen coords: ({screen_x}, {screen_y})")

            pyautogui.moveTo(screen_x, screen_y, duration=0.2)
            time.sleep(0.05)
            pyautogui.mouseDown()
            time.sleep(0.05)
            pyautogui.mouseUp()

            self.show_click_indicator(x, y, "lime", 2000)
            self.update_debug_overlay("✅ REAL mouse click successful!")
            time.sleep(2)
            self.save_debug_screenshot("after_click")
        except Exception as e:
            self.update_debug_overlay(f"❌ PyAutoGUI click failed: {e}")
            self.show_click_indicator(x, y, "orange", 2000)
            self.fallback_click(x, y)

    def fallback_click(self, x, y):
        """Fallback: use JavaScript dispatch if PyAutoGUI fails"""
        try:
            self.update_debug_overlay("⚠️ Fallback: Using JavaScript click")
            script = f"""
            var evt = new MouseEvent('click', {{
                bubbles: true,
                cancelable: true,
                view: window,
                clientX: {x},
                clientY: {y}
            }});
            arguments[0].dispatchEvent(evt);
            """
            self.driver.execute_script(script, self.canvas)
        except Exception as e:
            self.update_debug_overlay(f"❌ JS click also failed: {e}")

    def start_game(self):
        """Initialize game and UI overlays"""
        print("Starting Baloot Automation...")
        try:
            self.driver.get("https://kammelna.com/baloot/")
            print("Waiting for Unity canvas...")

            self.canvas = WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.ID, "unity-canvas"))
            )

            self.canvas_rect = self.driver.execute_script("""
                var rect = arguments[0].getBoundingClientRect();
                return { x: rect.left, y: rect.top, width: rect.width, height: rect.height };
            """, self.canvas)

            print(f"Canvas position: {self.canvas_rect}")

            self.create_debug_overlay()
            self.create_control_panel()

            ocr_status = "OCR Available" if self.ocr_available else "OCR Not Available"
            template_count = len(self.templates) if self.templates else 0
            self.update_debug_overlay(f"System ready! {ocr_status} | Templates: {template_count}")
        except Exception as e:
            print(f"Init error: {e}")
            self.canvas_rect = {'x': 0, 'y': 0, 'width': 1920, 'height': 1080}
            self.create_debug_overlay()
            self.create_control_panel()
            self.update_debug_overlay(f"Warning: {e} - using defaults")

    def create_control_panel(self):
        """Create interactive START/STOP control panel"""
        script = f"""
        if (document.getElementById('baloot_control_panel')) return;
        const panel = document.createElement('div');
        panel.id = 'baloot_control_panel';
        panel.style.cssText = `
            position: fixed; top: 50%; right: 20px; transform: translateY(-50%);
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            border: 2px solid #00bfff; border-radius: 15px; padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5); z-index: 999999;
            font-family: Arial, sans-serif; min-width: 220px;
        `;
        const title = document.createElement('div');
        title.textContent = '🎮 BALOOT BOT';
        title.style.cssText = 'color:#00d4ff; text-align:center; margin-bottom:15px; font-weight:bold;';
        panel.appendChild(title);

        const status = document.createElement('div');
        status.id = 'automation_status';
        status.style.cssText = 'background:#e74c3c; color:white; padding:10px; border-radius:8px; margin-bottom:15px; text-align:center; font-weight:bold;';
        status.textContent = 'STOPPED';
        panel.appendChild(status);

        const startBtn = document.createElement('button');
        startBtn.textContent = '▶️ START';
        startBtn.style.cssText = 'width:100%; background:#27ae60; color:white; border:none; padding:12px; border-radius:8px; margin-bottom:10px; cursor:pointer; font-weight:bold;';
        startBtn.onclick = () => {{ window.automationControl = {{action:'START'}}; }};
        panel.appendChild(startBtn);

        const stopBtn = document.createElement('button');
        stopBtn.textContent = '⏹️ STOP';
        stopBtn.style.cssText = 'width:100%; background:#c0392b; color:white; border:none; padding:12px; border-radius:8px; margin-bottom:15px; cursor:pointer; font-weight:bold;';
        stopBtn.onclick = () => {{ window.automationControl = {{action:'STOP'}}; }};
        panel.appendChild(stopBtn);

        const info = document.createElement('div');
        info.innerHTML = '<small style="color:#bdc3c7;">Hybrid Detection:<br>• Template Matching<br>• OCR & Visual Fallback</small>';
        info.style.cssText = 'text-align:center; font-size:11px;';
        panel.appendChild(info);

        document.body.appendChild(panel);

        // Make draggable
        let isDragging = false, startX, startY;
        title.onmousedown = (e) => {{
            isDragging = true;
            startX = e.clientX - panel.offsetLeft;
            startY = e.clientY - panel.offsetTop;
        }};
        document.onmousemove = (e) => {{
            if (!isDragging) return;
            panel.style.left = (e.clientX - startX) + 'px';
            panel.style.top = (e.clientY - startY) + 'px';
            panel.style.right = 'auto';
            panel.style.transform = 'none';
        }};
        document.onmouseup = () => {{ isDragging = false; }};
        """
        self.driver.execute_script(script)

    def create_debug_overlay(self):
        """Create debug log overlay"""
        script = f"""
        if (document.getElementById('{self.debug_overlay_id}')) return;
        const overlay = document.createElement('div');
        overlay.id = '{self.debug_overlay_id}';
        overlay.style.cssText = `
            position: fixed; top: 10px; left: 10px; width: 500px; max-height: 400px;
            background: rgba(0,0,0,0.95); color: #00ff41; font-family: monospace;
            font-size: 11px; padding: 15px; border: 2px solid #00ff41; border-radius: 10px;
            z-index: 999998; overflow-y: auto; box-shadow: 0 8px 30px rgba(0,255,65,0.3);
        `;
        const header = document.createElement('div');
        header.textContent = '🔍 DEBUG SYSTEM';
        header.style.cssText = 'color:#ffff00; text-align:center; margin-bottom:10px; font-weight:bold;';
        overlay.appendChild(header);

        const status = document.createElement('div');
        status.id = '{self.debug_overlay_id}_status';
        status.style.cssText = 'color:white; background:rgba(255,255,255,0.1); padding:5px; margin-bottom:12px; text-align:center; border-radius:5px;';
        overlay.appendChild(status);

        const logs = document.createElement('div');
        logs.id = '{self.debug_overlay_id}_logs';
        logs.style.cssText = 'max-height:250px; overflow-y:auto; padding:8px; background:rgba(0,0,0,0.7); border:1px solid #333; border-radius:5px;';
        overlay.appendChild(logs);

        const controls = document.createElement('div');
        controls.style.cssText = 'margin-top:12px; text-align:center;';
        const clearBtn = document.createElement('button');
        clearBtn.textContent = 'Clear';
        clearBtn.onclick = () => {{ logs.innerHTML = ''; }};
        clearBtn.style.cssText = 'background:#333; color:#00ff41; border:1px solid #00ff41; padding:5px 10px; border-radius:4px; margin-right:10px;';
        controls.appendChild(clearBtn);

        const screenshotBtn = document.createElement('button');
        screenshotBtn.textContent = 'Screenshot';
        screenshotBtn.onclick = () => {{ window.automationControl = {{action:'SCREENSHOT'}}; }};
        screenshotBtn.style.cssText = 'background:#333; color:#ffff00; border:1px solid #ffff00; padding:5px 10px; border-radius:4px;';
        controls.appendChild(screenshotBtn);

        overlay.appendChild(controls);
        document.body.appendChild(overlay);

        // Make draggable
        let isDragging = false, startX, startY;
        header.onmousedown = (e) => {{
            isDragging = true;
            startX = e.clientX - overlay.offsetLeft;
            startY = e.clientY - overlay.offsetTop;
        }};
        document.onmousemove = (e) => {{
            if (!isDragging) return;
            overlay.style.left = (e.clientX - startX) + 'px';
            overlay.style.top = (e.clientY - startY) + 'px';
        }};
        document.onmouseup = () => {{ isDragging = false; }};
        """
        self.driver.execute_script(script)

    def update_debug_overlay(self, message, status=None):
        """Log messages to overlay"""
        timestamp = time.strftime("%H:%M:%S")
        safe_msg = message.replace("\\", "\\\\").replace("'", r"\'").replace('"', r'\"')
        safe_status = status or ""
        script = f"""
        const overlay = document.getElementById('{self.debug_overlay_id}');
        if (overlay) {{
            if ('{safe_status}') {{
                const el = document.getElementById('{self.debug_overlay_id}_status');
                if (el) el.textContent = 'STATUS: {safe_status}';
            }}
            const logs = document.getElementById('{self.debug_overlay_id}_logs');
            if (logs) {{
                const entry = document.createElement('div');
                entry.innerHTML = '<span style="color:#888;">[{timestamp}]</span> <span style="color:#00ff41;">{safe_msg}</span>';
                entry.style.cssText = 'margin:2px 0; padding:4px 8px; border-left:3px solid #00ff41; background:rgba(0,255,65,0.1); border-radius:3px;';
                logs.appendChild(entry);
                logs.scrollTop = logs.scrollHeight;
                while (logs.children.length > 30) logs.removeChild(logs.firstChild);
            }}
        }}
        """
        try:
            self.driver.execute_script(script)
            print(f"[{timestamp}] {message}")
        except:
            pass

    def update_automation_status(self, status):
        """Update control panel status color"""
        colors = {"RUNNING": "#27ae60", "STOPPED": "#e74c3c", "WAITING": "#f39c12", "CLICKING": "#9b59b6"}
        color = colors.get(status, "#34495e")
        script = f"""
        const el = document.getElementById('automation_status');
        if (el) {{ el.textContent = '{status}'; el.style.background = '{color}'; }}
        """
        try:
            self.driver.execute_script(script)
        except:
            pass

    def check_control_commands(self):
        """Check for START/STOP commands from UI"""
        try:
            cmd = self.driver.execute_script("""
                const c = window.automationControl;
                window.automationControl = null;
                return c;
            """)
            return cmd
        except:
            return None

    def show_click_indicator(self, x, y, color="red", duration=2000):
        """Show visual pulse where click happened"""
        script = f"""
        const ind = document.createElement('div');
        ind.style.cssText = `
            position:fixed; left:{x-25}px; top:{y-25}px; width:50px; height:50px;
            border:4px solid {color}; border-radius:50%; background:rgba(255,0,0,0.2);
            z-index:999997; pointer-events:none; animation:clickPulse 0.6s infinite;
        `;
        if (!document.querySelector('[data-click-animation]')) {{
            const style = document.createElement('style');
            style.setAttribute('data-click-animation', 'true');
            style.textContent = '@keyframes clickPulse{{0%{{transform:scale(1)}}50%{{transform:scale(1.2);opacity:0.7}}100%{{transform:scale(1)}}}}';
            document.head.appendChild(style);
        }}
        document.body.appendChild(ind);
        setTimeout(() => {{ if (ind.parentNode) ind.parentNode.removeChild(ind); }}, {duration});
        """
        try:
            self.driver.execute_script(script)
        except:
            pass

    def detect_with_template_matching(self, img):
        """Detect buttons using template matching (highest priority)"""
        if not self.templates:
            return {"found": False}

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        best_match = None

        for state, template in self.templates.items():
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            h, w = template_gray.shape

            scales = [1.0, 0.95, 1.05]
            for scale in scales:
                scaled_w, scaled_h = int(w * scale), int(h * scale)
                if scaled_h > img_gray.shape[0] or scaled_w > img_gray.shape[1]:
                    continue
                resized = cv2.resize(template_gray, (scaled_w, scaled_h))
                res = cv2.matchTemplate(img_gray, resized, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)

                if max_val > 0.75:
                    center_x = max_loc[0] + scaled_w // 2
                    center_y = max_loc[1] + scaled_h // 2
                    if best_match is None or max_val > best_match['confidence']:
                        best_match = {
                            "found": True,
                            "state": state,
                            "confidence": max_val,
                            "button_location": (center_x, center_y),
                            "reason": f"Template Match ({state})"
                        }

        return best_match if best_match else {"found": False}

    def hybrid_button_detection(self, screenshot_path):
        """
        Detect buttons using hybrid methods with strict priority order.
        Returns the highest-priority button found.
        """
        img = cv2.imread(screenshot_path)
        if img is None:
            return {"state": "ERROR", "confidence": 0, "reason": "Screenshot failed"}

        img_height, img_width = img.shape[:2]
        exclude_x = int(img_width * 0.75)
        working_img = img[:, :exclude_x]
        hsv_img = cv2.cvtColor(working_img, cv2.COLOR_BGR2HSV)

        priority_order = [
            "PLAY_BALOOT",
            "GREEN_PARTICIPATE",
            "RETURN_GREEN",
            "RETURN_GREY",
            "LEAVE_GAME"
        ]

        for state in priority_order:
            result = self.detect_single_template_match(working_img, state)
            if result["found"]:
                result["reason"] = f"Template Match ({state})"
                return result

        # If no templates matched, fall back to OCR + Visual (optional fallback)
        # But keep it disabled to fully prioritize templates only
        # Uncomment below if you want fallbacks after all templates fail

        # OCR Fallback (Optional - disable if you want template-only logic)
        # ocr_result = self.detect_with_ocr(working_img)
        # if ocr_result["found"]:
        #     return ocr_result

        # Visual Fallback (Optional)
        # visual_result = self.detect_with_visual(working_img, hsv_img)
        # if visual_result["found"]:
        #     return visual_result

        return {"state": "WAITING", "confidence": 0, "reason": "No buttons detected"}
    def detect_single_template_match(self, img, state):
        """
        Detect a single button by its state using multi-scale template matching.
        Returns result only if confidence > threshold.
        """
        if state not in self.templates:
            return {"found": False}

        template = self.templates[state]
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        h, w = template_gray.shape

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        best_confidence = 0
        best_location = None

        scales = [1.0, 0.95, 1.05, 0.9, 1.1]
        for scale in scales:
            scaled_w = int(w * scale)
            scaled_h = int(h * scale)
            if scaled_h > img_gray.shape[0] or scaled_w > img_gray.shape[1]:
                continue

            resized_template = cv2.resize(template_gray, (scaled_w, scaled_h))
            res = cv2.matchTemplate(img_gray, resized_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val > best_confidence:
                best_confidence = max_val
                center_x = max_loc[0] + scaled_w // 2
                center_y = max_loc[1] + scaled_h // 2
                best_location = (center_x, center_y)

        if best_confidence >= 0.75:
            return {
                "found": True,
                "state": state,
                "confidence": best_confidence,
                "button_location": best_location
            }

        return {"found": False}

    def detect_with_ocr(self, img):
        """OCR-based detection for Arabic text"""
        try:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(hsv, (35, 100, 100), (85, 255, 255))
            gray_mask = cv2.inRange(hsv, (0, 0, 60), (180, 30, 180))
            combined = cv2.bitwise_or(green_mask, gray_mask)
            kernel = np.ones((7,7), np.uint8)
            combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 3000 < area < 50000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    roi = img[y:y+h, x:x+w]
                    text = self.extract_text_from_region(roi)
                    match = self.fuzzy_match_text(text)
                    if match:
                        cx, cy = x + w//2, y + h//2
                        return {
                            "found": True,
                            "state": match["state"],
                            "confidence": match["confidence"],
                            "button_location": (cx, cy),
                            "ocr_text": text[:50]
                        }
            return {"found": False}
        except Exception as e:
            self.update_debug_overlay(f"OCR error: {e}")
            return {"found": False}

    def extract_text_from_region(self, region):
        """Preprocess and extract text using Tesseract"""
        try:
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            h, w = thresh.shape
            if h > 0 and w > 0:
                resized = cv2.resize(thresh, (w*3, h*3), interpolation=cv2.INTER_CUBIC)
                pil_img = Image.fromarray(resized)
                configs = ['--psm 8', '--psm 6']
                texts = []
                for cfg in configs:
                    try:
                        txt = pytesseract.image_to_string(pil_img, lang='ara+eng', config=cfg)
                        if txt.strip():
                            texts.append(txt.strip())
                    except:
                        continue
                return ' '.join(texts)
            return ""
        except:
            return ""

    def fuzzy_match_text(self, text):
        if not text:
            return None
        clean = re.sub(r'[^\u0600-\u06FF\sa-zA-Z]', ' ', text.lower())
        patterns = {
            "PLAY_BALOOT": ["العب بلوت", "لعب بلوت", "بلوت", "play"],
            "RETURN": ["عودة", "رجوع", "back"],
            "LEAVE_GAME": ["مغادرة", "خروج", "leave"]
        }
        for state, keys in patterns.items():
            for k in keys:
                if k in clean:
                    return {"state": state, "confidence": 0.9}
        return None

    def detect_with_visual(self, img, hsv):
        specs = [
            {"name": "play", "color": ([35,100,100], [85,255,255]), "min_area": 8000, "state": "PLAY_BALOOT"},
            {"name": "return", "color": ([35,80,80], [85,255,255]), "min_area": 2000, "state": "RETURN"},
            {"name": "leave", "color": ([0,0,60], [180,30,180]), "min_area": 2000, "state": "LEAVE_GAME"}
        ]
        candidates = []
        for spec in specs:
            lo, hi = spec["color"]
            mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > spec["min_area"]:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cx, cy = x + w//2, y + h//2
                    candidates.append({
                        "state": spec["state"],
                        "button_location": (cx, cy),
                        "confidence": min(0.8, area / 20000),
                    })
        if candidates:
            best = max(candidates, key=lambda x: x["confidence"])
            best["found"] = True
            return best
        return {"found": False}

    def automation_loop(self):
        self.update_debug_overlay("🤖 Automation loop started...")
        self.update_automation_status("RUNNING")
        consecutive_failures = 0
        last_state = None
        state_repetition = 0

        while self.automation_running:
            cmd = self.check_control_commands()
            if cmd and cmd.get('action') == 'STOP':
                self.automation_running = False
                break
            elif cmd and cmd.get('action') == 'SCREENSHOT':
                self.save_debug_screenshot("manual")
                continue

            self.update_automation_status("DETECTING")
            temp_screenshot = f"{self.debug_folder}/temp_analysis.png"
            self.driver.save_screenshot(temp_screenshot)

            result = self.hybrid_button_detection(temp_screenshot)
            current_state = result.get("state", "ERROR")
            confidence = result.get("confidence", 0)

            if current_state == last_state:
                state_repetition += 1
            else:
                state_repetition = 0
            last_state = current_state

            self.update_debug_overlay(f"🎯 Detected: {current_state} (conf: {confidence:.2f})")

            if current_state == "GREEN_PARTICIPATE":
                self.update_debug_overlay("🟢 Green Participate detected! Clicking now...")
                x, y = result["button_location"]
                self.click_button_at_position(x, y, result)
                consecutive_failures = 0
                time.sleep(3)

            elif current_state == "PLAY_BALOOT":
                self.update_debug_overlay("🎮 Play Baloot detected! Clicking...")
                x, y = result["button_location"]
                self.click_button_at_position(x, y, result)
                consecutive_failures = 0
                time.sleep(4)

            elif current_state == "RETURN_GREEN":
                self.update_debug_overlay("🟡 Green Return: waiting 40s before clicking...")
                self.update_automation_status("WAITING")
                for i in range(40):
                    if not self.automation_running:
                        break
                    time.sleep(1)
                    if i % 10 == 0:
                        self.update_debug_overlay(f"⏳ Waiting... {40-i}s remaining")
                if self.automation_running:
                    x, y = result["button_location"]
                    self.click_button_at_position(x, y, result)
                time.sleep(3)

            elif current_state == "RETURN_GREY":
                self.update_debug_overlay("⚪ Grey Return detected! Clicking immediately...")
                x, y = result["button_location"]
                self.click_button_at_position(x, y, result)
                consecutive_failures = 0
                time.sleep(3)

            elif current_state == "LEAVE_GAME":
                self.update_debug_overlay("🚪 Leave Game detected! Returning to menu...")
                x, y = result["button_location"]
                self.click_button_at_position(x, y, result)
                consecutive_failures = 0
                time.sleep(4)

            elif current_state == "WAITING":
                consecutive_failures += 1
                if state_repetition > 10:
                    self.update_debug_overlay("🔄 Possible stuck state. Taking a break...")
                    time.sleep(10)
                    state_repetition = 0
                if consecutive_failures >= 5:
                    self.update_debug_overlay("⚠️ Too many failures. Pausing 30s...")
                    self.save_debug_screenshot("failure_pause")
                    time.sleep(30)
                    consecutive_failures = 0
                else:
                    time.sleep(3)

            else:
                consecutive_failures += 1
                self.update_debug_overlay(f"❌ Unknown state: {result.get('reason', 'N/A')}")
                time.sleep(5)

            try:
                if os.path.exists(temp_screenshot):
                    os.remove(temp_screenshot)
            except:
                pass

            time.sleep(1)

        self.update_automation_status("STOPPED")
        self.update_debug_overlay("🛑 Automation stopped.")

    def run_with_controls(self):
        try:
            self.start_game()
            method = "Hybrid (Template + OCR)" if self.templates else "OCR + Visual"
            self.update_debug_overlay(f"Ready! Mode: {method}")
            self.save_debug_screenshot("startup")

            while True:
                cmd = self.check_control_commands()
                if cmd:
                    act = cmd.get('action')
                    if act == 'START' and not self.automation_running:
                        self.automation_running = True
                        self.automation_loop()
                    elif act == 'STOP':
                        self.automation_running = False
                    elif act == 'EMERGENCY_STOP':
                        self.automation_running = False
                        self.update_debug_overlay("🚨 Emergency Stop!")
                        break
                    elif act == 'SCREENSHOT':
                        self.save_debug_screenshot("manual_request")

                if not self.automation_running:
                    time.sleep(0.5)
        except KeyboardInterrupt:
            self.update_debug_overlay("👋 Stopped by user (Ctrl+C)")
        except Exception as e:
            self.update_debug_overlay(f"💥 Critical error: {e}")
            self.save_debug_screenshot("critical_error")
        finally:
            self.cleanup_debug_folder()
            self.update_debug_overlay("🔚 Session ended.")
            input("\nPress Enter to close browser...")
            self.driver.quit()


def main():
    print("="*60)
    print("🎮 ROBUST BALOOT AUTOMATION SYSTEM")
    print("="*60)
    print("🔹 Features:")
    print("  • Hybrid Detection (Templates + OCR + Visual)")
    print("  • Real Mouse Clicks via PyAutoGUI")
    print("  • Interactive START/STOP Panel")
    print("  • Debug Screenshots & Logs")
    print("  • Auto Recovery from Errors")
    print("\n📁 Debug screenshots will be saved in 'debug_screenshots/'")

    try:
        import cv2
        print("✅ OpenCV: OK")
    except ImportError:
        print("❌ OpenCV missing: pip install opencv-python")
        return

    try:
        import pytesseract
        print("✅ Tesseract: OK")
    except ImportError:
        print("❌ pytesseract missing: pip install pytesseract")
        return

    input("\n📌 Press Enter to start...")
    bot = RobustBalootAutomation()
    bot.run_with_controls()


if __name__ == "__main__":
    main()