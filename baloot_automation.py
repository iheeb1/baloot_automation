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

class RobustBalootAutomation:
    def __init__(self):
        # Chrome setup with stability options
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
        # Create debug folder for screenshots
        self.debug_folder = "debug_screenshots"
        if not os.path.exists(self.debug_folder):
            os.makedirs(self.debug_folder)
        # Detection settings
        self.ocr_available = self.test_ocr()
        # Load button templates for template matching
        self.templates = self.load_templates()

    def load_templates(self):
        """Load template images for button matching."""
        template_files = {
            "PLAY_BALOOT": "play_baloot_template.png",
            "RETURN_GREEN": "return_template.png",  # Original green return
            "RETURN_GREY": "return_grey_template.png",  # New grey return
            "LEAVE_GAME": "leave_game_template.png"
        }
        templates = {}
        for state, filename in template_files.items():
            if os.path.exists(filename):
                template = cv2.imread(filename, cv2.IMREAD_COLOR)
                if template is not None:
                    templates[state] = template
                    print(f"✅ Loaded template for {state}: {filename}")
                else:
                    print(f"❌ Failed to load template image: {filename}")
            else:
                print(f"❌ Template file not found: {filename}")
        if not templates:
            print("⚠️  No templates loaded. Falling back to OCR/Visual methods.")
        return templates

    def test_ocr(self):
        """Test if OCR is available and working"""
        try:
            pytesseract.get_tesseract_version()
            return True
        except:
            return False

    def get_timestamp(self):
        """Get formatted timestamp for files and logs"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def save_debug_screenshot(self, action, detection_result=None):
        """Save screenshot with detection annotations for debugging"""
        timestamp = self.get_timestamp()
        filename = f"{self.debug_folder}/{timestamp}_{action}.png"
        try:
            # Take screenshot
            self.driver.save_screenshot(filename)
            # If we have detection results, annotate the image
            if detection_result and detection_result.get("button_location"):
                self.annotate_screenshot(filename, detection_result)
            self.update_debug_overlay(f"Debug screenshot: {filename}")
            return filename
        except Exception as e:
            self.update_debug_overlay(f"Screenshot error: {e}")
            return None

    def annotate_screenshot(self, screenshot_path, detection_result):
        """Add annotations to screenshot showing what was detected"""
        try:
            img = cv2.imread(screenshot_path)
            if img is None:
                return
            # Get detection info
            x, y = detection_result["button_location"]
            state = detection_result.get("state", "UNKNOWN")
            confidence = detection_result.get("confidence", 0)
            # Draw click point
            cv2.circle(img, (x, y), 15, (0, 0, 255), -1)  # Red filled circle
            cv2.circle(img, (x, y), 25, (255, 255, 255), 3)  # White outline
            # Add state label
            label = f"{state} ({confidence:.2f})"
            cv2.putText(img, label, (x - 50, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            # Add timestamp
            timestamp = datetime.now().strftime("%H:%M:%S")
            cv2.putText(img, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            # Save annotated image
            cv2.imwrite(screenshot_path, img)
        except Exception as e:
            self.update_debug_overlay(f"Annotation error: {e}")

    def click_button_at_position(self, x, y, detection_result):
        """Click at position using direct JavaScript mouse event (RELIABLE for Canvas)"""
        self.update_debug_overlay(f"Clicking at screen coordinates ({x}, {y})")
        self.update_automation_status("CLICKING")
        # Show visual click indicator
        self.show_click_indicator(x, y, "red", 3000)
        # Take pre-click screenshot
        self.save_debug_screenshot("before_click", detection_result)
        try:
            # Use JavaScript to create and dispatch a mouse event at the ABSOLUTE SCREEN POSITION
            click_script = f"""
                var evt = new MouseEvent('click', {{
                    bubbles: true,
                    cancelable: true,
                    view: window,
                    clientX: {x},
                    clientY: {y},
                    screenX: {x},
                    screenY: {y}
                }});
                arguments[0].dispatchEvent(evt);
            """
            # Execute the script on the canvas element
            self.driver.execute_script(click_script, self.canvas)
            # Show success indicator
            self.show_click_indicator(x, y, "lime", 2000)
            self.update_debug_overlay("✅ JavaScript click dispatched successfully!")
            # Take post-click screenshot
            time.sleep(2)
            self.save_debug_screenshot("after_click")
        except Exception as e:
            self.update_debug_overlay(f"❌ JavaScript click error: {e}")
            self.show_click_indicator(x, y, "orange", 2000)

    def start_game(self):
        """Initialize the game and UI"""
        print("Starting Robust Baloot Automation...")
        try:
            self.driver.get("https://kammelna.com/baloot/")
            # Wait for canvas
            print("Waiting for Unity canvas...")
            self.canvas = WebDriverWait(self.driver, 30).until(
                EC.presence_of_element_located((By.ID, "unity-canvas"))
            )
            # Get canvas dimensions
            self.canvas_rect = self.driver.execute_script("""
                var canvas = arguments[0];
                var rect = canvas.getBoundingClientRect();
                return {
                    'x': rect.left,
                    'y': rect.top,
                    'width': rect.width,
                    'height': rect.height
                };
            """, self.canvas)
            print(f"Canvas: {self.canvas_rect}")
            # Initialize UI
            self.create_debug_overlay()
            self.create_control_panel()
            ocr_status = "OCR Available" if self.ocr_available else "OCR Not Available"
            # FIXED: Changed self_rooms to self.templates
            template_status = f"Templates Loaded: {len(self.templates)}" if self.templates else "No templates loaded"
            self.update_debug_overlay(f"System ready! {ocr_status} | {template_status}")
        except Exception as e:
            print(f"Initialization error: {e}")
            self.canvas_rect = {'x': 0, 'y': 0, 'width': 1920, 'height': 1080}
            self.create_debug_overlay()
            self.create_control_panel()
            self.update_debug_overlay(f"Warning: {e} - using defaults")

    def create_control_panel(self):
        """Create enhanced control panel"""
        control_script = f"""
        var existingPanel = document.getElementById('baloot_control_panel');
        if (existingPanel) existingPanel.remove();
        var panel = document.createElement('div');
        panel.id = 'baloot_control_panel';
        panel.style.cssText = `
            position: fixed; top: 50%; right: 20px; transform: translateY(-50%);
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            border: 2px solid #0f3460; border-radius: 15px; padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5); z-index: 999999;
            font-family: Arial, sans-serif; min-width: 220px;
        `;
        var title = document.createElement('div');
        title.style.cssText = `
            color: #00d4ff; font-weight: bold; font-size: 16px; text-align: center;
            margin-bottom: 15px; padding-bottom: 10px; border-bottom: 2px solid #00d4ff;
        `;
        title.textContent = 'ROBUST AUTOMATION';
        panel.appendChild(title);
        var status = document.createElement('div');
        status.id = 'automation_status';
        status.style.cssText = `
            background: #e74c3c; color: white; text-align: center; padding: 10px;
            border-radius: 8px; margin-bottom: 15px; font-weight: bold;
        `;
        status.textContent = 'STOPPED';
        panel.appendChild(status);
        var startBtn = document.createElement('button');
        startBtn.textContent = 'START AUTOMATION';
        startBtn.style.cssText = `
            width: 100%; background: linear-gradient(135deg, #27ae60, #2ecc71);
            color: white; border: none; padding: 12px; border-radius: 8px;
            cursor: pointer; font-weight: bold; margin-bottom: 10px;
        `;
        // FIXED: Properly escaped braces for f-string
        startBtn.onclick = function() {{ window.automationControl = {{action: 'START'}}; }};
        panel.appendChild(startBtn);
        var stopBtn = document.createElement('button');
        stopBtn.textContent = 'STOP AUTOMATION';
        stopBtn.style.cssText = `
            width: 100%; background: linear-gradient(135deg, #c0392b, #e74c3c);
            color: white; border: none; padding: 12px; border-radius: 8px;
            cursor: pointer; font-weight: bold; margin-bottom: 15px;
        `;
        // FIXED: Properly escaped braces for f-string
        stopBtn.onclick = function() {{ window.automationControl = {{action: 'STOP'}}; }};
        panel.appendChild(stopBtn);
        var info = document.createElement('div');
        info.style.cssText = `
            color: #bdc3c7; font-size: 11px; text-align: center;
            padding-top: 10px; border-top: 1px solid #34495e; line-height: 1.4;
        `;
        info.innerHTML = 'Hybrid Detection:<br/>• Text + Visual<br/>• Auto Screenshots<br/>• Smart Fallbacks';
        panel.appendChild(info);
        document.body.appendChild(panel);
        title.style.cursor = 'grab';
        title.addEventListener('mousedown', function(e) {{
            var isDragging = true, startX = e.clientX - panel.offsetLeft, startY = e.clientY - panel.offsetTop;
            function onMouseMove(e) {{
                if (!isDragging) return;
                panel.style.left = (e.clientX - startX) + 'px';
                panel.style.top = (e.clientY - startY) + 'px';
                panel.style.right = 'auto'; panel.style.transform = 'none';
            }}
            function onMouseUp() {{ isDragging = false; document.removeEventListener('mousemove', onMouseMove); }}
            document.addEventListener('mousemove', onMouseMove);
            document.addEventListener('mouseup', onMouseUp);
        }});
        """
        self.driver.execute_script(control_script)

    def create_debug_overlay(self):
        """Create enhanced debug overlay"""
        overlay_script = f"""
        var existingOverlay = document.getElementById('{self.debug_overlay_id}');
        if (existingOverlay) existingOverlay.remove();
        var overlay = document.createElement('div');
        overlay.id = '{self.debug_overlay_id}';
        overlay.style.cssText = `
            position: fixed; top: 10px; left: 10px; width: 500px; max-height: 400px;
            background: rgba(0, 0, 0, 0.95); color: #00ff41; font-family: 'Courier New', monospace;
            font-size: 11px; padding: 15px; border-radius: 10px; border: 2px solid #00ff41;
            z-index: 999998; overflow-y: auto; box-shadow: 0 8px 30px rgba(0, 255, 65, 0.3);
        `;
        var header = document.createElement('div');
        header.style.cssText = `
            color: #ffff00; font-weight: bold; font-size: 14px; margin-bottom: 10px;
            border-bottom: 1px solid #00ff41; padding-bottom: 8px; text-align: center;
        `;
        header.textContent = 'ROBUST DETECTION SYSTEM';
        overlay.appendChild(header);
        var status = document.createElement('div');
        status.id = '{self.debug_overlay_id}_status';
        status.style.cssText = `
            color: #ffffff; margin-bottom: 12px; font-weight: bold; text-align: center;
            padding: 5px; background: rgba(255, 255, 255, 0.1); border-radius: 5px;
        `;
        overlay.appendChild(status);
        var logContainer = document.createElement('div');
        logContainer.id = '{self.debug_overlay_id}_logs';
        logContainer.style.cssText = `
            max-height: 250px; overflow-y: auto; border: 1px solid #333;
            padding: 8px; background: rgba(0, 0, 0, 0.7); border-radius: 5px;
        `;
        overlay.appendChild(logContainer);
        var controls = document.createElement('div');
        controls.style.cssText = `margin-top: 12px; text-align: center;`;
        var clearBtn = document.createElement('button');
        clearBtn.textContent = 'Clear';
        clearBtn.style.cssText = `
            background: #333; color: #00ff41; border: 1px solid #00ff41;
            padding: 5px 10px; border-radius: 4px; cursor: pointer; margin-right: 10px;
        `;
        clearBtn.onclick = function() {{ logContainer.innerHTML = ''; }};
        controls.appendChild(clearBtn);
        var screenshotBtn = document.createElement('button');
        screenshotBtn.textContent = 'Screenshot';
        screenshotBtn.style.cssText = `
            background: #333; color: #ffff00; border: 1px solid #ffff00;
            padding: 5px 10px; border-radius: 4px; cursor: pointer;
        `;
        screenshotBtn.onclick = function() {{ window.automationControl = {{action: 'SCREENSHOT'}}; }};
        controls.appendChild(screenshotBtn);
        overlay.appendChild(controls);
        document.body.appendChild(overlay);
        header.style.cursor = 'grab';
        header.addEventListener('mousedown', function(e) {{
            var isDragging = true, startX = e.clientX - overlay.offsetLeft, startY = e.clientY - overlay.offsetTop;
            function onMouseMove(e) {{
                if (!isDragging) return;
                overlay.style.left = (e.clientX - startX) + 'px';
                overlay.style.top = (e.clientY - startY) + 'px';
            }}
            function onMouseUp() {{ isDragging = false; document.removeEventListener('mousemove', onMouseMove); }}
            document.addEventListener('mousemove', onMouseMove);
            document.addEventListener('mouseup', onMouseUp);
        }});
        """
        self.driver.execute_script(overlay_script)

    def update_debug_overlay(self, message, status=None):
        """Update debug overlay with timestamps"""
        timestamp = time.strftime("%H:%M:%S")
        safe_message = message.replace('\\', '\\\\').replace("'", "\\'").replace('"', '\\"')
        safe_status = status.replace('\\', '\\\\').replace("'", "\\'") if status else "None"
        update_script = f"""
        var overlay = document.getElementById('{self.debug_overlay_id}');
        if (overlay) {{
            if ('{safe_status}' !== 'None') {{
                var statusEl = document.getElementById('{self.debug_overlay_id}_status');
                if (statusEl) statusEl.textContent = 'STATUS: {safe_status}';
            }}
            var logContainer = document.getElementById('{self.debug_overlay_id}_logs');
            if (logContainer) {{
                var logEntry = document.createElement('div');
                logEntry.style.cssText = `
                    margin: 2px 0; padding: 4px 8px; border-left: 3px solid #00ff41;
                    background: rgba(0, 255, 65, 0.1); border-radius: 3px; font-size: 10px;
                `;
                logEntry.innerHTML = '<span style="color: #888;">[{timestamp}]</span> <span style="color: #00ff41;">{safe_message}</span>';
                logContainer.appendChild(logEntry);
                logContainer.scrollTop = logContainer.scrollHeight;
                while (logContainer.children.length > 30) {{
                    logContainer.removeChild(logContainer.firstChild);
                }}
            }}
        }}
        """
        try:
            self.driver.execute_script(update_script)
            print(f"[{timestamp}] {message}")
        except Exception as e:
            print(f"[{timestamp}] {message} (UI update error)")

    def update_automation_status(self, status):
        """Update automation status indicator"""
        status_colors = {
            'RUNNING': '#27ae60', 'STOPPED': '#e74c3c',
            'WAITING': '#f39c12', 'CLICKING': '#9b59b6',
            'ERROR': '#e67e22', 'DETECTING': '#3498db'
        }
        color = status_colors.get(status, '#34495e')
        update_script = f"""
        var statusEl = document.getElementById('automation_status');
        if (statusEl) {{
            statusEl.textContent = '{status}';
            statusEl.style.background = '{color}';
        }}
        """
        try:
            self.driver.execute_script(update_script)
        except:
            pass

    def check_control_commands(self):
        """Check for UI control commands"""
        try:
            return self.driver.execute_script("""
                if (window.automationControl) {
                    var cmd = window.automationControl;
                    window.automationControl = null;
                    return cmd;
                }
                return null;
            """)
        except:
            return None

    def show_click_indicator(self, x, y, color="red", duration=2000):
        """Show visual click indicator"""
        indicator_script = f"""
        var indicator = document.createElement('div');
        indicator.style.cssText = `
            position: fixed; left: {x-25}px; top: {y-25}px; width: 50px; height: 50px;
            border: 4px solid {color}; border-radius: 50%; background: rgba(255, 0, 0, 0.2);
            z-index: 999997; pointer-events: none; animation: clickPulse 0.6s infinite;
        `;
        if (!document.querySelector('style[data-click-animation="true"]')) {{
            var style = document.createElement('style');
            style.setAttribute('data-click-animation', 'true');
            style.textContent = `
                @keyframes clickPulse {{
                    0% {{ transform: scale(1); opacity: 1; }}
                    50% {{ transform: scale(1.2); opacity: 0.7; }}
                    100% {{ transform: scale(1); opacity: 1; }}
                }}
            `;
            document.head.appendChild(style);
        }}
        document.body.appendChild(indicator);
        setTimeout(() => {{ if (indicator.parentNode) indicator.parentNode.removeChild(indicator); }}, {duration});
        """
        try:
            self.driver.execute_script(indicator_script)
        except:
            pass

    def detect_with_template_matching(self, img):
        """Detect buttons using template matching for highest accuracy."""
        # FIXED: Changed self.health to self.templates
        if not self.templates:
            return {"found": False}

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # FIXED: Changed self.archetypes to self.templates
        for state, template in self.templates.items():
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            h, w = template_gray.shape

            # Multi-scale matching to handle minor size variations
            scales = [1.0, 0.95, 1.05, 0.9, 1.1]
            best_match = None

            for scale in scales:
                scaled_template = cv2.resize(template_gray, (int(w * scale), int(h * scale)))
                if scaled_template.shape[0] > img_gray.shape[0] or scaled_template.shape[1] > img_gray.shape[1]:
                    continue

                res = cv2.matchTemplate(img_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

                # Threshold for a good match
                if max_val > 0.75:
                    top_left = max_loc
                    bottom_right = (top_left[0] + scaled_template.shape[1], top_left[1] + scaled_template.shape[0])
                    center_x = top_left[0] + scaled_template.shape[1] // 2
                    center_y = top_left[1] + scaled_template.shape[0] // 2

                    # Check if this is the best match so far
                    if best_match is None or max_val > best_match['confidence']:
                        best_match = {
                            "found": True,
                            "state": state,
                            "confidence": max_val,
                            "button_location": (center_x, center_y),
                            "is_green": state == "RETURN_GREEN"  # Only RETURN_GREEN is green
                        }

            if best_match:
                best_match["reason"] = f"Template Match (Scale: {scale:.2f})"
                return best_match

        return {"found": False}

    def hybrid_button_detection(self, screenshot_path):
        """Robust hybrid detection combining Template, OCR and visual methods"""
        img = cv2.imread(screenshot_path)
        if img is None:
            return {"state": "ERROR", "confidence": 0, "reason": "Screenshot failed"}

        # Exclude control panel area
        img_height, img_width = img.shape[:2]
        exclude_x = int(img_width * 0.75)
        working_img = img[:, :exclude_x]
        working_hsv = cv2.cvtColor(working_img, cv2.COLOR_BGR2HSV)

        detection_methods = []

        # Method 1: Template Matching (Highest Priority)
        template_result = self.detect_with_template_matching(working_img)
        if template_result["found"]:
            detection_methods.append(("TEMPLATE", template_result))

        # Method 2: OCR Detection (if available)
        if self.ocr_available:
            ocr_result = self.detect_with_ocr(working_img)
            if ocr_result["found"]:
                detection_methods.append(("OCR", ocr_result))

        # Method 3: Visual Color Detection
        visual_result = self.detect_with_visual(working_img, working_hsv)
        if visual_result["found"]:
            detection_methods.append(("VISUAL", visual_result))

        # Return the best result from the highest priority method
        if detection_methods:
            method_name, result = detection_methods[0]
            if "reason" not in result:
                result["reason"] = f"{method_name} detection"
            return result

        return {"state": "WAITING", "confidence": 0, "reason": "No buttons detected"}

    def detect_with_ocr(self, img):
        """OCR-based detection with fuzzy matching"""
        try:
            # Look for button regions first
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            # Detect potential button areas
            green_mask = cv2.inRange(hsv, np.array([35, 100, 100]), np.array([85, 255, 255]))
            gray_mask = cv2.inRange(hsv, np.array([0, 0, 60]), np.array([180, 30, 180]))
            blue_mask = cv2.inRange(hsv, np.array([100, 80, 80]), np.array([130, 255, 255]))
            combined_mask = cv2.bitwise_or(cv2.bitwise_or(green_mask, gray_mask), blue_mask)
            kernel = np.ones((7, 7), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if 3000 < area < 50000:
                    x, y, w, h = cv2.boundingRect(contour)
                    # Extract button region
                    padding = 20
                    x1, y1 = max(0, x - padding), max(0, y - padding)
                    x2, y2 = min(img.shape[1], x + w + padding), min(img.shape[0], y + h + padding)
                    button_region = img[y1:y2, x1:x2]
                    # OCR on button region
                    ocr_text = self.extract_text_from_region(button_region)
                    match_result = self.fuzzy_match_text(ocr_text)
                    if match_result:
                        # Determine if it's green
                        center_x, center_y = x + w//2, y + h//2
                        is_green = green_mask[center_y, center_x] > 0
                        return {
                            "found": True,
                            "state": match_result["state"],
                            "confidence": match_result["confidence"],
                            "location": (center_x, center_y),
                            "is_green": is_green,
                            "ocr_text": ocr_text[:50]
                        }
            return {"found": False}
        except Exception as e:
            self.update_debug_overlay(f"OCR detection error: {e}")
            return {"found": False}

    def extract_text_from_region(self, region):
        """Extract text from image region using OCR"""
        try:
            # Preprocess
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # Resize for better OCR
            h, w = thresh.shape
            if h > 0 and w > 0:
                scale = 3
                resized = cv2.resize(thresh, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
                pil_image = Image.fromarray(resized)
                # Multiple OCR attempts
                texts = []
                for config in ['--psm 8', '--psm 6', '--psm 7']:
                    try:
                        text = pytesseract.image_to_string(pil_image, lang='ara+eng', config=config)
                        if text.strip():
                            texts.append(text.strip())
                    except:
                        continue
                return ' '.join(texts)
            return ""
        except Exception as e:
            return ""

    def fuzzy_match_text(self, text):
        """Fuzzy match extracted text to target patterns"""
        if not text:
            return None
        # Clean text
        clean_text = re.sub(r'[^\u0600-\u06FF\sa-zA-Z]', ' ', text.lower())
        # Pattern definitions
        patterns = {
            "PLAY_BALOOT": {
                "patterns": ["العب بلوت", "لعب بلوت", "العب يلوت", "بلوت", "العب", "play", "baloot"],
                "confidence": 0.9
            },
            "RETURN": {
                "patterns": ["عودة", "العودة", "عوده", "رجوع", "return", "back"],
                "confidence": 0.85
            },
            "LEAVE_GAME": {
                "patterns": ["مغادرة", "مغادره", "خروج", "leave", "exit"],
                "confidence": 0.85
            }
        }
        for state, pattern_data in patterns.items():
            for pattern in pattern_data["patterns"]:
                if pattern in clean_text:
                    return {
                        "state": state,
                        "confidence": pattern_data["confidence"],
                        "matched_pattern": pattern
                    }
                # Partial matching for Arabic
                if len(pattern) >= 3 and any(part in clean_text for part in [pattern[:3], pattern[-3:]]):
                    return {
                        "state": state,
                        "confidence": pattern_data["confidence"] * 0.7,
                        "matched_pattern": f"{pattern} (partial)"
                    }
        return None

    def detect_with_visual(self, img, hsv):
        """Visual color-based detection as fallback"""
        try:
            # Define button characteristics
            button_specs = [
                {
                    "name": "play",
                    "color_range": ([35, 100, 100], [85, 255, 255]),
                    "min_area": 8000,
                    "min_circularity": 0.5,
                    "state": "PLAY_BALOOT"
                },
                {
                    "name": "return_green",
                    "color_range": ([35, 80, 80], [85, 255, 255]),
                    "min_area": 2000,
                    "min_circularity": 0.3,
                    "state": "RETURN",
                    "is_green": True
                },
                {
                    "name": "leave",
                    "color_range": ([0, 0, 60], [180, 30, 180]),
                    "min_area": 2000,
                    "min_circularity": 0.3,
                    "state": "LEAVE_GAME"
                }
            ]
            candidates = []
            for spec in button_specs:
                lower, upper = spec["color_range"]
                mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                kernel = np.ones((5, 5), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > spec["min_area"]:
                        # Check circularity
                        perimeter = cv2.arcLength(contour, True)
                        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                        if circularity > spec["min_circularity"]:
                            x, y, w, h = cv2.boundingRect(contour)
                            candidates.append({
                                "state": spec["state"],
                                "location": (x + w//2, y + h//2),
                                "area": area,
                                "circularity": circularity,
                                "confidence": min(0.8, circularity + area/20000),
                                "is_green": spec.get("is_green", False)
                            })
            if candidates:
                # Return best candidate (highest confidence)
                best = max(candidates, key=lambda x: x["confidence"])
                return {
                    "found": True,
                    "state": best["state"],
                    "confidence": best["confidence"],
                    "location": best["location"],
                    "is_green": best["is_green"]
                }
            return {"found": False}
        except Exception as e:
            self.update_debug_overlay(f"Visual detection error: {e}")
            return {"found": False}

    def automation_loop(self):
        """Main automation loop with robust error handling"""
        self.update_debug_overlay("Starting robust automation loop...")
        self.update_automation_status("RUNNING")
        consecutive_failures = 0
        max_failures = 5
        last_state = None
        state_repetition_count = 0

        while self.automation_running:
            try:
                # Check for control commands
                command = self.check_control_commands()
                if command:
                    if command.get('action') in ['STOP', 'EMERGENCY_STOP']:
                        self.update_debug_overlay(f"Stop command: {command.get('action')}")
                        break
                    elif command.get('action') == 'SCREENSHOT':
                        self.save_debug_screenshot("manual_screenshot")
                        continue

                # Update status
                self.update_automation_status("DETECTING")

                # Take screenshot for analysis
                screenshot_path = f"{self.debug_folder}/temp_analysis.png"
                self.driver.save_screenshot(screenshot_path)

                # Hybrid detection
                detection_result = self.hybrid_button_detection(screenshot_path)
                current_state = detection_result["state"]
                confidence = detection_result["confidence"]
                reason = detection_result["reason"]

                # State repetition check
                if current_state == last_state:
                    state_repetition_count += 1
                else:
                    state_repetition_count = 0
                last_state = current_state

                # Log detection result
                self.update_debug_overlay(
                    f"Detected: {current_state} (conf: {confidence:.2f}) - {reason}"
                )

                # Handle detected states
                if current_state == "PLAY_BALOOT":
                    self.update_debug_overlay("Play Baloot button detected - clicking immediately")
                    consecutive_failures = 0
                    x, y = detection_result["button_location"]
                    self.click_button_at_position(x, y, detection_result)
                    time.sleep(4)  # Wait for game to start

                elif current_state == "RETURN_GREEN":
                    # Green return button - wait 40 seconds
                    self.update_debug_overlay("Green return button - waiting 40 seconds...")
                    self.update_automation_status("WAITING")
                    # Countdown with interruption check
                    for remaining in range(40, 0, -5):
                        if not self.automation_running:
                            break
                        self.update_debug_overlay(f"Waiting {remaining}s before clicking return...")
                        time.sleep(5)
                        # Check for stop commands
                        cmd = self.check_control_commands()
                        if cmd and cmd.get('action') in ['STOP', 'EMERGENCY_STOP']:
                            self.automation_running = False
                            break
                    if self.automation_running:
                        self.update_automation_status("CLICKING")
                        self.update_debug_overlay("40 seconds elapsed - clicking return button")
                        x, y = detection_result["button_location"]
                        self.click_button_at_position(x, y, detection_result)
                    time.sleep(3)

                elif current_state == "RETURN_GREY":
                    # Grey return button - click immediately
                    self.update_debug_overlay("Grey return button detected - clicking immediately")
                    consecutive_failures = 0
                    x, y = detection_result["button_location"]
                    self.click_button_at_position(x, y, detection_result)
                    time.sleep(3)

                elif current_state == "LEAVE_GAME":
                    self.update_debug_overlay("Leave button detected - returning to main menu")
                    consecutive_failures = 0
                    x, y = detection_result["button_location"]
                    self.click_button_at_position(x, y, detection_result)
                    time.sleep(4)  # Wait for transition to main menu

                elif current_state == "WAITING":
                    consecutive_failures += 1
                    self.update_debug_overlay(f"No target buttons found ({consecutive_failures}/{max_failures})")
                    # Check for stuck state
                    if state_repetition_count > 10:
                        self.update_debug_overlay("Possible stuck state - taking longer break")
                        self.save_debug_screenshot("stuck_state")
                        time.sleep(10)
                        state_repetition_count = 0
                    # Handle too many consecutive failures
                    if consecutive_failures >= max_failures:
                        self.update_debug_overlay(f"Too many detection failures - pausing 30s")
                        self.update_automation_status("ERROR")
                        self.save_debug_screenshot("detection_failure")
                        time.sleep(30)
                        consecutive_failures = 0
                    else:
                        time.sleep(3)

                else:  # ERROR or unknown state
                    consecutive_failures += 1
                    self.update_debug_overlay(f"Error state: {reason}")
                    self.save_debug_screenshot("error_state")
                    time.sleep(5)

                # Clean up temp files
                try:
                    if os.path.exists(screenshot_path):
                        os.remove(screenshot_path)
                except:
                    pass

                # Update status back to running
                if self.automation_running and current_state != "WAITING":
                    self.update_automation_status("RUNNING")

                # Brief pause between iterations
                time.sleep(1)

            except Exception as e:
                consecutive_failures += 1
                self.update_debug_overlay(f"Loop error: {e}")
                self.update_automation_status("ERROR")
                self.save_debug_screenshot("loop_error")
                time.sleep(5)
                if consecutive_failures >= max_failures:
                    self.update_debug_overlay("Too many errors - stopping automation")
                    break

        self.update_debug_overlay("Automation loop ended")
        self.update_automation_status("STOPPED")
        self.automation_running = False

    def run_with_controls(self):
        """Main execution method"""
        try:
            self.start_game()
            # FIXED: Changed self.responseText to self.templates
            detection_method = "Hybrid (Template + OCR + Visual)" if self.templates else "Hybrid (OCR + Visual)"
            self.update_debug_overlay(f"Automation ready! Detection: {detection_method}")
            self.save_debug_screenshot("initial_state")

            # Main control loop
            while True:
                command = self.check_control_commands()
                if command:
                    action = command.get('action')
                    if action == 'START' and not self.automation_running:
                        self.automation_running = True
                        self.automation_loop()
                    elif action == 'STOP':
                        self.automation_running = False
                    elif action == 'EMERGENCY_STOP':
                        self.automation_running = False
                        self.update_debug_overlay("EMERGENCY STOP activated!")
                        break
                    elif action == 'SCREENSHOT':
                        self.save_debug_screenshot("manual_request")

                if not self.automation_running:
                    time.sleep(0.5)

        except KeyboardInterrupt:
            self.update_debug_overlay("Stopped by user (Ctrl+C)")
        except Exception as e:
            self.update_debug_overlay(f"Critical error: {e}")
            self.save_debug_screenshot("critical_error")
        finally:
            self.update_debug_overlay("Session ended. Debug screenshots saved in 'debug_screenshots' folder")
            self.update_automation_status("STOPPED")
            # Show debug folder info
            try:
                debug_files = len([f for f in os.listdir(self.debug_folder) if f.endswith('.png')])
                print(f"\nDebug Info:")
                print(f"- {debug_files} screenshots saved in '{self.debug_folder}' folder")
                print(f"- Check annotated images to see what was detected")
                print(f"- Each click has before/after screenshots")
            except:
                pass
            input("\nPress Enter to close browser...")
            self.driver.quit()

def main():
    """Main function with comprehensive setup"""
    print("ROBUST BALOOT AUTOMATION SYSTEM")
    print("=" * 50)
    print("Features:")
    print("• Hybrid Detection: Template + OCR + Visual fallbacks")
    print("• Debug Screenshots: Before/after every action")
    print("• Smart Error Handling: Auto-recovery from failures")
    print("• Visual Feedback: Click indicators and status updates")
    print("• State Management: Prevents infinite loops")
    print("• Manual Control: Start/stop anytime")

    # Check system requirements
    print("\nSystem Check:")
    try:
        import cv2
        print("✅ OpenCV available")
    except ImportError:
        print("❌ OpenCV missing: pip install opencv-python")
        return
    try:
        import pytesseract
        try:
            version = pytesseract.get_tesseract_version()
            print(f"✅ Tesseract OCR v{version}")
            ocr_available = True
        except:
            print("⚠️  Tesseract OCR not found - will use visual-only detection")
            ocr_available = False
    except ImportError:
        print("❌ pytesseract missing: pip install pytesseract")
        return

    print(f"\nCRITICAL SETUP STEP:")
    print("1. Run this script once to generate screenshots.")
    print("2. Go to the 'debug_screenshots' folder.")
    print("3. Crop and save button templates as:")
    print("   - play_baloot_template.png")
    print("   - return_template.png")
    print("   - return_grey_template.png")
    print("   - leave_game_template.png")
    print("4. Place these files in the same folder as this script.")

    print(f"\nDetection Method: Hybrid (Template Matching + OCR + Visual)")
    print("Debug screenshots will be saved to 'debug_screenshots' folder")
    print("\nAutomation Behavior:")
    print("1. Main Menu: Detects 'العب بلوت' → clicks immediately")
    print("2. Return Popup:")
    print("   - Green button → waits 40 seconds then clicks")
    print("   - Grey button → clicks immediately")
    print("3. Game End: Detects 'مغادرة' → clicks to return to menu")
    print("4. Repeats cycle indefinitely with error recovery")

    input("\nPress Enter to start the robust automation...")

    try:
        automation = RobustBalootAutomation()
        automation.run_with_controls()
    except Exception as e:
        print(f"Startup error: {e}")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()