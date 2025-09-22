import cv2
import pytesseract
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import re

# Set Tesseract path if needed (uncomment and adjust)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class ButtonDetectionTester:
    def __init__(self):
        self.results = []
    
    def clean_arabic_text(self, text):
        """Clean Arabic text for better matching"""
        if not text:
            return ""
        
        # Remove English characters and numbers
        text = re.sub(r'[a-zA-Z0-9]', '', text)
        
        # Remove punctuation and special characters
        text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Replace similar looking Arabic characters that OCR might confuse
        replacements = {
            'ي': 'ى',  # Different forms of ya
            'ة': 'ه',  # Ta marbuta vs ha
            'أ': 'ا',  # Alif with hamza
            'إ': 'ا',  # Alif with hamza below
            'آ': 'ا',  # Alif with madda
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text.strip()
    
    def fuzzy_text_search(self, ocr_text, target_words):
        """Fuzzy search for Arabic text patterns"""
        cleaned_text = self.clean_arabic_text(ocr_text.lower())
        
        # Define fuzzy patterns for each target word
        fuzzy_patterns = {
            # Play button patterns
            "العب بلوت": ["العب بلوت", "لعب بلوت", "العب يلوت", "لعب يلوت", "عب بلوت", "العب بنوت", "لعب بنوت"],
            "العب": ["العب", "لعب", "عب", "اللعب"],
            "بلوت": ["بلوت", "يلوت", "بنوت", "bloot", "baloot"],
            "play": ["play", "العب", "لعب"],
            "baloot": ["baloot", "بلوت", "يلوت"],
            
            # Return button patterns  
            "عودة": ["عودة", "العودة", "عوده", "العوده", "return"],
            "العودة": ["العودة", "عودة", "العوده", "عوده"],
            "رجوع": ["رجوع", "رجع", "back"],
            "return": ["return", "عودة", "العودة"],
            "back": ["back", "رجوع", "عودة"],
            
            # Leave button patterns
            "مغادرة": ["مغادرة", "مغادره", "مغاره", "exit", "leave"],
            "خروج": ["خروج", "خرج", "exit"],
            "إنهاء": ["إنهاء", "انهاء", "finish"],
            "leave": ["leave", "مغادرة", "خروج"],
            "exit": ["exit", "خروج", "مغادرة"]
        }
        
        matches = []
        
        # Check each target word
        for target in target_words:
            if target.lower() in fuzzy_patterns:
                patterns = fuzzy_patterns[target.lower()]
                
                for pattern in patterns:
                    clean_pattern = self.clean_arabic_text(pattern.lower())
                    
                    if clean_pattern in cleaned_text:
                        matches.append({
                            'target': target,
                            'found_pattern': pattern,
                            'confidence': 1.0
                        })
                        break
                    
                    # Check partial matches
                    elif len(clean_pattern) >= 3:
                        if clean_pattern[:3] in cleaned_text or clean_pattern[-3:] in cleaned_text:
                            matches.append({
                                'target': target,
                                'found_pattern': pattern,
                                'confidence': 0.7
                            })
                            break
        
        return matches
    
    def contains_text_with_location(self, button_region, text_list, global_x, global_y):
        """Check if button region contains target text and return match details"""
        try:
            if button_region is None or button_region.size == 0:
                return None
            
            # Preprocess for OCR
            gray = cv2.cvtColor(button_region, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Resize for better OCR
            scale_factor = 3
            height, width = thresh.shape
            if height <= 0 or width <= 0:
                return None
                
            resized = cv2.resize(thresh, (width * scale_factor, height * scale_factor), 
                               interpolation=cv2.INTER_CUBIC)
            
            pil_image = Image.fromarray(resized)
            
            # Try multiple OCR approaches
            ocr_results = []
            
            try:
                text1 = pytesseract.image_to_string(pil_image, lang='ara+eng', config='--psm 8')
                ocr_results.append(text1)
            except:
                pass
            
            try:
                text2 = pytesseract.image_to_string(pil_image, lang='ara', config='--psm 8')
                ocr_results.append(text2)
            except:
                pass
            
            try:
                text3 = pytesseract.image_to_string(pil_image, lang='ara+eng', config='--psm 6')
                ocr_results.append(text3)
            except:
                pass
            
            combined_text = " ".join(ocr_results).strip()
            
            # Use fuzzy matching
            matches = self.fuzzy_text_search(combined_text, text_list)
            
            if matches:
                return {
                    'location': (global_x, global_y),
                    'ocr_text': combined_text[:100] + "..." if len(combined_text) > 100 else combined_text,
                    'matches': matches,
                    'button_size': button_region.shape[:2]
                }
            
            return None
            
        except Exception as e:
            print(f"OCR Error: {e}")
            return None
    
    def find_buttons_in_image(self, img_path):
        """Find all potential buttons in an image"""
        print(f"\n=== ANALYZING: {img_path} ===")
        
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not load {img_path}")
            return
        
        # Get image dimensions to exclude control panel area (right 25%)
        img_height, img_width = img.shape[:2]
        exclude_x = int(img_width * 0.75)
        working_img = img[:, :exclude_x]
        working_hsv = cv2.cvtColor(working_img, cv2.COLOR_BGR2HSV)
        
        print(f"Image size: {img_width}x{img_height}")
        print(f"Analyzing area: {exclude_x}x{img_height} (excluding right panel)")
        
        detected_buttons = []
        
        # Look for colored buttons (green, blue, gray)
        color_ranges = {
            'green': ([35, 50, 50], [85, 255, 255]),
            'blue': ([100, 50, 50], [130, 255, 255]),
            'gray': ([0, 0, 50], [180, 50, 200])
        }
        
        for color_name, (lower, upper) in color_ranges.items():
            lower = np.array(lower)
            upper = np.array(upper)
            mask = cv2.inRange(working_hsv, lower, upper)
            
            # Clean up mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if 1000 < area < 50000:  # Button size range
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / float(h)
                    
                    if 0.3 < aspect_ratio < 4.0:  # Button-like proportions
                        # Extract button region with padding
                        padding = 15
                        x1 = max(0, x - padding)
                        y1 = max(0, y - padding)
                        x2 = min(working_img.shape[1], x + w + padding)
                        y2 = min(working_img.shape[0], y + h + padding)
                        
                        button_region = working_img[y1:y2, x1:x2]
                        center_x = x + w // 2
                        center_y = y + h // 2
                        
                        # Test for all target texts
                        all_targets = ["العب بلوت", "العب", "بلوت", "عودة", "العودة", "رجوع", "مغادرة", "خروج", "إنهاء"]
                        
                        match_result = self.contains_text_with_location(button_region, all_targets, center_x, center_y)
                        
                        if match_result:
                            detected_buttons.append({
                                'color': color_name,
                                'area': area,
                                'center': (center_x, center_y),
                                'bounds': (x, y, w, h),
                                'match_result': match_result
                            })
                            
                            print(f"  🎯 {color_name.upper()} BUTTON DETECTED:")
                            print(f"     Location: ({center_x}, {center_y})")
                            print(f"     Size: {w}x{h} (area: {area})")
                            for match in match_result['matches']:
                                print(f"     Match: '{match['target']}' -> '{match['found_pattern']}' (conf: {match['confidence']:.1f})")
                            print(f"     OCR: '{match_result['ocr_text'][:50]}...'")
        
        # Create output image with annotations
        if detected_buttons:
            self.create_annotated_image(img_path, detected_buttons, exclude_x)
        else:
            print("  ❌ No target buttons detected")
        
        return detected_buttons
    
    def create_annotated_image(self, original_path, buttons, exclude_x):
        """Create an annotated image showing detected buttons"""
        # Load original image
        img = cv2.imread(original_path)
        
        # Create output filename
        base_name = os.path.splitext(os.path.basename(original_path))[0]
        output_path = f"{base_name}_detected_buttons.jpg"
        
        # Draw exclusion zone
        cv2.line(img, (exclude_x, 0), (exclude_x, img.shape[0]), (0, 0, 255), 2)
        cv2.putText(img, "EXCLUDED", (exclude_x + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Color mapping for different button types
        color_map = {
            'green': (0, 255, 0),  # Green
            'blue': (255, 0, 0),   # Blue
            'gray': (128, 128, 128) # Gray
        }
        
        # Draw detected buttons
        for i, button in enumerate(buttons):
            center_x, center_y = button['center']
            x, y, w, h = button['bounds']
            color = color_map.get(button['color'], (255, 255, 255))
            
            # Draw bounding rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            
            # Draw center point
            cv2.circle(img, (center_x, center_y), 10, color, -1)
            cv2.circle(img, (center_x, center_y), 15, (255, 255, 255), 2)
            
            # Add label with detected text
            matches = button['match_result']['matches']
            if matches:
                label = f"{i+1}: {matches[0]['target']}"
                cv2.putText(img, label, (center_x - 50, center_y - 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Save annotated image
        cv2.imwrite(output_path, img)
        print(f"  📸 Annotated image saved as: {output_path}")
        
        return output_path
    
    def test_screenshot(self, img_path):
        """Test a single screenshot"""
        if not os.path.exists(img_path):
            print(f"Error: File {img_path} not found")
            return
        
        buttons = self.find_buttons_in_image(img_path)
        
        if buttons:
            print(f"\n📊 SUMMARY FOR {os.path.basename(img_path)}:")
            for i, button in enumerate(buttons):
                matches = button['match_result']['matches']
                print(f"  Button {i+1}: {button['color']} button at {button['center']}")
                for match in matches:
                    action = "CLICK_IMMEDIATELY"
                    if match['target'] in ['عودة', 'العودة'] and button['color'] == 'green':
                        action = "WAIT_40s_THEN_CLICK"
                    print(f"    → {match['target']} → {action}")
        else:
            print(f"\n❌ No target buttons found in {os.path.basename(img_path)}")
    
    def test_multiple_screenshots(self):
        """Test multiple screenshots"""
        print("🔍 BUTTON DETECTION TEST")
        print("=" * 50)
        
        # Look for common screenshot names
        common_names = [
            "screenshot.png", "screenshot.jpg",
            "play_screen.png", "play_screen.jpg",
            "return_screen.png", "return_screen.jpg", 
            "leave_screen.png", "leave_screen.jpg",
            "game_screen.png", "game_screen.jpg"
        ]
        
        found_files = []
        for name in common_names:
            if os.path.exists(name):
                found_files.append(name)
        
        if not found_files:
            # Ask user for file path
            while True:
                file_path = input("Enter screenshot path (or 'quit' to exit): ").strip()
                if file_path.lower() == 'quit':
                    break
                if os.path.exists(file_path):
                    self.test_screenshot(file_path)
                else:
                    print(f"File not found: {file_path}")
        else:
            print(f"Found {len(found_files)} screenshots to test:")
            for file_path in found_files:
                print(f"  - {file_path}")
            
            for file_path in found_files:
                self.test_screenshot(file_path)

if __name__ == "__main__":
    # Test Tesseract first
    try:
        version = pytesseract.get_tesseract_version()
        print(f"✅ Tesseract version: {version}")
    except Exception as e:
        print(f"❌ Tesseract not found: {e}")
        print("Please install Tesseract OCR first")
        exit()
    
    tester = ButtonDetectionTester()
    tester.test_multiple_screenshots()
    
    print("\n🎯 TEST COMPLETE!")
    print("Check the generated '*_detected_buttons.jpg' files to see:")
    print("  • Red circles show where automation would click")
    print("  • Rectangle colors: Green/Blue/Gray indicate button color")
    print("  • Labels show detected text")
    print("  • Red line shows excluded control panel area")