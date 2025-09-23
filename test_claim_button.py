import cv2
import numpy as np
import os
from datetime import datetime
import argparse

def test_multiple_buttons(screenshot_path,
                         claim_template="claim_button_template.png",
                         mouwafeq_template="mouwafeq_template.png",
                         back_template="return_grey_template.png"):
    """
    Detect 'استلم', 'موافق', and 'عودة' buttons.
    """
    
    templates = [
        (claim_template, "استلم", (0, 0, 255), "Claim"),
        (mouwafeq_template, "موافق", (0, 255, 255), "Agree"),
        (back_template, "عودة", (255, 0, 0), "Back")
    ]
    
    if not os.path.exists(screenshot_path):
        print(f"❌ Error: Screenshot file '{screenshot_path}' not found!")
        return False
    
    screenshot = cv2.imread(screenshot_path)
    if screenshot is None:
        print(f"❌ Error: Could not load screenshot from '{screenshot_path}'")
        return False
    
    print(f"✅ Screenshot loaded successfully: {screenshot_path}")
    print(f"Screenshot size: {screenshot.shape}")
    
    screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    
    result_img = screenshot.copy()
    
    found_buttons = []
    
    for template_path, label_ar, color, label_en in templates:
        if not os.path.exists(template_path):
            print(f"⚠️  Template not found: {template_path} (skipping)")
            continue
        
        template = cv2.imread(template_path, cv2.IMREAD_COLOR)
        if template is None:
            print(f"❌ Could not load template: {template_path}")
            continue
        
        print(f"✅ Loaded template: {label_ar} ({template_path}) | Size: {template.shape}")
        
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        
        screenshot_gray_blur = cv2.GaussianBlur(screenshot_gray, (3, 3), 0)
        template_gray_blur = cv2.GaussianBlur(template_gray, (3, 3), 0)
        
        result = cv2.matchTemplate(screenshot_gray_blur, template_gray_blur, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        threshold = 0.7 
        if max_val >= threshold:
            h, w = template_gray.shape
            center_x = max_loc[0] + w // 2
            center_y = max_loc[1] + h // 2
            
            cv2.circle(result_img, (center_x, center_y), 30, color, 3)
            
            coord_text = f"{label_en}: ({center_x}, {center_y})"
            cv2.putText(result_img, coord_text, (center_x - 100, center_y - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            conf_text = f"Confidence: {max_val:.2f}"
            cv2.putText(result_img, conf_text, (center_x - 100, center_y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            found_buttons.append({
                'label': label_ar,
                'x': center_x,
                'y': center_y,
                'confidence': max_val
            })
            
            print(f"✅ Found '{label_ar}' button!")
            print(f"   📍 Position: X={center_x}, Y={center_y}")
            print(f"   🎯 Confidence: {max_val:.2f}")
    
    if not found_buttons:
        print("❌ No buttons detected.")
        print("💡 Tip: Check template images or adjust similarity threshold.")
        return False
    
    debug_folder = "claim_debug_screenshots"
    if not os.path.exists(debug_folder):
        os.makedirs(debug_folder)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{debug_folder}/multi_detection_{timestamp}.png"
    cv2.imwrite(filename, result_img)
    
    print(f"\n📸 Saved annotated screenshot to: {filename}")
    print("\n💡 This is where the bot would click:")
    for btn in found_buttons:
        print(f"   → {btn['label']} at ({btn['x']}, {btn['y']}) [Confidence: {btn['confidence']:.2f}]")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Test multiple button detection')
    parser.add_argument('screenshot_path', help='Path to the screenshot to test')
    parser.add_argument('--claim-template', '-c', default='claim_button_template.png',
                       help='Path to "استلم" template')
    parser.add_argument('--mouwafeq-template', '-m', default='mouwafeq_template.png',
                       help='Path to "موافق" template')
    parser.add_argument('--back-template', '-b', default='return_grey_template.png',
                       help='Path to "عودة" (Back) template')  # renamed!
    
    args = parser.parse_args()
    
    success = test_multiple_buttons(
        screenshot_path=args.screenshot_path,
        claim_template=args.claim_template,
        mouwafeq_template=args.mouwafeq_template,
        back_template=args.back_template
    )
    
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")


if __name__ == "__main__":
    main()