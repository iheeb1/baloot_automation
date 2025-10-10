import cv2
import numpy as np
import argparse
import os

class GiftBoxPathTester:
    def __init__(self, screenshot_path, threshold=0.7):
        """
        Test gift box detection and path finding
        
        Args:
            screenshot_path: Path to full screenshot
            threshold: Template matching threshold (0.0 - 1.0)
        """
        self.screenshot_path = screenshot_path
        self.template_path = "gift_box_template.png"  # Fixed template name
        self.threshold = threshold
        
        # Load images
        self.screenshot = cv2.imread(screenshot_path)
        self.template = cv2.imread(self.template_path)
        
        if self.screenshot is None:
            raise ValueError(f"❌ Could not load screenshot: {screenshot_path}")
        if self.template is None:
            raise ValueError(f"❌ Could not load template: {template_path}")
        
        print(f"✅ Loaded screenshot: {self.screenshot.shape}")
        print(f"✅ Loaded template: {self.template.shape}")
        
        # Create output folder
        self.output_folder = "giftbox_test_output"
        os.makedirs(self.output_folder, exist_ok=True)
    
    def detect_giftbox(self):
        """
        Detect gift box location using multi-scale template matching
        Returns dict with location info or None
        """
        # Convert to grayscale
        gray_screenshot = cv2.cvtColor(self.screenshot, cv2.COLOR_BGR2GRAY)
        gray_template = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY)
        
        # Get template dimensions
        template_h, template_w = gray_template.shape
        
        print(f"\n🔍 Template Matching Configuration:")
        print(f"   Screenshot size: {gray_screenshot.shape}")
        print(f"   Template size: {gray_template.shape}")
        
        # Try multiple scales
        scales = [0.8, 0.9, 1.0, 1.1, 1.2]
        all_matches = []
        
        for scale in scales:
            # Resize template
            if scale != 1.0:
                new_w = int(template_w * scale)
                new_h = int(template_h * scale)
                resized_template = cv2.resize(gray_template, (new_w, new_h))
            else:
                resized_template = gray_template
                new_w, new_h = template_w, template_h
            
            # Skip if template is larger than screenshot
            if new_h > gray_screenshot.shape[0] or new_w > gray_screenshot.shape[1]:
                continue
            
            # Perform template matching
            result = cv2.matchTemplate(gray_screenshot, resized_template, cv2.TM_CCOEFF_NORMED)
            
            # Find all matches above a lower threshold to analyze
            threshold_map = result >= (self.threshold * 0.8)  # 80% of threshold
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
            print(f"❌ No matches found at any scale")
            print(f"💡 Try lowering threshold: --threshold 0.5")
            return None
        
        # Sort by confidence
        all_matches.sort(key=lambda x: x['confidence'], reverse=True)
        
        print(f"\n🎯 Found {len(all_matches)} potential matches:")
        
        # Show top 5 matches
        for i, match in enumerate(all_matches[:5]):
            loc = match['location']
            conf = match['confidence']
            scale = match['scale']
            print(f"   #{i+1}: Confidence={conf:.3f}, Scale={scale:.1f}x, Location={loc}")
        
        # Visualize all top matches
        self._visualize_all_matches(gray_screenshot, all_matches[:10])
        
        # Filter matches by location (gift box should be on the right side)
        screen_width = gray_screenshot.shape[1]
        
        # Filter: must be in right half of screen
        right_side_matches = [m for m in all_matches if m['location'][0] > screen_width * 0.6]
        
        if not right_side_matches:
            print(f"\n⚠️ No matches found on right side of screen")
            print(f"   Using best match regardless of location...")
            right_side_matches = all_matches
        
        # Take best match from right side
        best_match = right_side_matches[0]
        
        if best_match['confidence'] >= self.threshold:
            loc = best_match['location']
            w = best_match['width']
            h = best_match['height']
            
            top_left = loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            center = (top_left[0] + w // 2, top_left[1] + h // 2)
            
            result_info = {
                "found": True,
                "confidence": best_match['confidence'],
                "top_left": top_left,
                "bottom_right": bottom_right,
                "center": center,
                "width": w,
                "height": h,
                "scale": best_match['scale']
            }
            
            print(f"\n✅ Gift box found!")
            print(f"   Confidence: {best_match['confidence']:.3f}")
            print(f"   Scale: {best_match['scale']:.1f}x")
            print(f"   Top-left: {top_left}")
            print(f"   Bottom-right: {bottom_right}")
            
            return result_info
        else:
            print(f"\n❌ Best match below threshold: {best_match['confidence']:.3f} < {self.threshold}")
            print(f"💡 Tips:")
            print(f"   1. Lower threshold: --threshold {best_match['confidence'] - 0.05:.2f}")
            print(f"   2. Check template image quality and cropping")
            
            return None
    
    def _visualize_all_matches(self, screenshot, matches):
        """Visualize top matches for debugging"""
        debug_img = cv2.cvtColor(screenshot, cv2.COLOR_GRAY2BGR)
        
        colors = [
            (0, 255, 0),    # Green - best
            (0, 255, 255),  # Yellow
            (0, 165, 255),  # Orange
            (0, 0, 255),    # Red
            (255, 0, 255),  # Magenta
        ]
        
        for i, match in enumerate(matches[:10]):
            loc = match['location']
            w = match['width']
            h = match['height']
            conf = match['confidence']
            
            color = colors[min(i, len(colors)-1)]
            
            top_left = loc
            bottom_right = (top_left[0] + w, top_left[1] + h)
            
            cv2.rectangle(debug_img, top_left, bottom_right, color, 2)
            cv2.putText(debug_img, f"#{i+1}: {conf:.2f}", 
                       (top_left[0], top_left[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        output_path = os.path.join(self.output_folder, "0_all_matches.png")
        cv2.imwrite(output_path, debug_img)
        print(f"\n💾 Saved all matches visualization: {output_path}")
    
    def visualize_detection(self, giftbox_info):
        """Draw detection box on screenshot"""
        vis_img = self.screenshot.copy()
        
        if giftbox_info and giftbox_info["found"]:
            top_left = giftbox_info["top_left"]
            bottom_right = giftbox_info["bottom_right"]
            center = giftbox_info["center"]
            
            # Draw rectangle
            cv2.rectangle(vis_img, top_left, bottom_right, (0, 255, 0), 3)
            
            # Draw center point
            cv2.circle(vis_img, center, 5, (0, 0, 255), -1)
            
            # Add text
            text = f"Gift Box ({giftbox_info['confidence']:.2f})"
            cv2.putText(vis_img, text, (top_left[0], top_left[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Save
        output_path = os.path.join(self.output_folder, "1_detection.png")
        cv2.imwrite(output_path, vis_img)
        print(f"\n💾 Saved detection visualization: {output_path}")
        
        return vis_img
    
    def detect_path_in_giftbox(self, giftbox_info):
        """
        Detect the path inside the gift box area with adjusted boundaries
        Returns list of path points
        """
        if not giftbox_info or not giftbox_info["found"]:
            return None
        
        # Get original gift box boundaries
        orig_x1, orig_y1 = giftbox_info["top_left"]
        orig_x2, orig_y2 = giftbox_info["bottom_right"]
        
        # Adjust boundaries: move down 100px, up 50px to focus on path area
        x1 = orig_x1
        y1 = orig_y1 + 100  # Move down 100px
        x2 = orig_x2
        y2 = orig_y2 - 50   # Move up 50px from bottom
        
        # Make sure boundaries are valid
        y1 = max(0, y1)
        y2 = min(self.screenshot.shape[0], y2)
        
        if y2 <= y1:
            print("⚠️ Invalid adjusted boundaries")
            return None
        
        # Extract the adjusted region
        roi = self.screenshot[y1:y2, x1:x2].copy()
        
        print(f"\n🔍 Path Detection Area:")
        print(f"   Original box: ({orig_x1},{orig_y1}) to ({orig_x2},{orig_y2})")
        print(f"   Adjusted area: ({x1},{y1}) to ({x2},{y2})")
        print(f"   ROI size: {roi.shape}")
        
        # Save adjusted ROI
        roi_path = os.path.join(self.output_folder, "2a_adjusted_roi.png")
        cv2.imwrite(roi_path, roi)
        print(f"💾 Saved adjusted ROI: {roi_path}")
        
        # Convert to grayscale for simpler contour detection
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply binary threshold to detect dark path on light background
        # The path appears darker (brown/tan) than the background
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        
        # Clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Save threshold mask
        mask_path = os.path.join(self.output_folder, "2b_threshold_mask.png")
        cv2.imwrite(mask_path, thresh)
        print(f"💾 Saved threshold mask: {mask_path}")
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            print("⚠️ No path contours found with threshold method")
            print("💡 Trying HSV color detection as fallback...")
            return self._detect_path_hsv_fallback(roi, x1, y1)
        
        # Get largest contour (the path)
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        
        print(f"✅ Found path contour (area: {area:.0f} pixels)")
        
        # Extract and sort points from left to right
        points = [tuple(pt[0]) for pt in contour]
        points_sorted = sorted(points, key=lambda p: p[0])
        
        # Smooth the path by sampling every Nth point
        # Fewer samples = faster drag, more samples = smoother path
        sample_rate = max(1, len(points_sorted) // 20)  # ~20 points for fast drag
        sampled_points = points_sorted[::sample_rate]
        
        # Convert to global coordinates
        global_points = [(p[0] + x1, p[1] + y1) for p in sampled_points]
        
        print(f"🎯 Extracted {len(global_points)} path points (sampled for speed)")
        print(f"   Start: {global_points[0]}")
        print(f"   End: {global_points[-1]}")
        
        return {
            "local_points": sampled_points,
            "global_points": global_points,
            "roi": roi,
            "mask": thresh,
            "contour": contour,
            "adjusted_offset": (x1, y1)
        }
    
    def _detect_path_hsv_fallback(self, roi, offset_x, offset_y):
        """Fallback method using HSV color detection"""
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Brown/tan color range for path
        lower_brown = np.array([10, 20, 100])
        upper_brown = np.array([30, 150, 200])
        
        mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # Clean up
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        if not contours:
            print("❌ No path found with HSV fallback either")
            return None
        
        contour = max(contours, key=cv2.contourArea)
        points = [tuple(pt[0]) for pt in contour]
        points_sorted = sorted(points, key=lambda p: p[0])
        
        sample_rate = max(1, len(points_sorted) // 20)
        sampled_points = points_sorted[::sample_rate]
        global_points = [(p[0] + offset_x, p[1] + offset_y) for p in sampled_points]
        
        print(f"✅ HSV fallback found {len(global_points)} points")
        
        return {
            "local_points": sampled_points,
            "global_points": global_points,
            "roi": roi,
            "mask": mask,
            "contour": contour,
            "adjusted_offset": (offset_x, offset_y)
        }
    
    def visualize_path(self, giftbox_info, path_info):
        """Visualize the detected path"""
        if not path_info:
            return
        
        # Get adjusted offset
        offset_x, offset_y = path_info["adjusted_offset"]
        
        # Create visualization on ROI
        roi_vis = path_info["roi"].copy()
        
        # Draw the adjusted detection area boundary
        roi_height = roi_vis.shape[0]
        roi_width = roi_vis.shape[1]
        cv2.rectangle(roi_vis, (0, 0), (roi_width-1, roi_height-1), (255, 255, 0), 2)
        
        # Draw all contour points in blue
        cv2.drawContours(roi_vis, [path_info["contour"]], -1, (255, 0, 0), 2)
        
        # Draw sampled path points in red with connecting lines
        for i, pt in enumerate(path_info["local_points"]):
            cv2.circle(roi_vis, pt, 4, (0, 0, 255), -1)
            if i > 0:
                prev_pt = path_info["local_points"][i-1]
                cv2.line(roi_vis, prev_pt, pt, (0, 255, 0), 2)
        
        # Mark start and end
        if path_info["local_points"]:
            start = path_info["local_points"][0]
            end = path_info["local_points"][-1]
            cv2.circle(roi_vis, start, 8, (0, 255, 255), -1)  # Yellow start
            cv2.circle(roi_vis, end, 8, (255, 0, 255), -1)    # Magenta end
            
            cv2.putText(roi_vis, "START", (start[0]-30, start[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(roi_vis, "END", (end[0]-20, end[1]+25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        
        # Add text showing number of points
        cv2.putText(roi_vis, f"{len(path_info['local_points'])} points", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save ROI visualization
        roi_path = os.path.join(self.output_folder, "3_path_detection.png")
        cv2.imwrite(roi_path, roi_vis)
        print(f"💾 Saved path visualization: {roi_path}")
        
        # Create full screenshot with path overlay
        full_vis = self.screenshot.copy()
        
        # Draw original gift box border in green
        cv2.rectangle(full_vis, giftbox_info["top_left"], 
                     giftbox_info["bottom_right"], (0, 255, 0), 2)
        
        # Draw adjusted detection area in cyan
        adj_top_left = (offset_x, offset_y)
        adj_bottom_right = (offset_x + roi_vis.shape[1], offset_y + roi_vis.shape[0])
        cv2.rectangle(full_vis, adj_top_left, adj_bottom_right, (255, 255, 0), 2)
        cv2.putText(full_vis, "Detection Area", 
                   (adj_top_left[0], adj_top_left[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Draw path on full image with thicker lines
        for i, pt in enumerate(path_info["global_points"]):
            cv2.circle(full_vis, pt, 5, (0, 0, 255), -1)
            if i > 0:
                prev_pt = path_info["global_points"][i-1]
                cv2.line(full_vis, prev_pt, pt, (0, 255, 0), 3)
        
        # Mark start and end on full image
        if path_info["global_points"]:
            start_global = path_info["global_points"][0]
            end_global = path_info["global_points"][-1]
            cv2.circle(full_vis, start_global, 12, (0, 255, 255), -1)
            cv2.circle(full_vis, end_global, 12, (255, 0, 255), -1)
            
            # Add labels
            cv2.putText(full_vis, "START", 
                       (start_global[0]-35, start_global[1]-15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(full_vis, "END", 
                       (end_global[0]-25, end_global[1]+30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # Save full visualization
        full_path = os.path.join(self.output_folder, "4_full_visualization.png")
        cv2.imwrite(full_path, full_vis)
        print(f"💾 Saved full visualization: {full_path}")
        
        return full_vis
    
    def run_full_test(self):
        """Run complete test pipeline"""
        print("\n" + "="*60)
        print("🎮 GIFT BOX PATH DETECTION TEST")
        print("="*60)
        
        # Step 1: Detect gift box
        giftbox_info = self.detect_giftbox()
        
        if not giftbox_info:
            print("\n❌ Test failed: Gift box not detected")
            print("💡 Try lowering the threshold with --threshold parameter")
            return False
        
        # Step 2: Visualize detection
        self.visualize_detection(giftbox_info)
        
        # Step 3: Detect path
        path_info = self.detect_path_in_giftbox(giftbox_info)
        
        if not path_info:
            print("\n⚠️ Path detection failed")
            print("💡 Check the color mask image to adjust HSV ranges")
            return False
        
        # Step 4: Visualize path
        self.visualize_path(giftbox_info, path_info)
        
        # Print summary
        print("\n" + "="*60)
        print("✅ TEST COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\n📊 Results Summary:")
        print(f"   Gift box detected: YES ({giftbox_info['confidence']:.1%} confidence)")
        print(f"   Path points found: {len(path_info['global_points'])}")
        print(f"   Output folder: {self.output_folder}/")
        print(f"\n📁 Generated files:")
        print(f"   1. 1_detection.png - Gift box detection")
        print(f"   2. 2_color_mask.png - Color filtering mask")
        print(f"   3. 3_path_detection.png - Path in gift box")
        print(f"   4. 4_full_visualization.png - Complete overview")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Test gift box detection and path finding algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_giftbox.py screenshot.png
  python test_giftbox.py screenshot.png --threshold 0.6
  python test_giftbox.py image.png --threshold 0.8

Template:
  Uses fixed template: gift_box_template.png (must be in same folder)

Output:
  Creates 'giftbox_test_output' folder with visualization images
        """
    )
    
    parser.add_argument("screenshot", help="Path to screenshot image")
    parser.add_argument("--threshold", type=float, default=0.7,
                       help="Template matching threshold (0.0-1.0, default: 0.7)")
    
    args = parser.parse_args()
    
    # Validate screenshot exists
    if not os.path.exists(args.screenshot):
        print(f"❌ Error: Screenshot file not found: {args.screenshot}")
        return
    
    # Check if template exists
    template_path = "gift_box_template.png"
    if not os.path.exists(template_path):
        print(f"❌ Error: Template file not found: {template_path}")
        print(f"💡 Make sure '{template_path}' is in the same folder as this script")
        return
    
    # Validate threshold
    if not 0.0 <= args.threshold <= 1.0:
        print(f"❌ Error: Threshold must be between 0.0 and 1.0")
        return
    
    try:
        # Create tester and run
        tester = GiftBoxPathTester(args.screenshot, args.threshold)
        success = tester.run_full_test()
        
        if success:
            print("\n✨ Check the output folder to see the results!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()