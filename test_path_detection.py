import cv2
import numpy as np
import os
from datetime import datetime
import argparse

def test_path_detection_with_drag_visualization(screenshot_path):
    """
    Detect the path inside a fixed zone of interest and show the drag path (following contour points).
    """

    if not os.path.exists(screenshot_path):
        print(f"❌ Error: Screenshot file '{screenshot_path}' not found!")
        return False

    screenshot = cv2.imread(screenshot_path)
    if screenshot is None:
        print(f"❌ Error: Could not load screenshot from '{screenshot_path}'")
        return False

    print(f"✅ Screenshot loaded successfully: {screenshot_path}")
    print(f"Screenshot size: {screenshot.shape}")

    # Create debug folder
    debug_folder = "path_debug_screenshots"
    os.makedirs(debug_folder, exist_ok=True)

    # === ZONE OF INTEREST ===
    x1, y1 = 1205, 375
    x2, y2 = 1480, 560
    zone_of_interest = screenshot[y1:y2, x1:x2]

    # Convert to grayscale
    gray = cv2.cvtColor(zone_of_interest, cv2.COLOR_BGR2GRAY)

    # Threshold to isolate path
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours:
        print("❌ No path detected in zone")
        return False

    # Select largest contour (the path)
    contour = max(contours, key=cv2.contourArea)

    # Extract points and sort them by x (left -> right)
    points = sorted([tuple(pt[0]) for pt in contour], key=lambda p: p[0])

    # Create a copy of the screenshot for drawing
    result_img = screenshot.copy()

    # Shift points to global coordinates (sampled to reduce density)
    global_points = [(p[0] + x1, p[1] + y1) for p in points[::20]]

    # === DRAW DRAG PATH ===
    for i in range(len(global_points) - 1):
        cv2.line(result_img, global_points[i], global_points[i+1], (0, 255, 255), 2)  # yellow path

    # Start and end markers
    start_pt, end_pt = global_points[0], global_points[-1]
    cv2.circle(result_img, start_pt, 6, (0, 0, 255), -1)  # red = start
    cv2.circle(result_img, end_pt, 6, (255, 0, 0), -1)    # blue = end

    # Label
    cv2.putText(result_img, "Mouse drag path", (start_pt[0], start_pt[1] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Save result
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{debug_folder}/drag_path_{timestamp}.png"
    cv2.imwrite(filename, result_img)

    print(f"✅ Path detected! Saved annotated drag visualization to: {filename}")
    print(f"   Start: {start_pt}, End: {end_pt}")
    print("🎯 The bot would drag following the sinusoidal path")

    return True


def main():
    parser = argparse.ArgumentParser(description='Path detection with drag visualization')
    parser.add_argument('screenshot_path', help='Path to the screenshot to test')
    args = parser.parse_args()

    success = test_path_detection_with_drag_visualization(args.screenshot_path)
    if success:
        print("\n✅ Completed successfully! Check 'path_debug_screenshots' folder for results")
    else:
        print("\n❌ Path detection failed!")


if __name__ == "__main__":
    main()
