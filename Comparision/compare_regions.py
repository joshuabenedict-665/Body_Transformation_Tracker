import cv2
import numpy as np
import os
import glob

# ---------- STEP 1: Compute Absolute Difference ----------
def compute_absdiff(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray1, gray2)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    return thresh

# ---------- STEP 2: Get Contours of Changed Areas ----------
def get_contours(diff_img):
    contours, _ = cv2.findContours(diff_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# ---------- STEP 3: Draw Green Highlights ----------
def draw_highlighted_differences(original_img, contours):
    highlighted = original_img.copy()
    cv2.drawContours(highlighted, contours, -1, (0, 255, 0), 2)  # green contour outlines
    return highlighted

# ---------- STEP 4: Full Pipeline (Batch Mode) ----------
def main():
    os.makedirs("differences", exist_ok=True)
    before_images = glob.glob("*_1.*")  # find files like chest_1.jpg, full_body_1.png

    if not before_images:
        print("❌ No before images found. Please name them as *_1.ext and *_2.ext")
        return

    for before_path in before_images:
        after_path = before_path.replace("_1.", "_2.")
        if not os.path.exists(after_path):
            print(f"⚠️ Skipping {before_path} — no matching after image found.")
            continue

        img1 = cv2.imread(before_path)
        img2 = cv2.imread(after_path)

        if img1 is None or img2 is None:
            print(f"❌ Error reading: {before_path}, {after_path}")
            continue

        if img1.shape != img2.shape:
            print("ℹ️ Resizing image2 to match image1...")
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        diff_thresh = compute_absdiff(img1, img2)
        contours = get_contours(diff_thresh)
        result_img = draw_highlighted_differences(img2, contours)

        # Save result
        output_path = os.path.join("differences", os.path.basename(before_path).replace("_1.", "_diff."))
        cv2.imwrite(output_path, result_img)
        print(f"✅ Saved difference: {output_path}")

    print("🎯 All comparisons complete. See 'differences' folder.")

# ---------- STEP 5: Run ----------
if __name__ == "__main__":
    main()
