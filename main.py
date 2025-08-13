# main.py
import os
from CapturePhoto import PhotoCapture
from region_extraction import extract_regions
from Comparison import compare_regions

def main():
    print("=== STEP 1: Capture Images (Member 1) ===")

    # Capture first image
    if not os.path.exists("image1.jpg"):
        print("[INFO] Capturing first image...")
        PhotoCapture.main()
    else:
        print("[INFO] image1.jpg already exists, skipping first capture.")

    # Capture second image
    if not os.path.exists("image2.jpg"):
        print("[INFO] Capturing second image...")
        PhotoCapture.main()
    else:
        print("[INFO] image2.jpg already exists, skipping second capture.")

    print("\n=== STEP 2: Extract & Save Regions (Member 2) ===")
    extract_regions.main()

    print("\n=== STEP 3: Compare Images (Member 3) ===")
    compare_regions.main()

if __name__ == "__main__":
    main()
