import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

def get_landmarks(image):
    """Extract pose landmarks from an image"""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    if results.pose_landmarks:
        return results.pose_landmarks.landmark
    else:
        raise ValueError("Pose landmarks not detected!")

def align_and_scale_image(image, lm, target_shoulder_hip_dist=None):
    """
    Align image so shoulders are horizontal and scale so shoulder–hip distance matches target
    """
    h, w = image.shape[:2]
    left_shoulder = np.array([lm[11].x * w, lm[11].y * h])
    right_shoulder = np.array([lm[12].x * w, lm[12].y * h])
    left_hip = np.array([lm[23].x * w, lm[23].y * h])
    right_hip = np.array([lm[24].x * w, lm[24].y * h])

    mid_shoulder = (left_shoulder + right_shoulder) / 2
    mid_hip = (left_hip + right_hip) / 2

    angle = np.degrees(np.arctan2(
        right_shoulder[1] - left_shoulder[1],
        right_shoulder[0] - left_shoulder[0]
    ))

    if angle > 45:
        angle -= 180
    elif angle < -45:
        angle += 180

    M_rot = cv2.getRotationMatrix2D(tuple(mid_shoulder), angle, 1.0)
    rotated = cv2.warpAffine(image, M_rot, (w, h))

    rotated_lm = get_landmarks(rotated)
    cur_shoulder_hip_dist = np.linalg.norm(
        np.array([rotated_lm[11].x * w, rotated_lm[11].y * h]) -
        np.array([rotated_lm[23].x * w, rotated_lm[23].y * h])
    )

    if target_shoulder_hip_dist is not None and cur_shoulder_hip_dist > 0:
        scale_factor = target_shoulder_hip_dist / cur_shoulder_hip_dist
        rotated = cv2.resize(rotated, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

    return rotated

def crop_region(image, lm, main_idx1, main_idx2, extra_idx=None, padding=40, force_lower_full=False):
    
    if extra_idx is None:
        extra_idx = []

    h, w = image.shape[:2]

    try:
        xs = [lm[main_idx1].x, lm[main_idx2].x] + [lm[i].x for i in extra_idx]
        ys = [lm[main_idx1].y, lm[main_idx2].y] + [lm[i].y for i in extra_idx]
    except IndexError:
        return None

    x1 = int(min(xs) * w) - padding
    y1 = int(min(ys) * h) - padding
    x2 = int(max(xs) * w) + padding
    y2 = int(max(ys) * h) + padding

    if force_lower_full:
        y2 = h

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    return image[y1:y2, x1:x2]

def process_and_save(img, landmarks, label):
    """Processes and saves all regions"""
    regions = {
        "left_arm": crop_region(img, landmarks, 11, 15, extra_idx=[13, 17]),
        "right_arm": crop_region(img, landmarks, 12, 16, extra_idx=[14, 18]),
        "chest": crop_region(img, landmarks, 11, 12, extra_idx=[23, 24]),
        "left_leg": crop_region(img, landmarks, 23, 27, extra_idx=[25, 29, 31], force_lower_full=True),
        "right_leg": crop_region(img, landmarks, 24, 28, extra_idx=[26, 30, 32], force_lower_full=True),
        "full_body": crop_region(img, landmarks, 0, 32, extra_idx=list(range(33)))
    }

    for name, region in regions.items():
        if region is not None and region.size > 0:
            cv2.imwrite(f"{name}_{label}.jpg", region)

def main():
    img1 = cv2.imread("image1.jpg")
    img2 = cv2.imread("image2.jpg")

    if img1 is None or img2 is None:
        raise FileNotFoundError("One or both input images not found!")

    print("Extracting landmarks...")
    lm1 = get_landmarks(img1)
    lm2 = get_landmarks(img2)

    h1, w1 = img1.shape[:2]
    ref_shoulder_hip_dist = np.linalg.norm(
        np.array([lm1[11].x * w1, lm1[11].y * h1]) -
        np.array([lm1[23].x * w1, lm1[23].y * h1])
    )

    print("Aligning & scaling images...")
    img1_aligned = align_and_scale_image(img1, lm1, target_shoulder_hip_dist=ref_shoulder_hip_dist)
    img2_aligned = align_and_scale_image(img2, lm2, target_shoulder_hip_dist=ref_shoulder_hip_dist)

    lm1_aligned = get_landmarks(img1_aligned)
    lm2_aligned = get_landmarks(img2_aligned)

    print("Saving cropped regions...")
    process_and_save(img1_aligned, lm1_aligned, "1")
    process_and_save(img2_aligned, lm2_aligned, "2")

    print("Done. Cropped regions saved.")

if __name__ == "__main__":
    main()
