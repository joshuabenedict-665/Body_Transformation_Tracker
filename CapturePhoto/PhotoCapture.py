import cv2
import time
import os
import mediapipe as mp
import numpy as np

def main():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=True)
    draw_utils = mp.solutions.drawing_utils

    if not os.path.exists("image1.jpg"):
        img_path = "image1.jpg"
    elif not os.path.exists("image2.jpg"):
        img_path = "image2.jpg"
    else:
        print(" Both images already exist.")
        return

    cap = cv2.VideoCapture(0)
    print(" Capturing will start in 10 seconds... Stand straight!")
    time.sleep(10)

    if not cap.isOpened():
        print("Cannot access camera.")
        return

    print(" Camera started... Waiting for stable standing posture...")

    stable_start_time = None
    required_duration = 3 

    required_landmarks = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_ELBOW,
        mp_pose.PoseLandmark.RIGHT_ELBOW,
        mp_pose.PoseLandmark.LEFT_WRIST,
        mp_pose.PoseLandmark.RIGHT_WRIST,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP
    ]

    def all_upper_body_visible(landmarks):
        return all(landmarks[l.value].visibility > 0.7 for l in required_landmarks)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)

        if result.segmentation_mask is not None:
            condition = result.segmentation_mask > 0.5
            bg_image = np.zeros(frame.shape, dtype=np.uint8)
            frame = np.where(condition[..., None], frame, bg_image)

        if result.pose_landmarks:
            draw_utils.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = result.pose_landmarks.landmark

            if all_upper_body_visible(landmarks):
                ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                lh = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                la = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]

                if abs(ls.x - lh.x) < 0.05 and abs(lh.x - la.x) < 0.05:
                    if stable_start_time is None:
                        stable_start_time = time.time()
                    elif time.time() - stable_start_time >= required_duration:
                        cv2.imwrite(img_path, frame)
                        print(f" Stable standing posture detected — Image saved as '{img_path}'")
                        break
                else:
                    stable_start_time = None
            else:
                stable_start_time = None
        else:
            stable_start_time = None

        cv2.imshow("Pose Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(" Cancelled by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(" Camera released. Exiting...")

if __name__ == "__main__":
    main()
