import random
import cv2
import time
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import numpy as np
import datetime
import json
import os


# -------------------- RL Agent --------------------
class RLAgent:
    def __init__(self):
        self.actions = ["none", "popup"]
        self.q_table = {
            "focused": {"none": 0, "popup": 0},
            "distracted": {"none": 0, "popup": 1}
        }
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.2

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)
        return max(self.q_table[state], key=self.q_table[state].get)

    def update(self, state, action, reward):
        old_val = self.q_table[state][action]
        next_max = max(self.q_table[state].values())
        new_val = old_val + self.alpha * (reward + self.gamma * next_max - old_val)
        self.q_table[state][action] = new_val


# -------------------- Drowsiness / Attention Detector --------------------
class DrowsinessDetector:
    def __init__(self, thresh=0.25, frame_check=20,
                 model_path="models/shape_predictor_68_face_landmarks.dat",
                 music_path="music.wav"):
        self.thresh = thresh
        self.frame_check = frame_check
        self.flag = 0
        self.music_path = music_path
        self.detect = dlib.get_frontal_face_detector()
        self.predict = dlib.shape_predictor(model_path)
        self.lStart, self.lEnd = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
        self.rStart, self.rEnd = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

        mixer.init()
        if os.path.exists(music_path):
            mixer.music.load(music_path)

    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    def detect_attention(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = self.detect(gray, 0)

        for subject in subjects:
            shape = self.predict(gray, subject)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]

            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            if ear < self.thresh:
                self.flag += 1
                if self.flag >= self.frame_check:
                    # Drowsy or distracted
                    if os.path.exists(self.music_path):
                        try:
                            mixer.music.play()
                        except:
                            pass
                    return "distracted", ear
            else:
                self.flag = 0
                return "focused", ear

        return "focused", 0.3  # default if no face detected


# -------------------- Integration --------------------
def main():
    rl_agent = RLAgent()
    detector = DrowsinessDetector()

    cap = cv2.VideoCapture(0)

    print("Starting Attention + RL System... Press 'q' to quit.")
    last_state = "focused"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        state, ear = detector.detect_attention(frame)
        action = rl_agent.choose_action(state)

        # Define reward logic
        if state == "focused":
            reward = 1 if action == "none" else -1  # popup while focused = bad
        else:
            reward = 1 if action == "popup" else -1  # popup while distracted = good

        rl_agent.update(state, action, reward)

        # Show on screen
        cv2.putText(frame, f"State: {state.upper()}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Action: {action.upper()}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"EAR: {ear:.3f}", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if action == "popup":
            cv2.putText(frame, "⚠️ PAY ATTENTION!", (100, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        cv2.imshow("Attention RL System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Final Q-Table:")
    print(rl_agent.q_table)


if __name__ == "__main__":
    main()
