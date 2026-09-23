import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from gesture_recog import fingers_up, classify_gesture, detect_swipe

# Initial Setup
camera = cv2.VideoCapture(0) #continuously provides video frames of default cam.

# Screen size
screen_width, screen_height = pyautogui.size()

# Safety settings
pyautogui.FAILSAFE = True #stopd when the cursor reaches the top-left corner.
pyautogui.PAUSE = 0 # gives a pause between two pyautogui commands.

# MediaPipe setup
mp_hands = mp.solutions.hands 
mp_draw = mp.solutions.drawing_utils 

hands = mp_hands.Hands(
    max_num_hands=1, 
    min_detection_confidence=0.7, #confidence required for a new hand detection
    min_tracking_confidence=0.7 #landmark tracking model to follow the hand across consecutive frames
)

# Cursor settings
FRAME_MARGIN = 100
SMOOTHING = 0.25

prev_x = 0
prev_y = 0

# Storing fingertip history for swipe detection
history = []

while True:

    success, frame = camera.read()

    if not success:
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1) #horizontally flips the video frame

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #converts BGR to RGB

    results = hands.process(rgb)# mediapipe analyzes the frame

    if results.multi_hand_landmarks:

        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks, #landmarks
            results.multi_handedness #handlabels
        ):

            hand_label = handedness.classification[0].label #returns right or left

            # Landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            h, w, _ = frame.shape

            # Index fingertip
            index_tip = hand_landmarks.landmark[8]

            x = int(index_tip.x * w) #convert coordinates in pixels
            y = int(index_tip.y * h)

            # Tracking history for swipe
            history.append((x, y))

            if len(history) > 10:
                history.pop(0) #creates a sliding window

            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

            # Detecting gesture
            finger_states = fingers_up(hand_landmarks, hand_label)
            gesture, confidence = classify_gesture(finger_states)

            swipe = None

            if gesture == "Open Palm":
                swipe = detect_swipe(history)

            if swipe:
                gesture = swipe
                confidence = 100

            # Virtual Mouse Movement
            mouse_x = np.interp(
                x,
                (FRAME_MARGIN, w - FRAME_MARGIN),
                (0, screen_width)
            )

            mouse_y = np.interp(
                y,
                (FRAME_MARGIN, h - FRAME_MARGIN),
                (0, screen_height)
            )

            # Smooth movement
            smooth_x = prev_x + (mouse_x - prev_x) * SMOOTHING #The cursor gradually moves instead of jumping
            smooth_y = prev_y + (mouse_y - prev_y) * SMOOTHING

            prev_x = smooth_x
            prev_y = smooth_y

            # Moving cursor with open palm only
            if gesture == "Open Palm":
                pyautogui.moveTo(smooth_x, smooth_y) #update the mouse pointer's position on screen in real time."

            # Displaying Text

            cv2.putText( #text onto an image. image, text, position, font, scale, color, thickness
                frame,
                f"Gesture: {gesture} ({confidence}%)",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

            cv2.putText(
                frame,
                f"Hand: {hand_label}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

    else:

        history.clear() #prevents false positives

        cv2.putText(  
            frame,
            "No Hand Detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    h, w = frame.shape[:2]

    cv2.rectangle( #tracking box
        frame,
        (FRAME_MARGIN, FRAME_MARGIN),
        (w - FRAME_MARGIN, h - FRAME_MARGIN),
        (255, 255, 0),
        2
    )

    cv2.imshow("Stark's Workstation", frame) #displays the processed frame

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Cleanup and closing 
camera.release() #Releases the webcam
cv2.destroyAllWindows() #Closes every OpenCV window
