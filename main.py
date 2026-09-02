import cv2
import mediapipe as mp
from gesture_recog import fingers_up, classify_gesture, detect_swipe

camera = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
history = []
while True:
    success, frame = camera.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness):

            hand_label = handedness.classification[0].label

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            h, w, _ = frame.shape

            index_tip = hand_landmarks.landmark[8]

            x = int(index_tip.x * w)
            y = int(index_tip.y * h)
            history.append((x,y))
            if len(history)>10:
                history.pop(0)

            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

            finger_states = fingers_up(hand_landmarks, hand_label)
            gesture, confidence = classify_gesture(finger_states)
            swipe = detect_swipe(history)
            if swipe:
                gesture = swipe
                confidence = 100

            cv2.putText(
                frame,
                f"Gesture: {gesture} ({confidence}%)",
                (20,80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255,0,0),
                2
            )

            cv2.putText(
                frame,
                f"Hand: {hand_label}",
                (20,120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,255),
                2
            )

    else:
        history.clear()
        
        cv2.putText(
            frame,
            "No Hand Detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    cv2.imshow("Stark's Workstation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()