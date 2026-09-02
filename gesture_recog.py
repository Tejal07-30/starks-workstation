import math


def distance(p1, p2):
    #Euclidean distance between two landmarks.
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def fingers_up(hand_landmarks, hand_label):
    
    landmarks = hand_landmarks.landmark
    fingers = []

    thumb_distance = distance(landmarks[4], landmarks[5])

    if hand_label == "Right":
        thumb_open = thumb_distance > 0.09 and landmarks[4].x > landmarks[3].x
    else:
        thumb_open = thumb_distance > 0.09 and landmarks[4].x < landmarks[3].x

    fingers.append(thumb_open)

    tip_ids = [8, 12, 16, 20]

    for tip in tip_ids:
        fingers.append(landmarks[tip].y < landmarks[tip - 2].y)

    return fingers


def classify_gesture(fingers):
    gestures = {
        "Open Palm": [True, True, True, True, True],
        "Closed Fist": [False, False, False, False, False],
        "Victory": [False, True, True, False, False],
        "Thumbs Up": [True, False, False, False, False],
    }

    best_gesture = "Unknown"
    best_score = -1

    for name, pattern in gestures.items():
        score = sum(a == b for a, b in zip(fingers, pattern))

        if score > best_score:
            best_score = score
            best_gesture = name

    confidence = int((best_score / 5) * 100)

    if confidence < 60:
        return "Unknown", confidence

    return best_gesture, confidence


def detect_swipe(history):
    
    if len(history) < 10:
        return None

    start_x, start_y = history[0]
    end_x, end_y = history[-1]

    dx = end_x - start_x
    dy = abs(end_y - start_y)

    if dy > 40:
        return None

    if dx > 120:
        return "Swipe Right"

    if dx < -120:
        return "Swipe Left"

    return None