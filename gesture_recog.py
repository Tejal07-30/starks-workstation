def fingers_up(hand_landmarks, hand_label):
    """
    This returns a list of five booleans:
    [Thumb, Index finger, Middle finger, Ring finger, Pinky finger]
    True if finger is open
    False if finger is folded
    """

    landmarks = hand_landmarks.landmark
    fingers = []
# this is fr thumb
    if hand_label == "Right":
        thumb_open = landmarks[4].x > landmarks[3].x
    else:
        thumb_open = landmarks[4].x < landmarks[3].x

    fingers.append(thumb_open)

# other fingers
    tip_ids = [8, 12, 16, 20]

    for tip in tip_ids:
        fingers.append(
            landmarks[tip].y < landmarks[tip - 2].y
        )

    return fingers

def classify_gesture(fingers):
    
    # This converts finger state into gesture names.
    
    if fingers == [True, True, True, True, True]:
        return "Open Palm"

    elif fingers == [False, False, False, False, False]:
        return "Closed Fist"

    elif fingers == [False, True, True, False, False]:
        return "Victory"

    elif fingers == [True, False, False, False, False]:
        return "Thumbs Up"

    return "Unknown"