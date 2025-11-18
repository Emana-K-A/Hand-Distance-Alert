import cv2
import mediapipe as mp
import numpy as np
import pygame

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Constants for distance calculation
KNOWN_WIDTH = 8.0   # cm (average palm width)
FOCAL_LENGTH = 500  # calibrated value

# Initialize pygame mixer
pygame.mixer.init()
beep_sound = pygame.mixer.Sound("beep.mp3")

# Function to calculate distance
def calculate_distance(perceived_width):
    if perceived_width == 0:
        return 0
    distance = (KNOWN_WIDTH * FOCAL_LENGTH) / perceived_width

    # Play beep if distance < 30 cm
    if distance < 30:
        beep_sound.play()
    return distance

# Start webcam
cap = cv2.VideoCapture(0)
window_name = 'Hand Distance Measurement'
cv2.namedWindow(window_name)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_min, y_min = float('inf'), float('inf')
            x_max, y_max = 0, 0

            # Find bounding box of hand
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            perceived_width = x_max - x_min
            distance = calculate_distance(perceived_width)

            # Draw visuals
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f'Distance: {distance:.2f} cm', (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow(window_name, frame)

    # Close if 'q' pressed OR window closed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
