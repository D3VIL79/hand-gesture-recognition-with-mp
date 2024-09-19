import cv2
import numpy as np
import mediapipe as mp
import face_recognition
from tensorflow.keras.models import load_model

# Face Recognition Setup
known_face_encodings = []
known_face_names = []

def add_person(image_path, name):
    person_image = face_recognition.load_image_file(image_path)
    person_encoding = face_recognition.face_encodings(person_image)
    if person_encoding:
        known_face_encodings.append(person_encoding[0])
        known_face_names.append(name)
        print(f"Added {name}'s face.")
    else:
        print(f"No face found in the image for {name}.")

add_person("C:\\person.jpg", "om")

# Hand Gesture Recognition Setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

gesture_model = load_model('mp_hand_gesture')

f = open('gesture.names', 'r')
gesture_class_names = f.read().split('\n')
f.close()
print(gesture_class_names)

# Video Capture Setup
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()

    x, y, c = frame.shape

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face Recognition
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Hand Gesture Recognition
    result = hands.process(frame_rgb)
    gesture_class_name = ''

    if result.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])

            mpDraw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

            prediction = gesture_model.predict([landmarks])
            class_id = np.argmax(prediction)
            gesture_class_name = gesture_class_names[class_id]

    cv2.putText(frame, gesture_class_name, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    # Display Results
    cv2.imshow("Face and Hand Recognition", frame)

    # Exit Condition
    if cv2.waitKey(1) == ord('q'):
        break

# Release Resources
cap.release()
cv2.destroyAllWindows()
