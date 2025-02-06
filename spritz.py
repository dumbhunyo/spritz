import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

# Start video capture
video_capture = cv2.VideoCapture(0)

# Function to load and convert images to RGB
def load_and_encode_image(image_path):
    image = cv2.imread(image_path)  # Load image with OpenCV
    if image is None:
        raise RuntimeError(f"Error loading image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    encodings = face_recognition.face_encodings(image)
    
    if not encodings:
        raise RuntimeError(f"No face detected in image: {image_path}")
    
    return encodings[0]  # Return the first encoding found

# Load known faces
try:
    jen_encoding = load_and_encode_image("photos\jen.jpg")
    kate_encoding = load_and_encode_image("photos\kate.jpg")
except RuntimeError as e:
    print(e)
    exit()

# Store encodings and names
known_face_encoding = [jen_encoding, kate_encoding]
known_face_names = ["jen", "kate"]
students = known_face_names.copy()

# Face detection variables
face_locations = []
face_encodings = []
face_names = []
s = True

# Get current date for CSV file
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Create CSV file
with open(current_date + ".csv", "w+", newline="") as f:
    lnwriter = csv.writer(f)
    
    while True:
        _, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        if s:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
                name = ""
                face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
                
                if len(face_distance) > 0:
                    best_match_index = np.argmin(face_distance)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                face_names.append(name)

                if name in known_face_names and name in students:
                    students.remove(name)
                    print(f"{name} marked as present.")
                    current_time = datetime.now().strftime("%H-%M-%S")
                    lnwriter.writerow([name, current_time])

        cv2.imshow("Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release video capture and close windows
video_capture.release()
cv2.destroyAllWindows()