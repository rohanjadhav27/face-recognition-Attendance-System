import cv2
import face_recognition
import numpy as np

import os
print("Current working directory:", os.getcwd())

from datetime import datetime

# Load known faces
known_face_encodings = []
known_face_names = []

path = 'known_faces'
images = os.listdir(path)

for img_name in images:
    img = face_recognition.load_image_file(f"{path}/{img_name}")
    encoding = face_recognition.face_encodings(img)[0]
    known_face_encodings.append(encoding)
    known_face_names.append(os.path.splitext(img_name)[0])
print("Loaded known faces:", known_face_names)
from datetime import datetime
from pathlib import Path

def mark_attendance(name):
    file_path = Path(__file__).parent / 'attendance.csv'  # file in same folder as .py

    try:
        # Create file if it doesn't exist
        if not file_path.exists():
            file_path.touch()
            with open(file_path, 'w') as f:
                f.write('Name,Time,Date\n')

        # Now safely append attendance
        with open(file_path, 'a+') as f:
            f.seek(0)
            data = f.readlines()
            names = [line.split(',')[0] for line in data]

            if name not in names:
                now = datetime.now()
                f.write(f'{name},{now.strftime("%H:%M:%S")},{now.strftime("%Y-%m-%d")}\n')

        print(f"✅ Attendance marked for {name}")

    except Exception as e:
        print("❌ Error writing to CSV:", e)






# Load uploaded class photo
class_img_path = 'class_photo1.jpg'  # Replace with your image
image = face_recognition.load_image_file(class_img_path)
face_locations = face_recognition.face_locations(image)
print("Number of faces detected in class photo:", len(face_locations))
face_encodings = face_recognition.face_encodings(image, face_locations)

# Process each face found
for face_encoding, face_location in zip(face_encodings, face_locations):
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)

    if matches[best_match_index]:
        name = known_face_names[best_match_index].upper()
        mark_attendance(name)

        # Draw box
        top, right, bottom, left = face_location
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, name, (left, bottom + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# Show result
cv2.imshow('Class Image Attendance', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Found", len(face_locations), "faces in class photo")
print("Attendance writing completed.")
