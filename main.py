import pandas as pd
import face_recognition
from datetime import datetime
import numpy as np
import pickle


def load_encodings():
    try:
        with open('known_faces.pkl', 'rb') as f:
            data = pickle.load(f)
        known_face_encodings = data['encodings']
        known_face_rollnos = data['rollnos']
        known_face_names = data['names']
        return known_face_encodings, known_face_rollnos, known_face_names
    except Exception as e:
        print("Error loading encodings:", str(e))
        return [], [], []


def scan_photo(image_path, rollno_input, known_face_encodings, known_face_rollnos, known_face_names):
    image = face_recognition.load_image_file(image_path)

    face_locations = face_recognition.face_locations(image)
    face_encodings_in_image = face_recognition.face_encodings(
        image, face_locations)

    if len(face_encodings_in_image) == 0:
        return "No faces found in the image."

    for face_encoding in face_encodings_in_image:
        matches = face_recognition.compare_faces(
            known_face_encodings, face_encoding, tolerance=0.6)
        face_distances = face_recognition.face_distance(
            known_face_encodings, face_encoding)

        if len(face_distances) == 0:
            return "No valid face encodings found."

        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            recognized_rollno = known_face_rollnos[best_match_index]
            recognized_name = known_face_names[best_match_index]
            confidence = round(
                (1.0 - face_distances[best_match_index]) * 100, 2)

            if recognized_rollno.lower() == rollno_input.lower():
                return True
            else:
                return False
