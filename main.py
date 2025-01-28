import pandas as pd
import face_recognition
from datetime import datetime
import numpy as np
import pickle
import os

known_face_encodings, known_face_rollnos, known_face_names = [], [], []

ENCODED_FACES_TIMESTAMP_FILE = "encoded_faces_timestamp.txt"

def load_encodings():
    global known_face_encodings, known_face_rollnos, known_face_names
    try:
        if not known_face_encodings:
            if os.path.exists('known_faces.pkl'):
                with open('known_faces.pkl', 'rb') as f:
                    data = pickle.load(f)
                known_face_encodings = data['encodings']
                known_face_rollnos = data['rollnos']
                known_face_names = data['names']
            else:
                known_face_encodings, known_face_rollnos, known_face_names = [], [], []
    except Exception as e:
        print("Error loading encodings:", str(e))

    return known_face_encodings, known_face_rollnos, known_face_names

def scan_photo(image_path, rollno_input, known_face_encodings, known_face_rollnos, known_face_names):
    image = np.array(image_path) 
    face_locations = face_recognition.face_locations(image)
    face_encodings_in_image = face_recognition.face_encodings(image, face_locations)

    if len(face_encodings_in_image) == 0:
        return "No faces found in the image."

    best_match_index = -1
    min_distance = float('inf')

    for face_encoding in face_encodings_in_image:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        if len(face_distances) == 0:
            return "No valid face encodings found."

        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            min_distance = face_distances[best_match_index]

    if best_match_index != -1 and min_distance < 0.6:
        recognized_rollno = known_face_rollnos[best_match_index]
        recognized_name = known_face_names[best_match_index]
        confidence = round((1.0 - min_distance) * 100, 2)

        if recognized_rollno.lower() == rollno_input.lower():
            return True
        else:
            return False
    else:
        return "No matching face found."
