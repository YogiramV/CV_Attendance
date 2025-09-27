import sqlite3
import pandas as pd
import face_recognition
from datetime import datetime
import pytz
import numpy as np
import pickle
import os


timezone = pytz.timezone("Asia/Kolkata")


def initialize_database():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            name TEXT,
            rollno TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            period_1 INTEGER DEFAULT 0,
            period_2 INTEGER DEFAULT 0,
            period_3 INTEGER DEFAULT 0,
            period_4 INTEGER DEFAULT 0,
            period_5 INTEGER DEFAULT 0,
            period_6 INTEGER DEFAULT 0,
            period_7 INTEGER DEFAULT 0,
            period_8 INTEGER DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()


def log_attendance(name, rollno, period):
    if period is None:
        print("Attendance cannot be marked as it's outside school hours.")
        return "Attendance cannot be marked as it's outside school hours."

    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()

    c.execute("SELECT * FROM attendance WHERE name = ? AND rollno = ? AND timestamp >= datetime('now', 'localtime', 'start of day')", (name, rollno))
    existing_record = c.fetchone()

    if existing_record:
        period_column = f"period_{period}"
        c.execute(f"UPDATE attendance SET {
                  period_column} = 1 WHERE name = ? AND rollno = ? AND timestamp >= datetime('now', 'localtime', 'start of day')", (name, rollno))
    else:
        period_column = f"period_{period}"
        c.execute(f"INSERT INTO attendance (name, rollno, {
                  period_column}) VALUES (?, ?, 1)", (name, rollno))

    conn.commit()

    export_to_excel()

    conn.close()


def export_to_excel():
    """Export the attendance data to an Excel file."""
    try:
        conn = sqlite3.connect('attendance.db')
        df = pd.read_sql_query("SELECT * FROM attendance", conn)
        conn.close()

        if not df.empty:
            df.to_excel('attendance_data.xlsx', index=False)
            print("Attendance exported to attendance_data.xlsx successfully.")
        else:
            print("No attendance records to export.")
    except Exception as e:
        print("Error while exporting to Excel:", str(e))


def load_encodings():
    try:
        with open('known_faces.pkl', 'rb') as f:
            data = pickle.load(f)
        known_face_encodings = data['encodings']
        known_face_rollnos = data['rollnos']
        known_face_names = data['names']
        
        # Convert back to numpy array for face recognition
        known_face_encodings = [np.array(enc) for enc in known_face_encodings]
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

    # Load the PCA model if it exists
    pca = None
    if os.path.exists('pca_model.pkl'):
        with open('pca_model.pkl', 'rb') as f:
            pca = pickle.load(f)

    for face_encoding in face_encodings_in_image:
        # Apply PCA transformation if PCA model exists
        if pca is not None:
            face_encoding = pca.transform([face_encoding])[0]
            
        # Convert to numpy array for comparison
        face_encoding = np.array(face_encoding)
        
        # Calculate distances to all known faces
        face_distances = [np.linalg.norm(np.array(known_enc) - face_encoding) 
                         for known_enc in known_face_encodings]
        
        if not face_distances:  # If no known faces to compare with
            return "No known faces to compare with."
            
        best_match_index = np.argmin(face_distances)
        matches = [d <= 0.6 for d in face_distances]  # 0.6 is the threshold for a match

        if len(face_distances) == 0:
            return "No valid face encodings found."

        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            recognized_rollno = known_face_rollnos[best_match_index]
            recognized_name = known_face_names[best_match_index]
            confidence = round(
                (1.0 - face_distances[best_match_index]) * 100, 2)

            if recognized_rollno == rollno_input:
                current_period = get_current_period()
                if current_period is None:
                    return "Attendance cannot be marked as it's outside school hours."

                log_attendance(recognized_name,
                               recognized_rollno, current_period)
                return f"Attendance marked for {recognized_name} (Roll No: {recognized_rollno}) in Period {current_period} with {confidence}% confidence."
            else:
                return f"The face does not match the roll number {rollno_input}. Please try again."

        else:
            return "No known face recognized."


def get_current_period():
    hour = datetime.now(timezone).hour
    if 8 <= hour < 9:
        return 1
    elif 9 <= hour < 10:
        return 2
    elif 10 <= hour < 11:
        return 3
    elif 11 <= hour < 12:
        return 4
    elif 12 <= hour < 13:
        return 5
    elif 13 <= hour < 14:
        return 6
    elif 14 <= hour < 15:
        return 7
    elif 15 <= hour < 17:
        return 8
    return None
