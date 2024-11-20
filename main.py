import sqlite3
import pandas as pd
import face_recognition
from datetime import datetime
import numpy as np
import pickle

# Initialize the database


def initialize_database():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()

    # Create table if it doesn't exist
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

# Log attendance into the database


def log_attendance(name, rollno, period):
    if period is None:
        print("Attendance cannot be marked as it's outside school hours.")
        return "Attendance cannot be marked as it's outside school hours."

    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()

    # Check if the student is already logged for the current day and period
    c.execute("SELECT * FROM attendance WHERE name = ? AND rollno = ? AND timestamp >= datetime('now', 'localtime', 'start of day')", (name, rollno))
    existing_record = c.fetchone()

    if existing_record:
        # If record exists, update attendance for the current period
        period_column = f"period_{period}"
        c.execute(f"UPDATE attendance SET {
                  period_column} = 1 WHERE name = ? AND rollno = ? AND timestamp >= datetime('now', 'localtime', 'start of day')", (name, rollno))
    else:
        # If no record, create a new record
        period_column = f"period_{period}"
        c.execute(f"INSERT INTO attendance (name, rollno, {
                  period_column}) VALUES (?, ?, 1)", (name, rollno))

    # Commit the changes
    conn.commit()

    # Export the updated attendance to Excel
    export_to_excel()

    conn.close()

# Export attendance to Excel


def export_to_excel():
    """Export the attendance data to an Excel file."""
    try:
        conn = sqlite3.connect('attendance.db')
        df = pd.read_sql_query("SELECT * FROM attendance", conn)
        conn.close()

        if not df.empty:  # Check if DataFrame is not empty
            df.to_excel('attendance_data.xlsx', index=False)
            print("Attendance exported to attendance_data.xlsx successfully.")
        else:
            print("No attendance records to export.")
    except Exception as e:
        print("Error while exporting to Excel:", str(e))

# Load the face encodings from the pickle file


def load_encodings():
    try:
        with open('known_faces.pkl', 'rb') as f:
            data = pickle.load(f)
        known_face_encodings = data['encodings']
        known_face_rollnos = data['rollnos']
        known_face_names = data['names']  # Add names list to pickle
        return known_face_encodings, known_face_rollnos, known_face_names
    except Exception as e:
        print("Error loading encodings:", str(e))
        return [], [], []

# Scan the captured photo for face recognition


def scan_photo(image_path, rollno_input, known_face_encodings, known_face_rollnos, known_face_names):
    image = face_recognition.load_image_file(image_path)

    # Find face encodings in the image
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

        if len(face_distances) == 0:  # No faces found, skip processing
            return "No valid face encodings found."

        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            # Get the recognized roll number
            recognized_rollno = known_face_rollnos[best_match_index]
            # Get the recognized name
            recognized_name = known_face_names[best_match_index]
            confidence = round(
                (1.0 - face_distances[best_match_index]) * 100, 2)

            if recognized_rollno == rollno_input:
                # Log attendance if roll number matches
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

# Get the current period based on the time of day


def get_current_period():
    hour = datetime.now().hour
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
    return None  # Outside school hours
