import face_recognition
import os
import cv2
import numpy as np
import sqlite3
import pandas as pd
from mtcnn import MTCNN  # Import MTCNN for face detection
from datetime import datetime

# Function to calculate confidence


def face_confidence(face_distance, face_match_threshold=0.6):
    range_val = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range_val * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) +
                 (linear_val - 0.5) ** 2)) * 100
        return str(round(value, 2)) + '%'


def initialize_database():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()

    # Create the table if it doesn't exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        period_1 INTEGER DEFAULT 0,
        period_2 INTEGER DEFAULT 0,
        period_3 INTEGER DEFAULT 0,
        period_4 INTEGER DEFAULT 0,
        period_5 INTEGER DEFAULT 0,
        period_6 INTEGER DEFAULT 0,
        period_7 INTEGER DEFAULT 0,
        period_8 INTEGER DEFAULT 0,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    # Insert known students with default attendance (absent) for all periods
    known_students = os.listdir('Faces')
    for student_image in known_students:
        # Remove file extension to get the name
        student_name = student_image.split('.')[0]

        # Check if the student is already in the database
        c.execute('''
        SELECT * FROM attendance WHERE name = ?
        ''', (student_name,))
        existing_record = c.fetchone()

        # Insert only if the student is not already in the database
        if not existing_record:
            c.execute('''
            INSERT INTO attendance (name)
            VALUES (?)
            ''', (student_name,))
            print(f"Inserted {student_name} into the database.")

    conn.commit()
    conn.close()


# Function to export data to Excel
def export_to_excel():
    try:
        conn = sqlite3.connect('attendance.db')
        df = pd.read_sql_query("SELECT * FROM attendance", conn)
        conn.close()

        if not df.empty:  # Check if DataFrame is not empty
            df.to_excel('final.xlsx', index=False)
            print("Attendance exported to final.xlsx successfully.")
        else:
            print("No attendance records to export.")
    except Exception as e:
        print("Error while exporting to Excel:", str(e))


# Face Recognition class
class FaceRecognition:
    def __init__(self):
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.known_face_encodings = []
        self.known_face_names = []
        self.process_current_frame = True
        self.encode_faces()

        # Initialize MTCNN for face detection
        self.mtcnn_detector = MTCNN()

    def encode_faces(self):
        for image in os.listdir('Faces'):
            face_image = face_recognition.load_image_file(f'Faces/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            # Store names without file extension
            self.known_face_names.append(image.split('.')[0])

        print("Known faces:", self.known_face_names)

    def get_current_period(self):
        """Function to return the current period (1-8 based on time)."""
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
        elif 15 <= hour < 16:
            return 8
        else:
            return None  # Outside school hours, no period.

    def log_attendance(self, name):
        period = self.get_current_period()  # Get the current period
        if not period:
            print("Outside school hours. Attendance not logged.")
            return

        conn = sqlite3.connect('attendance.db')
        c = conn.cursor()

        # Check if the name is already logged today for the current period
        c.execute(
            "SELECT * FROM attendance WHERE name=? AND timestamp >= datetime('now', 'localtime', 'start of day')", (name,))
        existing_record = c.fetchone()

        if existing_record:
            # If the student is already present today, just mark their attendance for the current period
            period_column = f"period_{period}"
            c.execute(f"UPDATE attendance SET {
                      period_column} = 1 WHERE name=? AND timestamp >= datetime('now', 'localtime', 'start of day')", (name,))
            print(f"Attendance marked for {name} in Period {period}.")
        else:
            # If no record for the student today, create a new record with attendance for the current period
            period_column = f"period_{period}"
            c.execute(f"INSERT INTO attendance (name, {
                      period_column}) VALUES (?, 1)", (name,))
            print(f"Attendance logged for {name} in Period {period}.")

        conn.commit()
        conn.close()

        # Call export_to_excel() after logging attendance
        export_to_excel()  # Export attendance to Excel each time a name is logged

    def scan_photo(self, photo_path):
        """Function to scan a given photo for face recognition."""
        print(f"Scanning photo: {photo_path}")

        # Load the photo to be scanned
        image = face_recognition.load_image_file(photo_path)

        # Find all faces in the image
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        if not face_encodings:
            print("No faces found in the image.")
            return

        # Compare the faces found in the image with the known faces
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                self.known_face_encodings, face_encoding, tolerance=0.5)
            name = 'Unknown'
            confidence = 'Unknown'

            face_distances = face_recognition.face_distance(
                self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
                confidence = face_confidence(face_distances[best_match_index])
                # Log attendance for recognized students
                self.log_attendance(name)
                print(f"Recognized {name} with confidence: {confidence}")
            else:
                print("No known faces recognized.")

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            print('Video source not found ....')
            return

        while True:
            ret, frame = video_capture.read()

            if self.process_current_frame:
                # Resize frame to make processing faster
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

                # Use MTCNN to detect faces
                results = self.mtcnn_detector.detect_faces(rgb_small_frame)
                self.face_locations = []

                for result in results:
                    # Get the bounding box coordinates (x, y, w, h)
                    x, y, w, h = result['box']
                    # (top, right, bottom, left)
                    self.face_locations.append((y, x + w, y + h, x))

                self.face_encodings = face_recognition.face_encodings(
                    rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings, face_encoding, tolerance=0.5)
                    name = 'Unknown'
                    confidence = 'Unknown'

                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(
                            face_distances[best_match_index])
                        # Log attendance for recognized students
                        self.log_attendance(name)

                    self.face_names.append(f'{name} ({confidence})')

            self.process_current_frame = not self.process_current_frame

            # Draw bounding boxes and names for each detected face
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw rectangle around face
                cv2.rectangle(frame, (left, top),
                              (right, bottom), (0, 0, 255), 2)

                # Draw rectangle for name tag
                cv2.rectangle(frame, (left, bottom - 35),
                              (right, bottom), (0, 0, 255), -1)

                # Put name and confidence text
                cv2.putText(frame, name, (left + 6, bottom - 6),
                            cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            # Show the video with the face bounding boxes
            cv2.imshow('Face Recognition', frame)

            # Exit loop on 'q' key press
            if cv2.waitKey(1) == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    initialize_database()  # Initialize the database
    fr = FaceRecognition()
    # fr.run_recognition()
    fr.scan_photo('Group/group_4.jpeg')
