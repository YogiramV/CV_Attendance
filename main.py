import face_recognition
import os
import cv2
import numpy as np
import sqlite3
import pandas as pd

def face_confidence(face_distance, face_match_threshold=0.6):
    range_val = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range_val * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) + (linear_val - 0.5) ** 2)) * 100
        return str(round(value, 2)) + '%'

def initialize_database():
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()

    # Create a table if it doesn't already exist
    c.execute('''
    CREATE TABLE IF NOT EXISTS attendance (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')

    conn.commit()
    conn.close()

def export_to_excel():
    try:
        conn = sqlite3.connect('attendance.db')
        df = pd.read_sql_query("SELECT * FROM attendance", conn)
        conn.close()

        if not df.empty:  # Check if DataFrame is not empty
            df.to_excel('final.xlsx', index=False)
            print("Attendance exported to final_attendance.xlsx successfully.")
        else:
            print("No attendance records to export.")

    except Exception as e:
        print("Error while exporting to Excel:", str(e))

class FaceRecognition:
    def __init__(self):
        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.known_face_encodings = []
        self.known_face_names = []
        self.process_current_frame = True
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('Faces'):
            face_image = face_recognition.load_image_file(f'Faces/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image.split('.')[0])  # Store names without file extension

        print("Known faces:", self.known_face_names)

    def is_name_logged_in_excel(self, name):
        try:
            df = pd.read_excel('final_attendance.xlsx')
            return name in df['name'].values
        except FileNotFoundError:
            return False
        except Exception as e:
            print("Error reading Excel file:", str(e))
            return False

    def log_attendance(self, name):
        conn = sqlite3.connect('attendance.db')
        c = conn.cursor()

        # Check if the name is already logged in the Excel file
        if self.is_name_logged_in_excel(name):
            print(f"{name} has already been logged in the Excel file.")
            conn.close()
            return  # Exit the method if the name is already logged

        # Check for existing attendance records
        c.execute("SELECT * FROM attendance WHERE name=? AND timestamp >= datetime('now', 'localtime', 'start of day')", (name,))
        existing_record = c.fetchone()

        # Log attendance only if the name is not already recorded today
        if existing_record is None:
            c.execute("INSERT INTO attendance (name) VALUES (?)", (name,))
            conn.commit()
            print(f"Attendance logged for {name}.")
        else:
            print(f"Attendance for {name} already logged.")

        conn.close()

        # Call export_to_excel() after logging attendance
        export_to_excel()  # Export attendance to Excel each time a name is logged

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            print('Video source not found ....')
            return

        while True:
            ret, frame = video_capture.read()

            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, tolerance=0.5)
                    name = 'Unknown'
                    confidence = 'Unknown'

                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])
                        self.log_attendance(name)  # Log attendance for recognized students

                    self.face_names.append(f'{name} ({confidence})')

            self.process_current_frame = not self.process_current_frame

            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    initialize_database()  # Initialize the database
    fr = FaceRecognition()
    fr.run_recognition()
