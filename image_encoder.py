import face_recognition
import os
import pickle

# Encode faces only once if necessary
def encode_faces_once():
    known_face_encodings = []
    known_face_rollnos = []
    known_face_names = []  
    faces_directory = 'Faces'

    for filename in os.listdir(faces_directory):
        if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"):
            name_rollno = filename.split('.')[0]
            name, rollno = name_rollno.split('_')

            image = face_recognition.load_image_file(
                os.path.join(faces_directory, filename))
            encoding = face_recognition.face_encodings(image)[0]

            known_face_encodings.append(encoding)
            known_face_rollnos.append(rollno)
            known_face_names.append(name)

    with open('known_faces.pkl', 'wb') as f:
        pickle.dump({'encodings': known_face_encodings,
                     'rollnos': known_face_rollnos,
                     'names': known_face_names}, f)

    print(f"Encoded and saved {len(known_face_encodings)} faces.")
