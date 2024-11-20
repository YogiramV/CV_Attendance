import face_recognition
import os
import pickle


def encode_faces_once():
    known_face_encodings = []
    known_face_rollnos = []
    known_face_names = []  # Add names list

    # Path to the 'Faces' directory where the images are stored
    faces_directory = 'Faces'

    # Loop through all images in the directory
    for filename in os.listdir(faces_directory):
        if filename.endswith(".jpeg") or filename.endswith(".jpg") or filename.endswith(".png"):
            # Extract roll number from the filename (assuming the format is name_rollno.jpeg)
            name_rollno = filename.split('.')[0]
            name, rollno = name_rollno.split('_')

            # Load the image
            image = face_recognition.load_image_file(
                os.path.join(faces_directory, filename))
            # Get the face encoding for the image
            encoding = face_recognition.face_encodings(image)[0]

            # Append the encoding, roll number, and name to the respective lists
            known_face_encodings.append(encoding)
            known_face_rollnos.append(rollno)  # Store roll number as a string
            known_face_names.append(name)  # Store name

    # Save the encodings, roll numbers, and names to a .pkl file
    with open('known_faces.pkl', 'wb') as f:
        pickle.dump({'encodings': known_face_encodings,
                     'rollnos': known_face_rollnos,
                     'names': known_face_names}, f)

    print(f"Encoded and saved {len(known_face_encodings)} faces.")
