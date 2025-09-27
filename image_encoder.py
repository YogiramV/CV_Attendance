import face_recognition
import os
import pickle
import numpy as np
from sklearn.decomposition import PCA


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

    if known_face_encodings:
        encodings_array = np.array(known_face_encodings)

        pca = PCA(n_components=min(50, len(known_face_encodings) - 1))
        reduced_encodings = pca.fit_transform(encodings_array)

        with open('pca_model.pkl', 'wb') as f:
            pickle.dump(pca, f)

        known_face_encodings = reduced_encodings.tolist()

    with open('known_faces.pkl', 'wb') as f:
        pickle.dump({'encodings': known_face_encodings,
                     'rollnos': known_face_rollnos,
                     'names': known_face_names}, f)

    print(
        f"Encoded and saved {len(known_face_encodings)} faces with PCA dimensionality reduction (n_components={min(50, len(known_face_encodings) - 1)}).")
