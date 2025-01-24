import os
from main import scan_photo, load_encodings
from image_encoder import encode_faces_once
from PIL import Image
import io


def process_face_recognition(captured_image, rollno_input):

    known_face_encodings, known_face_rollnos, known_face_names = load_encodings()

    image_bytes = captured_image.getvalue()

    img = Image.open(io.BytesIO(image_bytes))

    temp_image_path = "temp_uploaded_image.jpg"
    img.save(temp_image_path)

    result = scan_photo(temp_image_path, rollno_input,
                        known_face_encodings, known_face_rollnos, known_face_names)

    os.remove(temp_image_path)

    return result
