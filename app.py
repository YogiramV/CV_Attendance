from main import scan_photo, load_encodings
from PIL import Image
import io

known_face_encodings, known_face_rollnos, known_face_names = [], [], []

encodings_loaded = False

def process_face_recognition(captured_image, rollno_input):
    global known_face_encodings, known_face_rollnos, known_face_names, encodings_loaded


    if not encodings_loaded:
        known_face_encodings, known_face_rollnos, known_face_names = load_encodings()
        encodings_loaded = True
    


    image_bytes = captured_image.getvalue()
    img = Image.open(io.BytesIO(image_bytes))

    result = scan_photo(img, rollno_input, known_face_encodings, known_face_rollnos, known_face_names)
    

    return result
