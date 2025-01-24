from main import scan_photo, load_encodings
from image_encoder import encode_faces_once  # Import the encoding function
from PIL import Image
import io

# Refactored function to process image and roll number
def process_face_recognition(captured_image, rollno_input):
    # This function will do the same as the Streamlit app, but now callable in any context

    # Load encodings from the pickle file
    known_face_encodings, known_face_rollnos, known_face_names = load_encodings()

    # Convert the UploadedFile object to a byte stream
    image_bytes = captured_image.getvalue()

    # Open the image from the byte stream using PIL
    img = Image.open(io.BytesIO(image_bytes))

    # Save the captured image temporarily (you can also skip this step if you don't need to save it)
    img.save("temp_uploaded_image.jpg")

    # Scan the uploaded image for the entered roll number and display result
    result = scan_photo("temp_uploaded_image.jpg", rollno_input,
                        known_face_encodings, known_face_rollnos, known_face_names)

    return result
