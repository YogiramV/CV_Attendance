import streamlit as st
from main import initialize_database, scan_photo, load_encodings
from image_encoder import encode_faces_once  # Import the encoding function
from PIL import Image
import io

# Streamlit app UI


def app():
    initialize_database()
    st.title("Face Recognition Attendance System")

    # Add option to encode faces (one-time process)
    if st.button("Encode Faces (One-time Process)"):
        encode_faces_once()  # Run the encoding process
        st.success("Faces have been successfully encoded and saved!")

    # Load encodings from the pickle file
    known_face_encodings, known_face_rollnos, known_face_names = load_encodings()

    # Capture image directly from webcam using Streamlit's camera_input
    st.subheader("Capture Photo for Recognition")
    rollno_input = st.text_input("Enter your Roll Number")

    # Capture image from webcam
    captured_image = st.camera_input("Take a photo")

    if captured_image and rollno_input:
        # Convert the UploadedFile object to a byte stream
        image_bytes = captured_image.getvalue()

        # Open the image from the byte stream using PIL
        img = Image.open(io.BytesIO(image_bytes))

        # Display the captured image with the updated parameter
        st.image(img, caption="Captured Image", use_container_width=True)

        # Save the captured image temporarily
        img.save("temp_uploaded_image.jpg")

        # Scan uploaded image for the entered roll number and display result
        result = scan_photo("temp_uploaded_image.jpg", rollno_input,
                            known_face_encodings, known_face_rollnos, known_face_names)

        # Display the result in Streamlit
        st.write(result)


if __name__ == "__main__":
    app()
