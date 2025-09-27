import streamlit as st
import pandas as pd
from main import initialize_database, scan_photo, load_encodings
from image_encoder import encode_faces_once
from PIL import Image
import io
import os

# Path to the local Excel file (replace with your actual file name)
ATTENDANCE_FILE = "attendance_data.xlsx"  # Replace with your actual file name

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

    # Button to display attendance (from the local file)
    if st.button("Show Attendance Records"):
        # Check if the attendance file exists in the current folder
        if os.path.exists(ATTENDANCE_FILE):
            # Read the Excel file into a DataFrame
            df = pd.read_excel(ATTENDANCE_FILE)

            # Simulate a pop-up with the attendance data
            with st.expander("Attendance Records"):
                st.dataframe(df)  # Display the content of the attendance file
        else:
            st.error(f"Attendance file '{ATTENDANCE_FILE}' not found!")


if __name__ == "__main__":
    app()
