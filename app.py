import streamlit as st
import pandas as pd
from main import initialize_database, scan_photo, load_encodings
from image_encoder import encode_faces_once
from PIL import Image
import io
import os

ATTENDANCE_FILE = "attendance_data.xlsx"


def app():
    initialize_database()
    st.title("Face Recognition Attendance System")

    if st.button("Encode Faces (One-time Process)"):
        encode_faces_once()
        st.success("Faces have been successfully encoded and saved!")

    known_face_encodings, known_face_rollnos, known_face_names = load_encodings()

    st.subheader("Capture Photo for Recognition")
    rollno_input = st.text_input("Enter your Roll Number")

    captured_image = st.camera_input("Take a photo")

    if captured_image and rollno_input:
        image_bytes = captured_image.getvalue()

        img = Image.open(io.BytesIO(image_bytes))

        st.image(img, caption="Captured Image", use_container_width=True)

        img.save("temp_uploaded_image.jpg")

        result = scan_photo("temp_uploaded_image.jpg", rollno_input,
                            known_face_encodings, known_face_rollnos, known_face_names)

        st.write(result)

    if st.button("Show Attendance Records"):
        if os.path.exists(ATTENDANCE_FILE):
            df = pd.read_excel(ATTENDANCE_FILE)

            with st.expander("Attendance Records"):
                st.dataframe(df)
        else:
            st.error(f"Attendance file '{ATTENDANCE_FILE}' not found!")


if __name__ == "__main__":
    app()
