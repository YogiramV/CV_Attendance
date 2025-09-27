import streamlit as st
import pandas as pd
from main import initialize_database, scan_photo, load_encodings
from image_encoder import encode_faces_once
from PIL import Image
import io
import os
import shutil
from datetime import datetime

ATTENDANCE_FILE = "attendance_data.xlsx"


def save_new_face(image, name, rollno):
    """Save a new face image to the Faces directory"""
    if not os.path.exists('Faces'):
        os.makedirs('Faces')

    filename = f"{name}_{rollno}.jpg"
    filepath = os.path.join('Faces', filename)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    image.save(filepath, 'JPEG')
    return filepath


def app():
    initialize_database()
    st.title("Face Recognition Attendance System")

    tab1, tab2 = st.tabs(["Mark Attendance", "Add New Face"])

    with tab1:
        if st.button("Encode Faces (Update after adding new faces)"):
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

            if os.path.exists("temp_uploaded_image.jpg"):
                os.remove("temp_uploaded_image.jpg")

        if st.button("Show Attendance Records"):
            if os.path.exists(ATTENDANCE_FILE):
                df = pd.read_excel(ATTENDANCE_FILE)

                with st.expander("Attendance Records"):
                    st.dataframe(df)
            else:
                st.error(f"Attendance file '{ATTENDANCE_FILE}' not found!")

    with tab2:
        st.subheader("Add New Face to Database")

        add_option = st.radio("How would you like to add a face?",
                              ["Take a photo", "Upload an image"])

        name = st.text_input("Enter person's name")
        rollno = st.text_input("Enter roll number")

        if add_option == "Take a photo":
            captured_face = st.camera_input(
                "Take a photo of the person's face")
            if captured_face and name and rollno:
                try:
                    img = Image.open(io.BytesIO(captured_face.getvalue()))
                    st.image(img, caption="Captured Face",
                             use_container_width=True)

                    if st.button("Save Face"):
                        filepath = save_new_face(img, name, rollno)
                        st.success(f"Face saved successfully as {filepath}")
                        st.info(
                            "Please click 'Encode Faces' in the 'Mark Attendance' tab to update the face encodings.")

                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")

        else:
            uploaded_file = st.file_uploader(
                "Choose an image...", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None and name and rollno:
                try:
                    img = Image.open(uploaded_file)
                    st.image(img, caption="Uploaded Face",
                             use_container_width=True)

                    if st.button("Save Face"):
                        filepath = save_new_face(img, name, rollno)
                        st.success(f"Face saved successfully as {filepath}")
                        st.info(
                            "Please click 'Encode Faces' in the 'Mark Attendance' tab to update the face encodings.")

                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")


if __name__ == "__main__":
    app()
