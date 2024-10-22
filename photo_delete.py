import cv2
import os
import numpy as np

# Load the pre-trained deep learning face detector model from OpenCV
prototxt_path = 'c:/Users/yusuf/Desktop/CV_Attendance/deploy.prototxt'
model_path = 'c:/Users/yusuf/Desktop/CV_Attendance/res10_300x300_ssd_iter_140000_fp16.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Directory containing images
image_folder = 'c:/Users/yusuf/Desktop/CV_Attendance/cropped_faces'  # Replace with your folder path

# Check if the folder exists
if not os.path.exists(image_folder):
    print(f"Error: Directory '{image_folder}' does not exist")
    exit()

# Loop through all files in the directory
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)

    # Check if it is an image file (optional)
    if not image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        continue

    # Load the image
    img = cv2.imread(image_path)

    if img is None:
        print(f"Error: Could not load image '{image_name}'")
        continue

    # Get dimensions of the image
    (h, w) = img.shape[:2]

    # Convert the image to a blob for input to the network
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # Set the input to the pre-trained network
    net.setInput(blob)

    # Perform face detection
    detections = net.forward()

    # Initialize face count
    face_count = 0

    # Iterate through detections and apply filters
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        # Filter out weak detections by requiring confidence > 0.7
        if confidence > 0.7:  # Adjust confidence threshold if necessary
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Consider the face only if it occupies a minimum size
            face_width = endX - startX
            face_height = endY - startY
            min_face_size = 30  # Minimum size of the face in pixels (adjust as needed)

            if face_width > min_face_size and face_height > min_face_size:
                face_count += 1

    # If no faces are detected, delete the image
    if face_count == 0:
        print(f"'{image_name}' does not contain any faces. Deleting the image.")
        os.remove(image_path)
    else:
        print(f"'{image_name}' contains {face_count} face(s). Keeping the image.")
