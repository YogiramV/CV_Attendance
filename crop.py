import cv2
import os

# Load the cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the image
image_path = 'Faces/group_photo.jpg'  # Change this to your image path
img = cv2.imread(image_path)

# Convert to grayscale for better detection
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

print(f"Found {len(faces)} face(s)")

# Create output directory for cropped faces
output_dir = 'cropped_faces'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Crop and save each face
for i, (x, y, w, h) in enumerate(faces):
    face = img[y:y + h, x:x + w]
    face_file_path = os.path.join(output_dir, f'face_{i+1}.jpg')
    cv2.imwrite(face_file_path, face)
    print(f"Face {i+1} saved at {face_file_path}")

# Optionally, display the image with rectangles drawn around detected faces
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow('Detected Faces', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
