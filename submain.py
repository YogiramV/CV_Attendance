from flask import Flask, request, jsonify
from io import BytesIO
from app import process_face_recognition
from image_encoder import encode_faces_once
import os

app = Flask(__name__)

initialized = False
ENCODED_FACES_TIMESTAMP_FILE = "encoded_faces_timestamp.txt"

def deploy_initialization():
    encode_faces_once()

def check_and_initialize_encoding():
    global initialized
    last_encoded_time = os.path.getmtime('Faces')

    if not os.path.exists(ENCODED_FACES_TIMESTAMP_FILE) or \
       last_encoded_time > float(open(ENCODED_FACES_TIMESTAMP_FILE).read()):
        deploy_initialization()
        with open(ENCODED_FACES_TIMESTAMP_FILE, 'w') as f:
            f.write(str(last_encoded_time))

if not initialized:
    check_and_initialize_encoding()
    initialized = True

@app.route('/submit', methods=['POST'])
def submit_data():
    try:
        print('Connected successfully')

        roll_no = request.form.get('roll_no')
        image_file = request.files.get('file')

        if not roll_no or not image_file:
            return jsonify({"error": "Missing roll number or image"}), 400

        result = process_face_recognition(image_file, roll_no)

        return jsonify({"result": result}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
