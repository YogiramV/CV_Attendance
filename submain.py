from flask import Flask, request, jsonify
from io import BytesIO
from app import process_face_recognition
from image_encoder import *

app = Flask(__name__)

initialized = False


def deploy_initialization():
    encode_faces_once()


if not initialized:
    deploy_initialization()
    initialized = True


@app.route('/submit', methods=['POST'])
def submit_data():
    try:
        print('connected successfully')

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
