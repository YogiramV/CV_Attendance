from flask import Flask, request, jsonify
from io import BytesIO
from app import process_face_recognition  # Import the refactored Streamlit function

app = Flask(__name__)

@app.route('/submit', methods=['POST'])
def submit_data():
    try:
        print('connected successfully')
        # Get the roll number from form data
        roll_no = request.form.get('roll_no')

        # Get the image from form data
        image_file = request.files.get('file')

        if not roll_no or not image_file:
            return jsonify({"error": "Missing roll number or image"}), 400

        # Pass the image and roll number to the face recognition function
        result = process_face_recognition(image_file, roll_no)

        # Return the result (you can structure this result as needed)
        return jsonify({"result": result}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
