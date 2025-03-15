import cv2
import numpy as np
import joblib
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
from werkzeug.utils import secure_filename
import base64
from datetime import datetime  # Ensure this is at the top of your file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize video capture
cap = cv2.VideoCapture(0)  # 0 for default webcam

# Flask app
app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Required for flashing messages

# Folder to save captured images and employee data
UPLOAD_FOLDER = 'static/uploads'
EMPLOYEE_FOLDER = 'static/employees'
DATASET_FOLDER = 'static/dataset'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(EMPLOYEE_FOLDER):
    os.makedirs(EMPLOYEE_FOLDER)
if not os.path.exists(DATASET_FOLDER):
    os.makedirs(DATASET_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['EMPLOYEE_FOLDER'] = EMPLOYEE_FOLDER
app.config['DATASET_FOLDER'] = DATASET_FOLDER

# Load or initialize the model
MODEL_PATH = 'face_recognition_model.pkl'
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = SVC(kernel='linear', probability=True)

# Function to perform face recognition
def recognize_face(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    access_status = "No Face Detected"

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (100, 100))
        face_flattened = face_resized.reshape(1, -1)
        predicted_label = model.predict(face_flattened)
        confidence_scores = model.decision_function(face_flattened)
        confidence = np.max(confidence_scores)

        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        label_text = f"{predicted_label[0]} ({confidence:.2f})"
        cv2.putText(img, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        if confidence > 50:
            access_status = "Access Granted"
        else:
            access_status = "Access Denied"

    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'processed_image.jpg')
    cv2.imwrite(output_path, img)
    return access_status

# Function to prepare the dataset
def prepare_dataset():
    faces = []
    labels = []

    for employee_name in os.listdir(app.config['DATASET_FOLDER']):
        employee_folder = os.path.join(app.config['DATASET_FOLDER'], employee_name)
        if os.path.isdir(employee_folder):
            for image_name in os.listdir(employee_folder):
                image_path = os.path.join(employee_folder, image_name)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                img_resized = cv2.resize(img, (100, 100))
                faces.append(img_resized.flatten())
                labels.append(employee_name)

    return np.array(faces), np.array(labels)

# Function to retrain the model
def retrain_model():
    faces, labels = prepare_dataset()
    if len(faces) == 0:
        return False

    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(faces, labels_encoded, test_size=0.2, random_state=42)

    # Train the model
    global model
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)

    # Save the model
    joblib.dump(model, MODEL_PATH)
    return True

# Route for the home page
@app.route('/')
def index():
    # Pass the current date/time to the template
    return render_template('index.html', now=datetime.now())  # Correct: Call the function

# Route to capture an image
@app.route('/capture', methods=['POST'])
def capture():
    data = request.get_json()
    image_data = data.get('image')

    if image_data:
        try:
            # Convert the data URL to an image file
            header, encoded = image_data.split(",", 1)
            binary_data = base64.b64decode(encoded)

            # Save the captured image
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'captured_image.jpg')
            with open(image_path, 'wb') as f:
                f.write(binary_data)

            # Perform face recognition
            access_status = recognize_face(image_path)

            return jsonify({
                'status': 'success',
                'access_status': access_status,
            })
        except Exception as e:
            print(f"Error processing image: {e}")
            return jsonify({
                'status': 'error',
                'message': 'Failed to process image.',
            })
    else:
        return jsonify({
            'status': 'error',
            'message': 'No image data received.',
        })

# Route to display the result
@app.route('/result')
def result():
    access_status = request.args.get('status', 'Unknown')
    # Pass the current date/time to the template
    return render_template('result.html', status=access_status, now=datetime.now())  # Correct: Call the function

# Route to register employees
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        employee_name = request.form.get('name')
        image_data = request.form.get('image')

        if employee_name and image_data:
            try:
                # Convert the data URL to an image file
                header, encoded = image_data.split(",", 1)
                binary_data = base64.b64decode(encoded)

                # Create a secure filename with timestamp
                filename = secure_filename(f"{employee_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg")
                image_path = os.path.join(app.config['EMPLOYEE_FOLDER'], filename)

                # Save the image
                with open(image_path, 'wb') as f:
                    f.write(binary_data)

                # Add the image to the dataset
                employee_folder = os.path.join(app.config['DATASET_FOLDER'], employee_name)
                if not os.path.exists(employee_folder):
                    os.makedirs(employee_folder)
                dataset_image_path = os.path.join(employee_folder, filename)
                with open(dataset_image_path, 'wb') as f:
                    f.write(binary_data)

                # Retrain the model
                if retrain_model():
                    flash(f"Employee {employee_name} registered successfully and model retrained!", "success")
                else:
                    flash(f"Employee {employee_name} registered successfully, but model retraining failed.", "warning")
            except Exception as e:
                print(f"Error saving image: {e}")
                flash("Failed to save the image. Please try again.", "error")
        else:
            flash("Please provide a name and capture an image.", "error")

    # Pass the current date/time to the template
    return render_template('register.html', now=datetime.now())  # Correct: Call the function

# Route to manage employees
@app.route('/employees')
def employees():
    employee_images = os.listdir(app.config['EMPLOYEE_FOLDER'])
    # Pass the current date/time to the template
    return render_template('employees.html', employees=employee_images, now=datetime.now())  # Correct: Call the function

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)