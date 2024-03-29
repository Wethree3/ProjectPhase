from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Set the path where uploaded images will be stored temporarily
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
model = load_model('model2.h5')

# Manually create the class indices mapping
class_indices = {
    0: 'Adenosis',
    1: 'Ductal_carcinoma',
    2: 'Fibroadenoma',
    3: 'Lobular_carcinoma',
    4: 'Mucinous_carcinoma',
    5: 'Papillary_carcinoma',
    6: 'Phyllodes_tumor',
    7: 'Tubular_adenoma',
}

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Define the home route
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            error_message = 'No file part'
            return render_template('index.html', error_message=error_message)

        file = request.files['file']

        # If the user does not select a file, submit an empty part without filename
        if file.filename == '':
            error_message = 'No selected file'
            return render_template('index.html', error_message=error_message)

        # Check if the file is allowed
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess the image
            processed_img = preprocess_image(filepath)

            # Make prediction
            predictions = model.predict(processed_img)

            # Get the predicted label
            predicted_label = np.argmax(predictions)

            # Map the predicted label to the actual class name using the loaded class indices
            class_name = class_indices[predicted_label]
            print(filepath)
            print(filename)
            filepath = "uploads/"+filename

            # Pass the result and the full path of the uploaded image to the template
            result = {"predicted_label": class_name, "image_path": '/' + filepath}  # Add a leading slash
            return render_template('index.html', result=result)

    return render_template('index.html')

# Function to check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

if __name__ == '__main__':
    app.run(debug=True,port=5300)