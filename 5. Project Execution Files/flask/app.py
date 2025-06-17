import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for,session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__, static_folder='static')
app.secret_key = 'your_secret_key'
# Load the pre-trained model
model = load_model('cnn.h5')

# Configuration for uploads folder
UPLOAD_FOLDER = 'uploads/'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Routes
@app.route('/')
def index():
    """Render the home page."""
    prediction = session.get('prediction', None)  # Retrieve and clear the prediction from the session
    session.pop('prediction', None)
    return render_template('index.html',prediction=prediction)
    if 'prediction' in session:
        prediction = session['prediction']
    return render_template('index.html', prediction=prediction)

@app.route('/about')
def about():
    """Render the About page."""
    return render_template('about.html')

@app.route('/contact')
def contact():
    """Render the Contact page."""
    return render_template('contact.html')

@app.route('/inspect', methods=['GET', 'POST'])
def predict():
    """Handle file upload and predictions."""
    prediction = None
    image_path = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Save the file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            # Preprocess the image
            img = load_img(filepath, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            # Make prediction
            pred = np.argmax(model.predict(img_array), axis=1)
            classes = ['Normal', 'Ulcerative Colitis', 'Polyps', 'Esophagitis']
            prediction = classes[int(pred)]
            image_path = f"/{filepath}"  # To serve image in templates
            session['prediction'] = prediction
            return redirect(url_for('index'))

    return render_template('inspect.html', prediction=prediction, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
