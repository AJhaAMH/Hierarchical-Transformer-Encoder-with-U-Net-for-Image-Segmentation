import os
from flask import Flask, request, render_template, redirect, url_for
import torch
import cv2
import numpy as np
from swinnUNet import SwinUNet  # Import your model

# Flask app setup
app = Flask(__name__)

# Set upload and output folder paths
UPLOAD_FOLDER = 'static/uploads/'
OUTPUT_FOLDER = 'static/outputs/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
channels = 3  # Assuming RGB input
num_classes = 91  # Adjust based on your dataset
model = SwinUNet(channels, num_classes)
model.load_state_dict(torch.load('trained_model.pth', map_location=device))
model.eval().to(device)

# Preprocess image function
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))  # Adjust size based on model input
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # [Batch, Channels, H, W]
    image = image.float() / 255.0
    return image.to(device)

# Predict function
def predict(image_path):
    input_image = preprocess_image(image_path)
    with torch.no_grad():
        output = model(input_image)
    predicted_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    return predicted_mask

# Endpoint for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint to handle image upload and prediction
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Perform segmentation
        predicted_mask = predict(filepath)

        # Save the segmented output
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'segmented_{filename}')
        cv2.imwrite(output_path, predicted_mask * 255)  # Scale mask to 0-255 for visualization

        return render_template('result.html', input_image=filepath, output_image=output_path)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
