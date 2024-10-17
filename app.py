from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import torch
import torchvision.transforms as transforms
import os
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.detection import maskrcnn_resnet50_fpn

# Load the pretrained Mask R-CNN model

app = Flask(__name__)

# Load the pretrained DeepLabV3 model for semantic segmentation
#model = maskrcnn_resnet50_fpn(pretrained=True)
model = 'trained_model.pth'  # Update this path to your trained model
model.load_state_dict(torch.load(model_path))
model.eval()  # Set the model to evaluation mode

# Save the model's state_dict to a .pth file
model_path = 'deeplabv3_resnet101.pth'
torch.save(model.state_dict(), model_path)
print(f"Model saved as {model_path}")

# Define image transformations
def transform_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize image
        transforms.ToTensor(),          # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
    ])
    return preprocess(image).unsqueeze(0)  # Add batch dimension

# Process and generate a segmented image
def get_segmented_image(image):
    image_tensor = transform_image(image)
    with torch.no_grad():
        output = model(image_tensor)['out']  # Get model output (batch_size, num_classes, H, W)
    
    # Get the class prediction for each pixel
    prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()  # (H, W)
    
    # Convert to image (optional: map class indices to colors for visualization)
    segmented_image = Image.fromarray((prediction * (255 // output.shape[1])).astype('uint8'))  # (H, W)
    
    return segmented_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image = Image.open(file.stream)  # Open the uploaded image
            segmented_image = get_segmented_image(image)  # Get segmented image
            
            # Save the uploaded and segmented images to static directory
            uploaded_image_path = os.path.join('static', 'uploads', 'uploaded_image.png')
            segmented_image_path = os.path.join('static', 'uploads', 'segmented_image.png')

            # Ensure the directory exists
            os.makedirs(os.path.dirname(uploaded_image_path), exist_ok=True)

            # Save images
            image.save(uploaded_image_path)
            segmented_image.save(segmented_image_path)

            return render_template('result.html', uploaded_image=uploaded_image_path, segmented_image=segmented_image_path)

if __name__ == '__main__':
    app.run(debug=True)
