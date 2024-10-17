# train.py
from onnx import save_model
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from swinnUNet import SwinUNet  # Import your Swin Transformer-based UNet model here
from datapreprocess import CocoDataset, Preprocess, collate_fn  # Import preprocessing functions
from pycocotools.coco import COCO
# Initialize model parameters
ch = 3   # RGB image input (channels)
C = 96   # Base channels for Swin Transformer
num_class = 5  # Number of segmentation classes (e.g., binary segmentation)

# Instantiate the model
model = SwinUNet(ch, C, num_class)

# Move model to device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Define loss function and optimizer
criterion = CrossEntropyLoss()  # CrossEntropyLoss for segmentation masks
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # Adam optimizer with learning rate 0.0001

# Define training loop
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(num_epochs):
        for images, masks in train_loader:
            # Move images to the device
            images = images.to(device)
            
            # Convert masks list to a tensor by padding them to the same size
            max_num_boxes = max([mask.shape[0] for mask in masks])
            
            padded_masks = []
            for mask in masks:
                padded_mask = torch.zeros((max_num_boxes, 4), dtype=torch.float32)
                padded_mask[:mask.shape[0], :] = mask
                padded_masks.append(padded_mask)
            
            masks = torch.stack(padded_masks).to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

# Create the dataset and data loader
coco_path = 'archive/coco2017/annotations/instances_train2017.json'  # Path to COCO annotations
images_path = 'archive/coco2017/train2017/'  # Path to COCO images
coco = COCO(coco_path)
# Instantiate the dataset with preprocessing (resizing to 256x256)
train_dataset = CocoDataset(coco=coco, transform=Preprocess(size=(256, 256)))  # Add preprocessing transform
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)  # Use custom collate_fn for batching

# Call the training function
train_model(model, train_loader, criterion, optimizer, num_epochs=10)

save_model(model)