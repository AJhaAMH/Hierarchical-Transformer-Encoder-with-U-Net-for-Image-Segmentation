import cv2
import torch
import numpy as np
from pycocotools.coco import COCO
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Load the COCO dataset
coco_path = 'archive/coco2017/annotations/instances_train2017.json'  # Change this to your path
images_path = 'archive/coco2017/train2017/'  # Change this to your path
coco = COCO(coco_path)

class Preprocess:
    def __init__(self, size=(256, 256)):
        self.size = size
        
    def __call__(self, image, annotations):
        # Resize the image
        image = cv2.resize(image, self.size)
        
        # Normalize the image
        image = image / 255.0  # Normalize to [0, 1]
        
        # Prepare bounding boxes
        boxes = []
        for ann in annotations:
            bbox = ann['bbox']  # [x, y, width, height]
            x1, y1 = bbox[0], bbox[1]
            x2, y2 = bbox[0] + bbox[2], bbox[1] + bbox[3]
            boxes.append([x1, y1, x2, y2])
        
        return image, np.array(boxes)

class CocoDataset(Dataset):
    def __init__(self, coco, transform=None):
        self.coco = coco
        self.image_ids = coco.getImgIds()
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_data = self.coco.loadImgs(image_id)[0]
        image_path = images_path + image_data['file_name']

        # Load the image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Load annotations
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)
        boxes = np.array([[ann['bbox'][0], ann['bbox'][1],
                           ann['bbox'][0] + ann['bbox'][2],  # x_max = x_min + width
                           ann['bbox'][1] + ann['bbox'][3]]   # y_max = y_min + height
                          for ann in annotations])
        # Apply preprocessing
        if self.transform:
            image, boxes = self.transform(image, annotations)
        else:
            height, width = image.shape[:2]
            boxes[:, [0, 2]] /= width   # Normalize x_min, x_max by width
            boxes[:, [1, 3]] /= height
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return image, boxes

# Padding function
def pad_boxes(boxes, max_num_boxes):
    """ Pad boxes to a fixed size with zeros. """
    padded_boxes = np.zeros((max_num_boxes, 4), dtype=np.float32)
    num_boxes = boxes.shape[0]
    padded_boxes[:num_boxes] = boxes
    return padded_boxes

# Create a DataLoader
transform = Preprocess(size=(256, 256))  # Set the desired size
dataset = CocoDataset(coco, transform=transform)

# Update DataLoader with collate function
def collate_fn(batch):
    images, boxes = zip(*batch)  # Unzipping images and boxes

    # Find the maximum size in the batch
    max_height = max(image.shape[1] for image in images)
    max_width = max(image.shape[2] for image in images)

    # Find the maximum number of boxes in the batch
    max_num_boxes = max(len(box) for box in boxes)

    # Pad images to the maximum size
    padded_images = []
    for image in images:
        # Convert PyTorch dtype to NumPy dtype for padding
        np_dtype = np.float32 if image.dtype == torch.float32 else np.uint8
        padded_image = np.zeros((image.shape[0], max_height, max_width), dtype=np_dtype)
        padded_image[:, :image.shape[1], :image.shape[2]] = image.numpy()  # Convert PyTorch tensor to NumPy array
        padded_images.append(padded_image)

    # Pad boxes to the maximum number of boxes
    padded_boxes = []
    for box in boxes:
        if len(box) > 0:
            padded_box = np.zeros((max_num_boxes, 4), dtype=np.float32)  # Allocate space for all boxes
            padded_box[:len(box), :] = box  # Copy existing box data
        else:
            padded_box = np.zeros((max_num_boxes, 4), dtype=np.float32)  # No box, but we pad with zeroes
        padded_boxes.append(padded_box)

    # Convert padded images and boxes to tensors
    return torch.tensor(np.array(padded_images), dtype=torch.float32), torch.tensor(np.array(padded_boxes), dtype=torch.float32)




data_loader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

# Example to iterate through the DataLoader
for images, boxes in data_loader:
    if len(boxes) > 0:
        max_num_boxes = max(box.size(0) for box in boxes if box.dim() == 2)  # Find the maximum number of boxes, ignoring 1D tensors
        padded_boxes = []
        
        for box in boxes:
            # Debugging: Print the current box and its shape
            print(f'Current box: {box}, shape: {box.shape}, dim: {box.dim()}')

            # Check if box is a 2D tensor
            if box.dim() == 2:  # 2D tensor
                pad_size = max_num_boxes - box.size(0)
                if pad_size > 0:
                    # Pad with zeros
                    padded_box = torch.cat([box, torch.zeros(pad_size, box.size(1))], dim=0)  
                else:
                    padded_box = box
            elif box.dim() == 1:  # 1D tensor (e.g., empty or single box)
                # Handle 1D tensor case
                if box.numel() == 0:  # If it's empty, pad it accordingly
                    padded_box = torch.zeros(max_num_boxes, 4)  # Create a tensor filled with zeros
                else:
                    # If it has only one box, reshape it to [1, 4]
                    padded_box = box.unsqueeze(0)  # Convert to 2D tensor
            
            else:
                # If box is not 1D or 2D, raise an error
                raise ValueError(f'Expected 2D or 1D tensor for boxes, got {box.dim()}D tensor.')

            padded_boxes.append(padded_box)

        # Convert padded boxes to a single tensor
        boxes_tensor = torch.stack(padded_boxes)  # This should have shape (batch_size, max_num_boxes, 4)
        print(f'Batch of boxes shape: {boxes_tensor.shape}')  # Print the shape after stacking
    else:
        print('No boxes found in this batch.')
    # Here you can feed images and boxes to your model
    # model.train_on(images, boxes)  # Replace with your actual training code