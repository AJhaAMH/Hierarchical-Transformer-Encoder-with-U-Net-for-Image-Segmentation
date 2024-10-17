import os
import random  # Ensure you import the random module
import cv2
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

# Path to your COCO annotations and images
annotation_file = r'C:\Users\amang\OneDrive\Desktop\My Work\BTech CSE\SELF\SEMESTER 7\Image Seg Project\archive\coco2017\annotations\instances_train2017.json'
coco = COCO(annotation_file)

# Get all image IDs
image_ids = coco.getImgIds()

# Select a random image ID
random_image_id = random.choice(image_ids)

# Load the image metadata
image_info = coco.loadImgs(random_image_id)[0]

# Construct the image path
image_path = os.path.join(r'C:\Users\amang\OneDrive\Desktop\My Work\BTech CSE\SELF\SEMESTER 7\Image Seg Project\archive\coco2017\train2017', image_info['file_name'])

# Load the image using OpenCV
image = cv2.imread(image_path)

if image is None:
    print(f"Failed to load image at {image_path}")
else:
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')  # Hide axes
    plt.show()
# Get annotations for the image (e.g., bounding boxes, keypoints)
annotation_ids = coco.getAnnIds(imgIds=image_info['id'])
annotations = coco.loadAnns(annotation_ids)
print(annotations)
for annotation in annotations:
    bbox = annotation['bbox']  # Bounding box in [x, y, width, height]
    x, y, width, height = map(int, bbox)
    cv2.rectangle(image, (x, y), (x + width, y + height), (255, 0, 0), 2)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
