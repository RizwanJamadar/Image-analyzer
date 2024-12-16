import os
from ultralytics import YOLO
import cv2
from PIL import Image
import easyocr
import numpy as np
import json

# Hardcoded image path
image_path = "C:\\Users\\acer\\Desktop\\Image Analyzer\\og_images\\IMG_3021.jpg" 

# Check if the image file exists
if not os.path.isfile(image_path):
    print(f"Error: {image_path} does not exist.")
    exit(1)

# Create a reader to do OCR.
reader = easyocr.Reader(['en'])  # specify the language

# Load a pretrained YOLOv8n model
model = YOLO("sfit_id.pt")

# Run inference on the source
results = model(source=[image_path], conf=0.5)

# Load the image using OpenCV
image = cv2.imread(image_path)

# Create a list to store the extracted text
extracted_text = []
labels = results[0].names

for result in results:
    # Ensure proper handling of the bounding box data
    boxes = result.boxes  # YOLO returns object with .boxes property
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy()  # Extract coordinates
        class_index = int(box.cls[0].cpu().numpy())  # Class index
        conf = float(box.conf[0].cpu().numpy())  # Confidence score
        
        # Crop the region of interest (ROI) from the image
        roi = image[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]

        if class_index == 4:  # Example for a specific class, adjust as needed
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            border_size = 10  # Adjust this value as needed
            roi_np = roi[border_size:-border_size, border_size:-border_size]
            roi_pil = Image.fromarray(roi_np)
            roi_pil.save('profile-pic.jpg')  # Save profile picture
        else:
            roi_pil = Image.fromarray(roi)
            roi_np = np.array(roi_pil)
            
            # Use EasyOCR to extract text from ROI
            result = reader.readtext(roi_np)

            for detection in result:
                text = detection[1]  # Extract detected text
                extracted_text.append({labels[class_index]: text.upper()})

# Optionally, save extracted text to a file
with open('extracted_text.json', 'w') as f:
    json.dump(extracted_text, f)

print("Extraction complete. Results saved to 'extracted_text.json'.")
