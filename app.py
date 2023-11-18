import streamlit as st
import requests
import io
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
  

class_name = ["Flute", "Guitar", "Drum", "Piano", "Saxophone", "Trumpet", "Watercraft", "Rose", "Houseplant",
           "Flowerpot", "Baseball bat", "Ball", "Tennis racket", "Tennis ball", "Table tennis racket", "Cricket ball",
           "Bicycle helmet", "Helmet", "Football helmet", "Book", "Shirt", "Jacket", "Footwear", "Watch", "Handbag",
           "Luggage and bags", "Sandal", "Necklace", "Table", "Chair", "Couch", "Wall clock", "Lamp", "Sofa bed", "Bed",
           "Stool", "Cat", "Dog", "Mouse", "Car", "Motorcycle"
           ]

model_path = 'models/best.pt'
model = YOLO(model_path)


def detect_objects(img):
    img_copy = img.copy()  
    img_height, img_width, _ = img.shape
    results = model.predict(img_copy, device='cpu')
    
    for result in results:
        boxes = result.boxes.xyxy
        conf = result.boxes.conf
        cls = result.boxes.cls

        boxes = boxes.tolist()

        
        for box, conf_val, cls_val in zip(boxes, conf, cls):
            x_min = int(round(box[0], 2))
            y_min = int(round(box[1], 2))
            x_max = int(round(box[2], 2))
            y_max = int(round(box[3], 2))

            st.success(f"Class Name: {class_name[int(cls_val)]} - Confidence Score: {round(conf_val.item(), 2)} - Bboxes: {x_min, y_min, x_max, y_max}")
            # cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            # label = f"{class_name[int(cls_val)]} ({round(conf_val.item(), 2)})"
            # cv2.putText(img_copy, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img_copy


def main():
    st.title("Object Detection App")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

        if st.button("Detect Objects"):
            img_response = detect_objects(image)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax1.set_title("Uploaded Image")
            ax1.axis("off")

            ax2.imshow(cv2.cvtColor(img_response, cv2.COLOR_BGR2RGB))
            ax2.set_title("Detected Objects")
            ax2.axis("off")

            st.pyplot(fig)
            
if __name__ == "__main__":
    main()
