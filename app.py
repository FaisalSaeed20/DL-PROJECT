import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw
import numpy as np
from io import BytesIO
from ultralytics import YOLO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


yolo_model = YOLO('best.pt')  #

mobilenet_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
mobilenet_model.classifier[1] = torch.nn.Linear(in_features=1280, out_features=2)
mobilenet_model.load_state_dict(torch.load('fire_smoke_model (1).pth', map_location=device))
mobilenet_model.eval()
mobilenet_model = mobilenet_model.to(device)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_image_mobilenet(image):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = mobilenet_model(image_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    return predicted.item(), confidence.item()


st.title("Fire and Smoke Detection")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)


    st.write("Running YOLO for object detection...")
    results = yolo_model(img)  

    cropped_images = []
    cropped_coords = []
    for result in results:
        boxes = result.boxes.xyxy  
        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box)
            cropped_image = img.crop((x_min, y_min, x_max, y_max))  
            cropped_images.append(cropped_image)
            cropped_coords.append((x_min, y_min, x_max, y_max))

    img_with_boxes = img.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    for (x_min, y_min, x_max, y_max) in cropped_coords:
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
    st.image(img_with_boxes, caption="YOLO Object Detection", use_column_width=True)


    st.write("Running MobileNet for classification (Fire or Smoke)...")
    predicted_class, confidence = predict_image_mobilenet(img)
    result = 'Fire' if predicted_class == 0 else 'Smoke'


    st.subheader("Prediction Result:")
    st.write(f"Prediction: {result}")
    st.write(f"Confidence: {(confidence * 100):.2f}%")
