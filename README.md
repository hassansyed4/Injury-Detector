## 1) Injury Detector (Localhost Prototype)

A simple **Gradio + Transformers** prototype that runs on **localhost**.  
Upload an image of a wound/injury and the app will:

- Detect the main injury/object region using **DETR panoptic segmentation**
- Draw a bounding box around the detected region
- Es timate width/height/depth/area (rough estimates)
- Calculate average RGB color values in the detected region

> Note: This is a prototype and **not a medical tool**. Measurements are approximate and depend heavily on image distance, lighting, and camera resolution.

---

## Features

- Local web UI using **Gradio**
- Segmentation using `facebook/detr-resnet-50-panoptic`
- Bounding box extraction using OpenCV contours
- Basic size/area estimates using a fixed conversion factor (`px_per_cm = 37.8`)
- Color analysis (average RGB)

---

## Tech Stack

- Python
- Gradio
- OpenCV
- NumPy
- PyTorch
- Hugging Face Transformers
- PIL

---

## 2) Install dependencies
```bash
pip install --upgrade pip
pip install gradio opencv-python numpy pillow torch transformers
```

## 3) Save your code
Create a file named app.py and paste your code into it.

## 4) Run the code
```bash
python app.py
```

---
How It Works (High Level)

The uploaded image is converted to a PIL image.

The DETR panoptic model generates segmentation masks.

Masks are combined to find the biggest detected object region.

OpenCV finds contours and creates a bounding rectangle.

Size is estimated using a fixed pixel-to-cm ratio.

Average RGB values are calculated for the detected region.

The UI returns:

Text measurements + RGB values

Image with a drawn bounding box











