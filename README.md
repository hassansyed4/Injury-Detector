# Injury Detector (Localhost Prototype)

A simple **Gradio + Transformers** prototype that runs on **localhost**.  
Upload an image of a wound/injury and the app will:

- Detect the main injury/object region using **DETR panoptic segmentation**
- Draw a bounding box around the detected region
- Estimate width/height/depth/area (rough estimates)
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
