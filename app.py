import gradio as gr
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import DetrForSegmentation, DetrFeatureExtractor

# Load model
model = DetrForSegmentation.from_pretrained("facebook/detr-resnet-50-panoptic")
feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50-panoptic")

def analyze_injury(image):
    image_pil = Image.fromarray(image)
    
    # Convert to model input
    inputs = feature_extractor(images=image_pil, return_tensors="pt")

    # Run model
    with torch.no_grad():
        outputs = model(**inputs)

    # Get masks
    masks = outputs.pred_masks[0].numpy()
    masks = (masks > 0.5).astype("uint8")

    # Combine masks to get the biggest object
    combined_mask = np.sum(masks, axis=0)
    combined_mask = (combined_mask > 0).astype("uint8")

    # Find contours for size
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return "No injury detected", image

    cnt = max(contours, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(cnt)
    # Pixel-to-cm conversion (simple estimate)
    px_per_cm = 37.8

    width_cm = w / px_per_cm
    height_cm = h / px_per_cm

    # Approx depth (assuming wound is circular)
    depth_cm = ( (w + h) / 4 ) / px_per_cm   # rough estimate

    # Area in cm²
    area_cm2 = (w / px_per_cm) * (h / px_per_cm)

    # Cut out the injury area for color analysis
    injury_area = image[y:y+h, x:x+w]

    # Color detection (average redness)
    avg_color = np.mean(injury_area, axis=(0,1))
    red, green, blue = avg_color

    # Prepare final image
    output = image.copy()
    cv2.rectangle(output, (x,y), (x+w, y+h), (255,0,0), 3)

    # Results
    result = f"""
    Injury Width: {w} px  (~{width_cm:.2f} cm)
    Injury Height: {h} px  (~{height_cm:.2f} cm)
    Estimated Depth: ~{depth_cm:.2f} cm
    Approx Area: {w*h} px²  (~{area_cm2:.2f} cm²)

    Average Red Value: {red:.2f}
    Average Green Value: {green:.2f}
    Average Blue Value: {blue:.2f}
    """

    return result, output

# Gradio UI
interface = gr.Interface(
    fn=analyze_injury,
    inputs=gr.Image(type="numpy"),
    outputs=[
        gr.Textbox(lines=10, max_lines=20, show_copy_button=True),
        gr.Image(type="numpy")
    ],
    title="Injury Detection Prototype",
    description="Upload an image of an injury/wound to detect size and color."
)

interface.launch()