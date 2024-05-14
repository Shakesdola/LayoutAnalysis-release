import cv2
import numpy as np
import os
from ultralytics import YOLO

# Path to the trained model weights
weights_path = 'results/15epoches/best.pt'

# Initialize the YOLOv8 model for inference
model = YOLO(weights_path)

# Path to the test image folder
test_images_folder = 'data/prediction'
# Output folder to store the overlayed masks
output_folder = 'pred_gen_masks'
os.makedirs(output_folder, exist_ok=True)

H = 960
W = 640

# List all files in the test images folder
test_image_paths = [os.path.join(test_images_folder, f) for f in os.listdir(test_images_folder) if os.path.isfile(os.path.join(test_images_folder, f))]

for test_image_path in test_image_paths:
    # Predict using YOLO model
    results = model.predict(test_image_path, save=True, imgsz=640, save_txt=True, conf=0.5)

    # Initialize an empty canvas to store the overlayed masks
    overlay = np.zeros((H, W, 3), dtype=np.uint8)

    for result in results:
        for i, mask in enumerate(result.masks.data):
            class_id = result.boxes.cls[i].item()  # Get class ID for each mask
            # Red for class 0, Yellow for class 1
            if class_id == 0:
                color = (0,0,255)
            else:
                color = (0,255,255)


            # Resize the mask to match the dimensions of the canvas
            mask = cv2.resize(mask.cpu().numpy(), (640, 960)).astype(np.uint8) * 255

            # Convert single-channel mask to 3-channel mask
            mask = cv2.merge([mask, mask, mask])

            # Apply the color to the mask
            mask = cv2.bitwise_and(mask, color)

            # Overlay the mask on the canvas
            overlay = cv2.addWeighted(overlay, 1, mask, 1, 0)





    # Save the overlayed mask as an image
    filename = os.path.splitext(os.path.basename(test_image_path))[0]
    cv2.imwrite(os.path.join(output_folder, f"{filename}.png"), overlay)
