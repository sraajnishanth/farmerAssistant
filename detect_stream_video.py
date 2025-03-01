import cv2
import torch
from transformers import MobileNetV2ImageProcessor, AutoModelForImageClassification
from PIL import Image

# -----------------------------
# Configuration
# -----------------------------
STREAM_URL = "http://192.0.0.4:8080/video"  # Replace with your phone's stream URL
MODEL_NAME = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"

# -----------------------------
# Load the model and image processor
# -----------------------------
print("Loading model and image processor...")
processor = MobileNetV2ImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.eval()  # Set model to evaluation mode

# Debug: Print the id2label mapping to see its format
print("id2label mapping:", model.config.id2label)

# -----------------------------
# Set up video capture
# -----------------------------
cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    print(f"Error: Unable to open video stream at {STREAM_URL}")
    exit()
print("Video stream opened. Starting plant disease detection...")

# -----------------------------
# Main loop: Process video frames
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Warning: Failed to capture frame. Exiting loop.")
        break

    # Convert the frame from BGR (OpenCV default) to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert to a PIL Image for processing
    pil_image = Image.fromarray(frame_rgb)

    # Preprocess the image for the model
    inputs = processor(images=pil_image, return_tensors="pt")

    # Run inference (disable gradient calculation for efficiency)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits

    # Get the predicted index
    predicted_idx = logits.argmax(-1).item()

    # Debug: Print the predicted index
    print("Predicted index:", predicted_idx)

    # Determine if the id2label mapping uses string or integer keys
    key_type = type(list(model.config.id2label.keys())[0])
    if key_type is int:
        predicted_label = model.config.id2label.get(predicted_idx, "Unknown")
    else:
        predicted_label = model.config.id2label.get(str(predicted_idx), "Unknown")

    # Debug: Print the predicted label
    print("Predicted label:", predicted_label)

    # Overlay the prediction on the frame
    cv2.putText(frame, f"Prediction: {predicted_label}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with the overlay
    cv2.imshow("Plant Disease Detection", frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()
print("Video stream closed. Exiting program.")
