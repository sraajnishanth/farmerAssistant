#MODEL_NAME = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
# MODEL_NAME = "ozair23/mobilenet_v2_1.0_224-finetuned-plantdisease"
# Replace "YOUR_API_KEY" with your actual WeatherAPI key
#weather_info = get_weather_weatherapi(latitude, longitude, api_key="98b9a5b299374075bff111840252102")
# openai.api_key = "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"

import cv2
import torch
from transformers import MobileNetV2ImageProcessor, AutoModelForImageClassification
from PIL import Image
import requests
import time
from openai import OpenAI

# -----------------------------
# Configuration
# -----------------------------
STREAM_URL = "http://100.100.255.133:8080/video"  # Replace with your phone's stream URL
# MODEL_NAME = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
MODEL_NAME = "ozair23/mobilenet_v2_1.0_224-finetuned-plantdisease"
TOP_K = 3  # Number of top predictions to consider per frame
CHANGE_THRESHOLD = 50  # Threshold for significant frame change

# -----------------------------
# Global Tracking Variables
# -----------------------------
disease_counts = {}         # Dictionary to hold cumulative counts of detections
previous_most_probable = None  # To track changes in most probable disease
final_recommendation = "Recommendation unavailable"


# -----------------------------
# OpenAI Client Configuration
# -----------------------------
openAiKey = ""

client = OpenAI(api_key=openAiKey)

def get_recommendation(disease, location, weather):
    """
    Sends the detected disease, location, and weather to OpenAI's chat endpoint
    and returns the recommended course of action.
    """
    messages = [
        {"role": "system", "content": "You are a plant pathology expert who provides concise treatment recommendations."},
        {"role": "user", "content": (
            f"Plant Disease Detection Report:\n"
            f"- Detected Disease: {disease}\n"
            f"- Location: {location}\n"
            f"- Weather: {weather}\n\n"
            "Based on the above information, provide the best course of action to counter this disease. "
            "Be concise and specific."
        )}
    ]
    
    try:
        print("Calling OpenAI API...")  # Log API call
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        recommendation = completion.choices[0].message.content.strip()
        print("OpenAI API call complete.")
        return recommendation
    except Exception as e:
        print("Error fetching recommendation:", e)
        return "Recommendation unavailable"

# -----------------------------
# Functions to Get Location and Weather using WeatherAPI
# -----------------------------
def get_location_from_ip():
    """
    Retrieves approximate latitude and longitude using ipinfo.io.
    """
    try:
        response = requests.get("https://ipinfo.io/json")
        if response.status_code == 200:
            data = response.json()
            loc = data.get("loc", None)
            if loc:
                lat_str, lon_str = loc.split(",")
                return float(lat_str), float(lon_str)
    except Exception as e:
        print("Error retrieving location:", e)
    return None, None

def get_weather_weatherapi(latitude, longitude, api_key="98b9a5b299374075bff111840252102"):
    """
    Retrieves current weather data from WeatherAPI.
    """
    try:
        url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={latitude},{longitude}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            current = data.get("current", {})
            temperature = current.get("temp_c", "N/A")
            condition = current.get("condition", {}).get("text", "N/A")
            return f"Temp: {temperature}°C, Condition: {condition}"
    except Exception as e:
        print("Error retrieving weather:", e)
    return "Weather data unavailable"


# -----------------------------
# Get Location and Weather Data
# -----------------------------
print("Retrieving location...")
latitude, longitude = get_location_from_ip()
if latitude is not None and longitude is not None:
    location_info = f"Lat: {latitude}, Lon: {longitude}"
    weather_info = get_weather_weatherapi(latitude, longitude, api_key="YOUR_WEATHERAPI_KEY")
else:
    location_info = "Location unavailable"
    weather_info = "Weather data unavailable"

print(location_info)
print(weather_info)

# -----------------------------
# Load the Model and Image Processor
# -----------------------------
print("Loading model and image processor...")
processor = MobileNetV2ImageProcessor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
model.eval()  # Set model to evaluation mode

# Debug: Print id2label mapping
print("id2label mapping:", model.config.id2label)

# -----------------------------
# Set Up Video Capture
# -----------------------------
cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    print(f"Error: Unable to open video stream at {STREAM_URL}")
    exit()
print("Video stream opened. Starting plant disease detection...")

# For detecting drastic frame changes, store the previous frame in grayscale
previous_gray = None

# -----------------------------
# Main Loop: Process Video Frames, Run Inference, and Update Recommendation when Most Probable Changes
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Warning: Failed to capture frame. Exiting loop.")
        break

    # Compute grayscale image for change detection
    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if previous_gray is not None:
        diff = cv2.absdiff(current_gray, previous_gray)
        mean_diff = diff.mean()
        if mean_diff > CHANGE_THRESHOLD:
            print(f"Drastic frame change detected (mean diff = {mean_diff:.2f}). Resetting detection counts.")
            disease_counts = {}  # Reset counts
    previous_gray = current_gray

    # Convert frame from BGR to RGB and to PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # Preprocess the image for the model
    inputs = processor(images=pil_image, return_tensors="pt")

    # Run inference (disable gradient computation for efficiency)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits

    # Get top K predictions
    topk = torch.topk(logits, k=TOP_K, dim=-1)
    topk_indices = topk.indices[0]

    # Convert indices to labels and update cumulative counts
    predicted_labels = []
    key_type = type(list(model.config.id2label.keys())[0])
    for idx in topk_indices:
        idx_val = idx.item()
        if key_type is int:
            label = model.config.id2label.get(idx_val, "Unknown")
        else:
            label = model.config.id2label.get(str(idx_val), "Unknown")
        predicted_labels.append(label)
        disease_counts[label] = disease_counts.get(label, 0) + 1

    # Determine the most probable disease based on cumulative counts
    if disease_counts:
        most_probable = max(disease_counts, key=disease_counts.get)
    else:
        most_probable = "Unknown"

    # If the most probable disease has changed, call the OpenAI API to update the recommendation.
    if most_probable != previous_most_probable and most_probable != "Unknown":
        final_recommendation = get_recommendation(most_probable, location_info, weather_info)
        previous_most_probable = most_probable

    # Compose overlay text for predictions and cumulative counts
    current_text = " | ".join(predicted_labels)
    tracked_text = "All Detections: " + ", ".join(f"{k}: {v}" for k, v in disease_counts.items())
    info_text = f"{location_info} | {weather_info}"

    # -----------------------------
    # Draw Prominent Overlay for Most Probable Disease
    # -----------------------------
    prominent_height = 80  # Height for prominent overlay
    prominent_overlay = frame.copy()
    cv2.rectangle(prominent_overlay, (0, 0), (frame.shape[1], prominent_height), (50, 50, 50), -1)
    cv2.addWeighted(prominent_overlay, 0.8, frame, 0.2, 0, frame)
    cv2.putText(frame, f"Most Probable: {most_probable}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 4, cv2.LINE_AA)

    # -----------------------------
    # Draw Info Overlay for Predictions and Location/Weather
    # -----------------------------
    info_y = prominent_height  # Start drawing info overlay below prominent overlay
    info_height = 70  # Height for info overlay
    info_overlay = frame.copy()
    cv2.rectangle(info_overlay, (0, info_y), (frame.shape[1], info_y + info_height), (0, 0, 0), -1)
    cv2.addWeighted(info_overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, f"Prediction: {current_text}", (10, info_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, tracked_text, (10, info_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, info_text, (10, info_y + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    # -----------------------------
    # Draw Recommendation Overlay
    # -----------------------------
    recommendation_y = info_y + info_height + 20
    recommendation_overlay = frame.copy()
    cv2.rectangle(recommendation_overlay, (0, recommendation_y), (frame.shape[1], recommendation_y + 100), (20, 20, 20), -1)
    cv2.addWeighted(recommendation_overlay, 0.8, frame, 0.2, 0, frame)
    cv2.putText(frame, "Recommendation:", (10, recommendation_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, final_recommendation, (10, recommendation_y + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    # -----------------------------
    # Display the Frame with Overlays
    # -----------------------------
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
