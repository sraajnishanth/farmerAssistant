import os
import requests
import time
import yaml
import cv2
import torch
import socketio
import openai
from datetime import datetime
from PIL import Image
from transformers import MobileNetV2ImageProcessor, AutoModelForImageClassification

# -----------------------------
# Load Config from config.yaml
# -----------------------------
try:
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    print("[ERROR] config.yaml not found. Please create it with the necessary keys.")
    exit(1)

openAiKey = config["openai"].get("api_key", "")
stream_url = config["video"].get("stream_url", "")
weather_api_key = config["weather"].get("api_key", "")

if not openAiKey or not stream_url:
    print("[ERROR] config.yaml missing 'openai.api_key' or 'video.stream_url'.")
    exit(1)

# -----------------------------
# Global placeholders for location & weather
# Will get updated in get_location_and_weather()
# -----------------------------
location_info = "Lat: 12.97, Lon: 77.59"
weather_info = "28°C, Cloudy"

# -----------------------------
# Socket.IO Client Setup
# -----------------------------
sio = socketio.Client()

@sio.event
def connect():
    log_event("[INFO] main_script connected to WebSocket server.")
    get_location_and_weather()
    # Now that location_info and weather_info are set, start the detection loop
    run_detection_loop()

@sio.event
def connect_error(e):
    log_event("[ERROR] Connection error: " + str(e))

@sio.event
def disconnect():
    log_event("[INFO] main_script disconnected from server.")

# -----------------------------
# Helper function to log events (print and emit)
# -----------------------------
def log_event(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    print(full_message)
    if sio.connected:
        sio.emit("log", {"message": full_message})
    else:
        print("[DEBUG] Not connected; skipping socket emit.")

# -----------------------------
# Weather and Location Functions
# -----------------------------
def get_location_from_ip():
    """
    Retrieves approximate latitude and longitude using ipinfo.io.
    """
    try:
        log_event("[LOG] Retrieving approximate latitude and longitude using ipinfo.io")
        response = requests.get("https://ipinfo.io/json")
        if response.status_code == 200:
            data = response.json()
            loc = data.get("loc", None)
            if loc:
                lat_str, lon_str = loc.split(",")
                return float(lat_str), float(lon_str)
    except Exception as e:
        print("Error retrieving location:", e)
        log_event("[LOG] Error retrieving location from ipinfo.io")
    return None, None

def get_weather_weatherapi(latitude, longitude, api_key):
    """
    Retrieves current weather data from WeatherAPI.
    """
    log_event(f"[LOG] Retrieving weather info for Lat: {latitude}, Lon: {longitude}")
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
        log_event("[LOG] Error retrieving weather from weatherapi.com")
    return "Weather data unavailable"

def get_location_and_weather():
    global location_info, weather_info
    lat, lon = get_location_from_ip()
    if lat is not None and lon is not None:
        location_info = f"Lat: {lat}, Lon: {lon}"
        weather_info = get_weather_weatherapi(lat, lon, api_key=weather_api_key)
        
        sio.emit("update_data", {
            "most_probable": '',
            "top_predictions": '',
            "location": location_info,
            "weather": weather_info,
            "recommendation": ''
        })
        log_event("[LOG] Real time Weather info for " + location_info + " : " + weather_info)
    else:
        location_info = "Lat: 12.97, Lon: 77.59"  # default
        weather_info = "28°C, Cloudy"            # default
        log_event("[LOG] Unable to retrieve real time weather. Defaulting to static values")

# -----------------------------
# OpenAI Setup using new API client
# -----------------------------
client = openai.OpenAI(api_key=openAiKey)

def get_openai_recommendation(disease, location, weather):
    """
    Calls the OpenAI API to get a multi-language HTML snippet with inline styles.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a plant pathology expert. Provide treatment recommendations for the detected disease in three languages: "
                "English, Hindi, and Kannada. Return your answer as a complete HTML snippet. "
                "Each language section must be enclosed in its own <div>, with a clear <h2> heading, and the recommendations "
                "should be formatted as a bullet list using <ul> and <li> tags. Do not include markdown code fences or backticks."
            )
        },
        {
            "role": "user",
            "content": (
                f"Plant Disease Detection Report:\n"
                f"- Detected Disease: {disease}\n"
                f"- Location: {location}\n"
                f"- Weather: {weather}\n\n"
                "Based on the above information, provide the best course of action to counter this disease for Indian Farmers. "
                "Your response must be in three sections: English, Hindi, and Kannada. "
                "Each heading (<h2>) and list (<ul>) should have inline styles for proper spacing."
            )
        }
    ]
    
    try:
        log_event("Calling OpenAI API for recommendations...")
        completion = client.chat.completions.create(
            model="gpt-4o",  # or "gpt-4", "gpt-3.5-turbo", etc.
            messages=messages
        )
        recommendation = completion.choices[0].message.content.strip()
        # Remove markdown code fences if present
        recommendation = recommendation.replace("```html", "").replace("```", "").strip()
        log_event("OpenAI API call complete.")
        return recommendation
    except Exception as e:
        print("Error fetching recommendation:", e)
        log_event("[LOG] Error fetching recommendation from OpenAI.")
        return "Recommendation unavailable"

# -----------------------------
# Detection and Emission Loop
# -----------------------------
def run_detection_loop():
    log_event("[INFO] Loading AI model...")
    MODEL_NAME = "linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification"
    processor = MobileNetV2ImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
    model.eval()

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        log_event(f"[ERROR] Unable to open video stream: {stream_url}")
        return

    log_event("[INFO] Video stream opened. Starting detection...")
    previous_gray = None
    disease_counts = {}
    previous_most_probable = None
    last_inference_time = time.time()
    last_recommendation = "No recommendation available"

    global location_info, weather_info  # Use updated location & weather

    while True:
        ret, frame = cap.read()
        if not ret:
            log_event("[WARNING] Failed to capture frame. Exiting loop.")
            break

        current_time = time.time()
        # Limit how often we run inference (e.g., every 2 seconds)
        if current_time - last_inference_time < 2.0:
            continue
        last_inference_time = current_time

        # Optional: scene-change detection if needed
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if previous_gray is not None:
            diff = cv2.absdiff(current_gray, previous_gray)
            mean_diff = diff.mean()
            if mean_diff > 50:
                log_event(f"[INFO] Drastic scene change detected (mean diff={mean_diff:.2f}). Resetting disease counts.")
                disease_counts.clear()
        previous_gray = current_gray

        # Convert frame to PIL Image for inference
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        inputs = processor(images=pil_image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits

        # Top-3 predictions
        TOP_K = 3
        topk = torch.topk(logits, k=TOP_K, dim=-1)
        topk_indices = topk.indices[0]

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

        most_probable = "Unknown"
        if disease_counts:
            most_probable = max(disease_counts, key=disease_counts.get)

        # If there's a newly detected disease, query OpenAI for a recommendation
        if most_probable != previous_most_probable and most_probable != "Unknown":
            log_event(f"[INFO] New probable disease detected: {most_probable}. Querying OpenAI for recommendation...")
            last_recommendation = get_openai_recommendation(most_probable, location_info, weather_info)
            previous_most_probable = most_probable

        # Prepare payload for Socket.IO
        payload = {
            "most_probable": most_probable,
            "top_predictions": predicted_labels,
            "location": location_info,
            "weather": weather_info,
            "recommendation": last_recommendation
        }
        # log_event("[INFO] Emitting data: " + str(payload))
        sio.emit("update_data", payload)

    cap.release()
    log_event("[INFO] Detection loop ended.")


# -----------------------------
# Main Entry Point
# -----------------------------
if __name__ == "__main__":
    server_url = "http://127.0.0.1:8000"
    log_event(f"[INFO] Connecting to WebSocket server at {server_url}...")
    try:
        sio.connect(server_url)
    except Exception as e:
        log_event(f"[ERROR] Could not connect to {server_url}: {e}")
        exit(1)

    # We do NOT call run_detection_loop() here anymore. It's called after connect() -> get_location_and_weather() -> run_detection_loop()

    log_event("[INFO] Waiting for events. Press Ctrl+C to quit.")
    sio.wait()
