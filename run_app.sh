#!/usr/bin/env bash

# Use osascript to open a new Terminal window and Start the Flask server
echo "Opening new Terminal window for Flask Server..."
osascript <<EOF
tell application "Terminal"
    activate
    do script "cd \"$(pwd)\" && source venv/bin/activate && python3 ws_flask_app.py"
end tell
EOF

echo "Waiting 5 seconds for the Flask server to initialize..."
sleep 5

# Use osascript to open a new Terminal window and run the main script
echo "Opening new Terminal window for main_script.py..."
osascript <<EOF
tell application "Terminal"
    activate
    do script "cd \"$(pwd)\" && source venv/bin/activate && python3 v2_model2_detect_stream_video.py"
end tell
EOF

