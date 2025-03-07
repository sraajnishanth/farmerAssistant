from flask import Flask, render_template
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config["SECRET_KEY"] = "supersecretkey"

# Use gevent as the async mode
socketio = SocketIO(app, async_mode="gevent")

# (Optional) data store for detection data
latest_data = {
    "most_probable": None,
    "recommendation": None,
    "location": None,
    "weather": None
}

@app.route("/")
def index():
    """
    Renders your index.html from the templates/ folder.
    """
    return render_template("index.html")

@socketio.on("update_data")
def handle_update_data(data):
    print("[INFO] Received data via Socket.IO:", data)
    latest_data.update(data)
    # Broadcast new data to all connected clients
    socketio.emit("new_data", latest_data)

# New handler: receive log events from the main script and broadcast them.
@socketio.on("log")
def handle_log(data):
    print("[SERVER] Log received:", data)
    # Broadcast the log event to all connected clients
    socketio.emit("log", data)

if __name__ == "__main__":
    # debug=False to avoid kqueue conflicts on macOS with gevent
    socketio.run(app, debug=False, host="0.0.0.0", port=8000)
