import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import speech_recognition as sr
import threading
from flask import Flask, render_template, Response

app = Flask(__name__)

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

def say(text):
    engine.say(text)
    engine.runAndWait()

# Function to detect red clothes
def is_wearing_red(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    red_pixels = cv2.countNonZero(red_mask)
    total_pixels = frame.shape[0] * frame.shape[1]
    return (red_pixels / total_pixels) > 0.05

# Speech recognition thread
def listen_and_respond():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    while True:
        with mic as source:
            print("Listening...")
            recognizer.adjust_for_ambient_noise(source)
            audio = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio).lower()
            print(f"User said: {command}")
            if "hello" in command:
                say("Hello! How can I help you?")
            elif "color" in command:
                say("Please stand in front of the camera.")
            elif "exit" in command:
                say("Goodbye!")
                break
        except Exception:
            pass

listener_thread = threading.Thread(target=listen_and_respond, daemon=True)
listener_thread.start()

# Camera generator for HTML web streaming
def gen_frames():
    cap = cv2.VideoCapture(0)
    said_red = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if is_wearing_red(frame):
            if not said_red:
                # Threading is used here to avoid freezing the camera view while speaking
                threading.Thread(target=say, args=("Your color is red",), daemon=True).start()
                said_red = True
            cv2.putText(frame, "Red color detected!", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            said_red = False

        # Convert image to bytes format for HTML browser rendering
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    # Looks for index.html inside templates folder
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
            
