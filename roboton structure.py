import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import speech_recognition as sr
import threading

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

# Function for speech recognition listening
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
            elif "exit" in command or "quit" in command:
                say("Goodbye!")
                break
        except sr.UnknownValueError:
            print("Sorry, I did not understand.")
        except sr.RequestError:
            print("Could not request results from speech recognition service.")

# Start speech recognition in a separate thread so it doesn't block camera processing
listener_thread = threading.Thread(target=listen_and_respond, daemon=True)
listener_thread.start()

cap = cv2.VideoCapture(0)
said_red = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Optional: Draw pose landmarks on frame
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Check red clothes
    if is_wearing_red(frame):
        if not said_red:
            say("Your color is red")
            said_red = True
        cv2.putText(frame, "Red color detected!", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        said_red = False

    cv2.imshow("Robot View", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
