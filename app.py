from flask import Flask, render_template, Response, jsonify
import cv2
import time
from rl_agent import RLAgent
from scrape_lessons import fetch_lessons

app = Flask(__name__)

camera = cv2.VideoCapture(0)
agent = RLAgent()
last_popup_time = 0
popup_cooldown = 10  # seconds

lessons = fetch_lessons()  # scraped lessons

def generate_frames():
    global last_popup_time
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    while True:
        success, frame = camera.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        state = "focused" if len(faces) > 0 else "distracted"
        action = agent.choose_action(state)

        # Draw rectangle on detected face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Popup trigger when distracted
        if state == "distracted" and (time.time() - last_popup_time) > popup_cooldown:
            last_popup_time = time.time()
            yield (b'--frame\r\n'
                   b'Content-Type: text/plain\r\n\r\nPOPUP\r\n')
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    lessons = fetch_lessons()
    main_video = lessons[0] if lessons else {
        "title": "8th Grade Science",
        "url": "https://www.youtube.com/watch?v=ur0hCdne2Ew",
        "description": "Learn about exciting science concepts for Class 8."
    }
    return render_template('index.html', lessons=lessons, main_video=main_video)



@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
