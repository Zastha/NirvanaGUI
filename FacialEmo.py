import cv2
from deepface import DeepFace
import tensorflow as tf
import pandas as pd
import time
import datetime

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video
cap = cv2.VideoCapture(0)

# Variables to store emotion data
emotions_list = []
start_time = time.time()
duration = 10  # Duration in seconds for capturing emotions

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame to RGB format
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = rgb_frame[y:y + h, x:x + w]

        # Perform emotion analysis on the face ROI
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        # Get the emotion predictions and their confidences
        emotions = result[0]['emotion']

        # Determine the dominant emotion and its confidence
        dominant_emotion = max(emotions, key=emotions.get)
        emotion_confidence = emotions[dominant_emotion]

        # Store the emotions data
        emotions_list.append(emotions)

        # Draw rectangle around face and label with predicted emotion
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, dominant_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Display all emotions and their confidences
        y_offset = y + h + 20
        for emotion, confidence in emotions.items():
            text = f"{emotion}: {confidence:.2f}%"
            cv2.putText(frame, text, (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 20

        # Display the accuracy of the dominant emotion
        accuracy_text = f"Accuracy: {emotion_confidence:.2f}%"
        cv2.putText(frame, accuracy_text, (x, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)

    # Check if the duration has passed
    elapsed_time = time.time() - start_time
    if elapsed_time >= duration:
        break

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()

# Calculate the most prevalent emotion during the period
df = pd.DataFrame(emotions_list)
most_prevalent_emotion = df.mean().idxmax()

# Convert timestamps to readable format
start_time_formatted = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
end_time = time.time()
end_time_formatted = datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')

# Create a unique filename with a timestamp
timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
filename = f'emotion_detection_results_{timestamp}.csv'

# Save the results to a CSV file
results = {
    "start_timestamp": [start_time_formatted],
    "end_timestamp": [end_time_formatted],
    "prevalent_emotion": [most_prevalent_emotion]
}
results_df = pd.DataFrame(results)
results_df.to_csv(filename, index=False)

print(f"Resultados guardados en {filename}")
