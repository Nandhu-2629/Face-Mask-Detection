
import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model

# Get project directory
BASE_DIR = Path(__file__).resolve().parent

# Load trained model
model = load_model(
    str(BASE_DIR / "django_project" / "face_mask_model_fixed.h5"),
    compile=False
)

# Load face detector
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Unable to open webcam")
    exit()

print("Webcam started! Press Q or ESC to close.")

try:
    while True:

        ret, frame = cap.read()

        if not ret:
            print("Unable to capture webcam frame")
            break

        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_classifier.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:

            # Extract face
            face = frame[y:y+h, x:x+w]

            # Preprocess image
            face = cv2.resize(face, (100, 100))
            face = face.astype("float32") / 255.0
            face = np.reshape(face, (1, 100, 100, 3))

            # Predict
            probabilities = model.predict(face, verbose=0)[0]
            prediction = np.argmax(probabilities)
            confidence = probabilities[prediction] * 100

            # Display prediction
            if prediction == 0:
                label = "MASK"
                color = (0, 255, 0)
            else:
                label = "NO MASK"
                color = (0, 0, 255)

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                color,
                2
            )

            cv2.putText(
                frame,
                f"{label} {confidence:.1f}%",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        # Display webcam frame OUTSIDE the face loop
        cv2.imshow("Face Mask Detection", frame)

        # Press Q or ESC to exit
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
