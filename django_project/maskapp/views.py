
import os
import cv2
import base64
import numpy as np

from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from tensorflow.keras.models import load_model


# --------------------------------------------------
# MODEL AND FACE DETECTOR
# --------------------------------------------------

MODEL_PATH = os.path.join(
    settings.BASE_DIR,
    "face_mask_model_fixed.h5"
)

YUNET_PATH = os.path.join(
    settings.BASE_DIR,
    "face_detection_yunet_2023mar.onnx"
)

mask_model = None
face_detector = None


def load_detection_models():
    global mask_model, face_detector

    if mask_model is None:
        mask_model = load_model(MODEL_PATH, compile=False)

    if face_detector is None:
        if not os.path.exists(YUNET_PATH):
            raise FileNotFoundError(
                "YuNet model file not found: " + YUNET_PATH
            )

        face_detector = cv2.FaceDetectorYN_create(
            YUNET_PATH,
            "",
            (320, 320),
            0.6,
            0.3,
            5000
        )

    return mask_model, face_detector


# --------------------------------------------------
# FACE DETECTION AND MASK CLASSIFICATION
# --------------------------------------------------

def process_frame(image):
    model, detector = load_detection_models()

    height, width = image.shape[:2]

    detector.setInputSize((width, height))

    _, faces = detector.detect(image)

    results = []

    if faces is not None:
        for face in faces:

            x, y, w, h = face[:4].astype(int)

            # Keep bounding box within image boundaries
            x = max(0, x)
            y = max(0, y)
            w = min(w, width - x)
            h = min(h, height - y)

            if w <= 0 or h <= 0:
                continue

            face_crop = image[y:y+h, x:x+w]

            if face_crop.size == 0:
                continue

            # Same preprocessing used during training
            face_crop = cv2.resize(face_crop, (100, 100))
            face_crop = face_crop.astype("float32") / 255.0
            face_crop = np.expand_dims(face_crop, axis=0)

            prediction = model.predict(face_crop, verbose=0)[0]

            class_index = int(np.argmax(prediction))
            confidence = float(prediction[class_index]) * 100

            if class_index == 0:
                label = "MASK"
                color = (0, 200, 0)
            else:
                label = "NO MASK"
                color = (0, 0, 255)

            display_text = f"{label}: {confidence:.2f}%"

            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                color,
                2
            )

            cv2.putText(
                image,
                display_text,
                (x, max(y - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            results.append({
                "label": label,
                "confidence": round(confidence, 2)
            })

    return image, results


# --------------------------------------------------
# IMAGE UPLOAD PAGE
# --------------------------------------------------

def home(request):
    context = {}

    if request.method == "POST":
        uploaded_file = request.FILES.get("image")

        if not uploaded_file:
            context["error"] = "Please select an image."
            return render(request, "index.html", context)

        try:
            image_bytes = uploaded_file.read()

            image_array = np.frombuffer(
                image_bytes,
                dtype=np.uint8
            )

            image = cv2.imdecode(
                image_array,
                cv2.IMREAD_COLOR
            )

            if image is None:
                raise ValueError("Invalid image file.")

            output_image, results = process_frame(image)

            success, encoded_image = cv2.imencode(
                ".jpg",
                output_image
            )

            if not success:
                raise ValueError("Could not process the output image.")

            image_base64 = base64.b64encode(
                encoded_image.tobytes()
            ).decode("utf-8")

            context["result_image"] = (
                "data:image/jpeg;base64," + image_base64
            )

            context["results"] = results

            if not results:
                context["message"] = "No face detected in the image."

        except Exception as e:
            context["error"] = str(e)

    return render(request, "index.html", context)


# --------------------------------------------------
# LIVE WEBCAM FRAME API
# --------------------------------------------------

@require_POST
def webcam_detect(request):
    try:
        import json

        data = json.loads(request.body)

        image_data = data.get("image", "")

        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        image_bytes = base64.b64decode(image_data)

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:
            return JsonResponse(
                {"error": "Invalid webcam image."},
                status=400
            )

        output_image, results = process_frame(image)

        success, encoded_image = cv2.imencode(
            ".jpg",
            output_image
        )

        if not success:
            raise ValueError("Could not encode webcam frame.")

        image_base64 = base64.b64encode(
            encoded_image.tobytes()
        ).decode("utf-8")

        return JsonResponse({
            "image": "data:image/jpeg;base64," + image_base64,
            "results": results
        })

    except Exception as e:
        return JsonResponse(
            {"error": str(e)},
            status=500
        )
