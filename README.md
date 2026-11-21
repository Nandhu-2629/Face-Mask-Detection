# Face Mask Detection System

## 🚀 Live Demo

**Live Website:** [Face Mask Detection](https://face-mask-detection-pjpp.onrender.com)

Try the following features:
- Upload images to detect whether a person is wearing a face mask.
- Perform real-time face mask detection using your webcam.
- View detected faces with bounding boxes, classification results, and confidence scores.

**GitHub Repository:** [Face-Mask-Detection](https://github.com/Nandhu-2629/Face-Mask-Detection)

---

## Project Overview

An AI-powered Face Mask Detection System that identifies human faces and classifies whether a person is wearing a face mask using Deep Learning and Computer Vision.

This project combines a trained Convolutional Neural Network (CNN) with OpenCV-based face detection to identify mask usage.

Users can:
- Upload images for face mask detection.
- Perform real-time face mask detection using a webcam.
- View detected faces with bounding boxes.
- View classification results and confidence scores.

## Technologies Used

- **Programming Language:** Python
- **Deep Learning:** TensorFlow, Keras, CNN
- **Computer Vision:** OpenCV, YuNet
- **Web Framework:** Django
- **Frontend:** HTML, CSS, Bootstrap, JavaScript
- **Numerical Processing:** NumPy

## Project Structure

```text
FaceMaskDetection/
│
├── dataset/
├── django_project/
│   ├── django_project/
│   ├── maskapp/
│   │   ├── templates/
│   │   │   └── index.html
│   │   ├── views.py
│   │   └── urls.py
│   ├── face_mask_model_fixed.h5
│   ├── face_detection_yunet_2023mar.onnx
│   └── manage.py
│
├── train_mask_detector.py
├── detect_mask_video.py
├── convert_model.py
├── requirements.txt
├── .gitignore
├── .python-version
└── README.md
```

## Features

### 1. Image-Based Detection
Upload an image to detect faces and classify mask usage.

### 2. Real-Time Webcam Detection
Use the webcam to detect faces and classify mask usage in real time.

### 3. Face Detection
Uses the YuNet face detection model to locate faces and draw bounding boxes.

### 4. Mask Classification
Uses a trained CNN model to classify detected faces into:
- MASK
- NO MASK

### 5. Confidence Score
Displays the prediction confidence for detected faces.

## Installation and Setup

### Prerequisites
- Python 3.10
- Git
- VS Code (recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/Nandhu-2629/Face-Mask-Detection.git
cd Face-Mask-Detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the Django Application

Navigate to the Django project directory:

```bash
cd django_project
```

Run migrations:

```bash
python manage.py migrate
```

Start the development server:

```bash
python manage.py runserver
```

Open your browser:

http://127.0.0.1:8000/

## Models Used

- **CNN Model:** Trained face mask classification model.
- **YuNet:** Pretrained face detection model.

## Deployment

The application is deployed using **Render** and is publicly accessible.

**Live Application:** https://face-mask-detection-pjpp.onrender.com

The deployed application supports image-based detection and real-time webcam detection.

## Future Enhancements

- Improve model accuracy with a larger and more diverse dataset.
- Add detection statistics and analytics.
- Improve the user interface and user experience.
- Explore additional deployment and scalability options.

## Author

**Nandhitha Sri**

GitHub: [Nandhu-2629](https://github.com/Nandhu-2629)

GitHub Profile: https://github.com/Nandhu-2629