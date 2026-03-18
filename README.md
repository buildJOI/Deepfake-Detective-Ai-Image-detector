# 🕵️‍♂️ Deepfake Detective: AI Image & Video Detector

Deepfake Detective is a mobile application designed to identify AI-generated images and videos with high precision. By leveraging advanced Machine Learning models, this tool helps users distinguish between authentic media and synthetic content created by AI.

## 🚀 Overview

In an era of increasing digital misinformation, Deepfake Detective provides a reliable, on-the-go solution for media verification. The project utilizes a powerful Python-based backend for model inference and a sleek, cross-platform Flutter frontend for a seamless user experience.

### Key Features
- **Image Detection:** Upload or capture photos to check for AI manipulation.
- **Video Analysis:** Scan video files to identify deepfake patterns.
- **Real-time Results:** Get instant feedback on whether media is "Real" or "AI-Generated."
- **User-Friendly Interface:** Built with Flutter for smooth transitions and intuitive navigation.

## 🛠️ Tech Stack

- **Frontend:** [Flutter](https://flutter.dev/) (Dart)
- **Backend:** [Python](https://www.python.org/) (Flask/FastAPI)
- **Machine Learning:** PyTorch / TensorFlow (CNN-based architectures)
- **Deployment:** Docker, Railway

## 📁 Project Structure

```text
├── app/              # Python backend (ML models & API)
├── frontend/         # Flutter mobile application
nstallation & Setup
Prerequisites
Flutter SDK

Python 3.9+

Git

Backend Setup
Navigate to the app directory:

Bash
cd app
Install dependencies:

Bash
pip install -r ../requirements.txt
Run the server:

Bash
python main.py
Frontend Setup
Navigate to the frontend directory:

Bash
cd frontend
Get Flutter packages:

Bash
flutter pub get
Run the application:

Bash
flutter run
├── images/           # Documentation and asset files
├── Dockerfile        # Containerization configuration
└── requirements.txt  # Python dependencies
