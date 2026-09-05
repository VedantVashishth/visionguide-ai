# VisionGuide AI 👁️

VisionGuide AI is a real-time, AI-powered visual companion designed specifically for blind and visually impaired users. It consolidates critical accessibility tools into a single, unified web application driven by a natural voice-command interface. 

By strategically blending **Cloud-based Generative AI (Gemini 3.6 Flash)** for high-level reasoning with **Local Machine Learning Models (YOLO, OpenCV)** for real-time edge processing, VisionGuide acts as a pair of "smart eyes"—running smoothly even on low-end CPUs without a dedicated GPU.

## ✨ Core Features

* **🎙️ Push-to-Talk Voice Interface:** Hold the screen, speak your intent (e.g., *"What's in front of me?"*, *"Read this sign"*), and the system routes the request intelligently.
* **🗣️ Context-Aware Scene Narration:** Powered by the Gemini API, get concise, natural-language descriptions of your surroundings.
* **🛑 Throttled Obstacle Detection:** A continuous background loop (using a lightweight YOLO11n model) scans for obstacles every 2.5 seconds to keep CPU usage low. Includes **Web Audio Proximity Cues** (distinct beeps) for objects approaching too closely.
* **👤 Privacy-First Face Recognition:** Learn and remember faces completely locally (via OpenCV LBPH). Face data never leaves your device. When recognized, names are dynamically injected into scene descriptions.
* **💵 Currency & 📄 Text Reading:** Point the camera at a bank note or a document to have the denomination or text extracted and read aloud (powered by Gemini & EasyOCR).
* **🆘 Location & Emergency Fall Detection:** Uses the device's accelerometer. If a potential fall is detected and the user fails to respond verbally, an emergency alert containing a Google Maps-resolved address is dispatched via Email.

## 🛠️ Tech Stack & Architecture

* **Frontend:** High-contrast, single-file Vanilla JavaScript and HTML "HUD". Zero heavy framework dependencies.
* **Backend:** Asynchronous Python backend powered by **FastAPI**.
* **AI Engines:** 
  * **Gemini 3.6 Flash** (Scene Narration, Currency Identification, Intent Routing)
  * **YOLO11n** (Obstacle Detection)
  * **OpenCV / Haar Cascades + LBPH** (Local Face Recognition)
  * **EasyOCR** (Text Extraction)
* **Optimization Highlights:** To accommodate low-end CPUs, VisionGuide employs **Frame-Change Gating**. Before pinging the Gemini API, a lightweight grayscale histogram compares the current frame to the previous one. If the camera hasn't moved, it skips the API call entirely and reuses the cached response, saving both API quota and compute time.

## 🚀 Getting Started

### Prerequisites
* Python 3.10+
* A modern browser (Chrome or Edge recommended for the Web Speech API)

### 1. Installation
Clone the repository and install the dependencies in a virtual environment:
```bash
git clone https://github.com/yourusername/visionguide-ai.git
cd visionguide-ai
python -m venv visionguide_env
.\visionguide_env\Scripts\activate  # On Windows
pip install -r backend/requirements.txt
```

### 2. Configuration
Create a `.env` file in the root directory (do not commit this file):
```env
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL_NAME=gemini-3.6-flash
APP_USER_EMAIL=test@example.com
APP_USER_PASSWORD=changeme
```

### 3. Running the App
VisionGuide includes a helper PowerShell script to start both the FastAPI backend and the frontend HTTP server simultaneously:
```powershell
.\start.ps1
```
The backend will launch on `http://127.0.0.1:8000` and the frontend will automatically open at `http://127.0.0.1:5500`.

## 🔒 Privacy & Safety
VisionGuide is designed with privacy in mind. Face recognition embeddings and models are stored strictly on disk (`models/faces/`). The app does not continuously stream video to the cloud; it only sends discrete, user-triggered snapshots to Gemini when explicit features (like Scene Narration) are requested.

## 🔮 Future Scope
* **Native Mobile App:** Migrating the frontend to React Native or Flutter for background execution and deeper hardware access.
* **Offline VLMs:** Integrating quantized local Vision-Language Models (e.g., Llama-3-Vision) to function without internet connectivity.
* **Spatial Audio:** Mapping depth-estimation data to spatial audio cues (e.g., hearing a beep in your left ear when an object is approaching from the left).
