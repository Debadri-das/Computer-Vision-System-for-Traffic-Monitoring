# 🚦 AI Traffic Monitoring System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-FF6B35?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9+-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)
![MacOS](https://img.shields.io/badge/MacOS-Compatible-000000?style=for-the-badge&logo=apple&logoColor=white)

<br/>

**A real-time Computer Vision system for intelligent traffic monitoring, vehicle detection, tracking, and congestion analysis — built for smart city infrastructure.**

<br/>

[Features](#-features) • [Demo](#-demo) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Roadmap](#-roadmap)

</div>

---

## 📸 Demo

> Upload your own traffic video and watch the AI detect, track, and count vehicles in real-time.

```
🚗 Vehicle detected → 🔢 Unique ID assigned → 📊 Stats updated → 🚦 Dashboard live
```

---

## ✨ Features

| Feature | Description | Status |
|---|---|---|
| 🎯 Real-time Detection | YOLOv8 Nano for fast, accurate vehicle detection | ✅ Live |
| 🔢 Multi-Object Tracking | ByteTrack assigns persistent IDs across frames | ✅ Live |
| 📊 Vehicle Counting | Counts vehicles crossing a virtual tripwire | ✅ Live |
| 🚗 Vehicle Classification | Detects cars, trucks, buses, motorcycles | ✅ Live |
| 📈 Live Dashboard | Real-time Streamlit web interface with metrics | ✅ Live |
| 🎥 Webcam Support | Works with live webcam feed | ✅ Live |
| ⚡ FPS Monitoring | Real-time performance tracking | ✅ Live |
| 🚑 Emergency Detection | Ambulance/police vehicle prioritization | 🔄 Coming Soon |
| 🗺️ Heatmap Analysis | Traffic density heatmaps per lane | 🔄 Coming Soon |
| ☁️ Cloud Deployment | AWS/Railway deployment ready | 🔄 Coming Soon |

---

## 🏗️ Architecture

```
📁 TrafficMonitor/
├── 📄 app.py               # Streamlit web dashboard (Frontend)
├── 📄 detector.py          # YOLOv8 detection engine (Backend)
├── 📄 requirements.txt     # Python dependencies
├── 📄 README.md            # You are here
└── 🎥 traffic.mp4          # Sample test video
```

### Tech Stack

```
┌─────────────────────────────────────────────────┐
│                  USER BROWSER                   │
│              (Streamlit Dashboard)               │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│               STREAMLIT SERVER                  │
│                  (app.py)                       │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────┐
│             DETECTION ENGINE                    │
│               (detector.py)                     │
│                                                 │
│   ┌──────────┐  ┌───────────┐  ┌────────────┐  │
│   │ YOLOv8n  │→ │ ByteTrack │→ │  OpenCV    │  │
│   │Detection │  │ Tracking  │  │ Annotation │  │
│   └──────────┘  └───────────┘  └────────────┘  │
└─────────────────────────────────────────────────┘
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.9 or higher
- MacOS / Linux / Windows
- 4GB RAM minimum (8GB recommended)

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/TrafficMonitor.git
cd TrafficMonitor
```

### Step 2: Create Virtual Environment

```bash
python3 -m venv traffic_env
source traffic_env/bin/activate     # MacOS/Linux
# traffic_env\Scripts\activate      # Windows
```

### Step 3: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Download Sample Video (Optional)

```bash
curl -o traffic.jpg "https://ultralytics.com/images/bus.jpg"
```

---

## 🚀 Usage

### Run the Web Dashboard

```bash
streamlit run app.py
```

Then open your browser at: `http://localhost:8501`

### Run Detection Engine Directly

```bash
python3 detector.py
```

### Dashboard Controls

1. **Upload Video** — Upload any MP4 traffic video from the sidebar
2. **Select Source** — Choose between video file or live webcam
3. **Click Start** — Begin real-time AI analysis
4. **Monitor Stats** — Watch live FPS, vehicle counts, and crossing data

---

## 📊 How It Works

### 1. Detection (YOLOv8)
YOLOv8 Nano scans each video frame and draws bounding boxes around detected vehicles. It classifies them into:
- 🚗 Car
- 🚌 Bus  
- 🚛 Truck
- 🏍️ Motorcycle

### 2. Tracking (ByteTrack)
Each detected vehicle gets a **unique persistent ID** (`#1`, `#2`, `#3`...). ByteTrack is smart enough to keep the same ID even if a vehicle temporarily disappears behind another object (occlusion).

### 3. Counting (Virtual Tripwire)
A virtual line is drawn across the road. Every time a vehicle crosses this line, the **IN** or **OUT** counter increments depending on direction of travel.

### 4. Dashboard (Streamlit)
All data streams live into a professional web dashboard showing:
- Live annotated video feed
- Active vehicle count
- IN / OUT crossing totals
- Real-time FPS

---

## 📦 Dependencies

```txt
ultralytics==8.1.0        # YOLOv8 object detection
opencv-python==4.9.0      # Video processing
streamlit==1.30.0         # Web dashboard
supervision==0.18.0       # Tracking & annotation utilities
numpy==1.26.0             # Numerical computing
pandas==2.1.0             # Data handling
```

---

## 🗺️ Roadmap

- [x] Real-time vehicle detection
- [x] Multi-object tracking with unique IDs
- [x] Vehicle counting (IN/OUT)
- [x] Live Streamlit dashboard
- [x] Webcam support
- [ ] Speed estimation (km/h)
- [ ] Lane-specific counting
- [ ] Emergency vehicle alerts (ambulance priority)
- [ ] Traffic density heatmaps
- [ ] Congestion alerts
- [ ] REST API endpoints
- [ ] Docker containerization
- [ ] AWS/Railway deployment
- [ ] Indian traffic dataset fine-tuning (rickshaws, autos)

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/SpeedEstimation`)
3. Commit your changes (`git commit -m 'Add speed estimation'`)
4. Push to the branch (`git push origin feature/SpeedEstimation`)
5. Open a Pull Request

---

## 👤 Author

**Your Name**  
Third-year CS Student | AI/ML Enthusiast  
📍 Kolkata, India

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/YOUR_USERNAME)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/YOUR_PROFILE)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — Object detection model
- [Supervision](https://github.com/roboflow/supervision) — CV utility library by Roboflow
- [Streamlit](https://streamlit.io/) — Web dashboard framework
- [ByteTrack](https://github.com/ifzhang/ByteTrack) — Multi-object tracking algorithm

---

<div align="center">
  <p>⭐ If this project helped you, please give it a star!</p>
  <p>Built with ❤️ for smart city infrastructure and Indian traffic management</p>
</div>
