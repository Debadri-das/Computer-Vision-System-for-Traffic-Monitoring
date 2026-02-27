import sys
print("=" * 50)
print("ENVIRONMENT TEST")
print("=" * 50)
print("Python location:", sys.executable)
print("Python version:", sys.version)
print()

try:
    import cv2
    print("✅ OpenCV version:", cv2.__version__)
except ImportError as e:
    print("❌ OpenCV FAILED:", e)

try:
    import numpy as np
    print("✅ NumPy version:", np.__version__)
except ImportError as e:
    print("❌ NumPy FAILED:", e)

try:
    import supervision as sv
    print("✅ Supervision version:", sv.__version__)
except ImportError as e:
    print("❌ Supervision FAILED:", e)

try:
    from ultralytics import YOLO
    print("✅ YOLOv8 installed successfully")
except ImportError as e:
    print("❌ YOLOv8 FAILED:", e)

try:
    import streamlit as st
    print("✅ Streamlit installed successfully")
except ImportError as e:
    print("❌ Streamlit FAILED:", e)

print()
print("=" * 50)
print("If all ✅, you're ready to build!")
print("=" * 50)
