# 🦾 Markerless Kinematic Tracking & Exoskeleton Control Simulation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.x-orange)
![Machine Learning](https://img.shields.io/badge/ML-Random_Forest-yellow)

An end-to-end Machine Learning Operations (MLOps) and Computer Vision pipeline designed to detect biomechanical gait abnormalities without physical markers, and simulate the corrective robotic torque required for an assistive lower-limb exoskeleton.

## 📖 Table of Contents
- [About the Project](#about-the-project)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage & Pipeline Execution](#usage--pipeline-execution)
- [Results & Performance](#results--performance)
- [Future Roadmap](#future-roadmap)
- [Contact](#contact)

## 🎯 About the Project
Traditional clinical biomechanics relies on expensive, marker-based Optical Motion Capture (OMC) systems. This project democratizes gait analysis by utilizing single-camera, lateral-view video feeds to extract highly accurate 3D joint kinematics. 

By feeding this kinematic data into a tuned machine learning classifier, the system autonomously diagnoses mobility impairments (specifically Stiff-Knee Gait). Furthermore, it bridges the gap between software and hardware by passing the diagnostic data through a simulated Proportional (P) Controller to calculate real-time, physical torque requirements (in Nm) for robotic assistive orthoses.

### Key Capabilities:
* **Markerless Pose Estimation:** Extracts 3D spatial coordinates for the hip, knee, and ankle joints in real-time.
* **Signal Processing:** Implements Savitzky-Golay filtering and percentile-based outlier rejection to eliminate visual tracking noise and camera artifacts.
* **Autonomous Diagnosis:** Classifies gait as "Normal" or "Abnormal" with **83.33% accuracy** based on engineered Range of Motion (ROM) features.
* **Control Theory Integration:** Calculates dynamic assistive actuator torque ($\tau = K_p \cdot \Delta\theta$) to correct joint extension deficits.

## ⚙️ System Architecture

The pipeline is broken down into three primary phases:

1. **Phase 1: Extraction (Computer Vision)**
   - Raw `.mp4` video is parsed frame-by-frame.
   - MediaPipe's BlazePose topology maps 33 anatomical landmarks.
   - 3D vector math calculates the interior angle of the knee joint.
2. **Phase 2: Processing & ML (Data Science)**
   - Raw kinematic waveforms are smoothed.
   - The 5th and 95th percentiles are extracted to determine the true, robust Range of Motion (ROM), rejecting visual anomalies.
   - A Random Forest classifier evaluates the features against clinical baselines.
3. **Phase 3: Control Simulation (Robotics)**
   - If ROM falls below the clinical healthy threshold (e.g., < 65°), the system calculates the delta between the target angle and actual angle.
   - A Proportional gain ($K_p$) translates this delta into physical torque commands.

## 💻 Tech Stack
* **Language:** Python
* **Computer Vision:** OpenCV, Google MediaPipe
* **Data Processing & Math:** NumPy, Pandas, SciPy (Signal Processing)
* **Machine Learning:** Scikit-Learn
* **Development Environment:** Jupyter Notebooks, VS Code

## 📁 Project Structure

```text
markerless-kinematics/
│
├── data/
│   ├── raw/                  # (Git-Ignored) Raw lateral-view .mp4 files
│   └── processed/            # Extracted kinematic time-series CSVs
│
├── notebooks/                # Research & Development Environment
│   ├── 01_landmark_extraction.ipynb
│   ├── 02_kinematic_modeling.ipynb
│   ├── 03_gait_classification_ml.ipynb
│   └── 04_exoskeleton_torque_sim.ipynb
│
├── src/                      # Production Modules
│   └── biomechanics.py       # Core mathematical and control theory logic
│
├── main.py                   # Automated MLOps inference pipeline
├── requirements.txt          # Python dependencies
└── README.md
