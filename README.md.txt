# Markerless Kinematic Tracking & Exoskeleton Control Simulation

An end-to-end MLOps pipeline that utilizes Computer Vision to detect biomechanical gait abnormalities and simulates the corrective robotic torque required for an assistive exoskeleton. 

## Project Overview
Traditional biomechanical motion capture is expensive and requires physical markers. This project uses **MediaPipe** to extract 3D kinematic data directly from lateral-view video feeds. The data is processed through a **Random Forest Classifier** to diagnose stiff-knee gait patterns, and a **Proportional (P) Control System** calculates the real-time torque (Nm) required to correct the user's stride.

### Key Features:
* **Markerless Pose Estimation:** Extracts 3D hip, knee, and ankle coordinates.
* **Signal Processing:** Implements Savitzky-Golay filtering for robust outlier rejection and sensor noise reduction.
* **Machine Learning:** Diagnoses gait abnormalities with **83.33% accuracy** based on engineered Range of Motion (ROM) features.
* **Control Theory:** Simulates real-time assistive actuator torque ($\tau = K_p \cdot \Delta\theta$).

## Repository Structure
```text
markerless-kinematics/
│
├── data/
│   ├── raw/                  # (Ignored) Place raw .mp4 files here
│   └── processed/            # Extracted kinematic CSV files
│
├── notebooks/
│   ├── 01_landmark_extraction.ipynb
│   ├── 02_kinematic_modeling.ipynb
│   ├── 03_gait_classification_ml.ipynb
│   └── 04_exoskeleton_torque_sim.ipynb
│
├── src/
│   └── biomechanics.py       # Core mathematical and control logic
│
├── main.py                   # Automated inference pipeline
├── requirements.txt          # Environment dependencies
└── README.md