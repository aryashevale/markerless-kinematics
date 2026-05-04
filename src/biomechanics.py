import numpy as np

def calculate_angle_3d(a, b, c):
    """Calculates 3D angle with b as vertex."""
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def get_proportional_torque(target_angle, actual_angle, kp=0.5):
    """Calculates corrective torque using P-control."""
    return kp * (target_angle - actual_angle)