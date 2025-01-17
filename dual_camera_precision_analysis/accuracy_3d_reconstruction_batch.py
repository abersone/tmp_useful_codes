import math
import matplotlib.pyplot as plt
import numpy as np

# Function to calculate Ex, Ey, Ez
def calculate_errors(ps, wid, max_match_err, Z, b, fov_wid, scale=0.8):
    radians = (fov_wid / 2) / 180 * math.pi  # Half horizontal FOV in radians
    f = ps * wid / 2 / math.tan(radians)    # Focal length
    
    X = math.tan(radians) * Z * scale       # mm
    hori_range = 2.0 * X / scale            # mm
    
    sqrt_2 = math.sqrt(2)
    s = sqrt_2 * max_match_err
    Ez = (Z / b) * (Z / f) * max_match_err * ps
    Ex = math.sqrt(1.0 + (X / (sqrt_2 * b)) ** 2) * Z * s / (f / ps)
    Ey = Ex  # Since Ex and Ey are calculated the same way
    
    return Ex, Ey, Ez, hori_range

# Initial Parameters
ps = 0.002          # pixel size, unit mm
wid = 1024          # image width, unit pixel
max_match_err = 0.5 # dualcam match error, unit pixel
Z_initial = 2000.0  # measure distance ,unit mm
b_initial = 80     # dualcam baseline, unit mm 
fov_wid_initial = 60 # horizontal fov of camera, unit degree

scale = 0.7

# ============================
# 1. Varying Z from 400mm to 2000mm
# ============================
Z_values = np.arange(300, 5001, 50)  # 400mm to 2000mm with step 10mm
Ex_Z, Ey_Z, Ez_Z = [], [], []
hori_range_Z = []

for Z in Z_values:
    Ex, Ey, Ez, hori = calculate_errors(ps, wid, max_match_err, Z, b_initial, fov_wid_initial, scale)
    Ex_Z.append(Ex)
    Ey_Z.append(Ey)
    Ez_Z.append(Ez)
    hori_range_Z.append(hori)

# Plotting Measurement Errors vs Z
plt.figure(figsize=(10, 6))
plt.plot(Z_values, Ex_Z, label='Ex (Error X,Y)')
#plt.plot(Z_values, Ey_Z, label='Ey (Error Y)')
plt.plot(Z_values, Ez_Z, label='Ez (Error Z)')
plt.title('Measurement Errors vs Distance Z')
plt.xlabel('Distance Z (mm)')
plt.ylabel('Measurement Error (mm)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================
# 2. Varying Baseline b from 100mm to 500mm
# ============================
b_values = np.arange(50, 121, 10)  # 100mm to 500mm with step 50mm
Ex_b, Ey_b, Ez_b = [], [], []

for b in b_values:
    Ex, Ey, Ez, hori = calculate_errors(ps, wid, max_match_err, Z_initial, b, fov_wid_initial, scale)
    Ex_b.append(Ex)
    Ey_b.append(Ey)
    Ez_b.append(Ez)

# Plotting Measurement Errors vs Baseline b
plt.figure(figsize=(10, 6))
plt.plot(b_values, Ex_b, label='Ex (Error X,Y)')
#plt.plot(b_values, Ey_b, label='Ey (Error Y)')
plt.plot(b_values, Ez_b, label='Ez (Error Z)')
plt.title('Measurement Errors vs Baseline b')
plt.xlabel('Baseline b (mm)')
plt.ylabel('Measurement Error (mm)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================
# 3. Varying Field of View Width from 30° to 60°
# ============================
fov_values = np.arange(50, 111, 5)  # 30° to 60° with step 5°
Ex_fov, Ey_fov, Ez_fov = [], [], []

for fov_wid in fov_values:
    Ex, Ey, Ez, hori = calculate_errors(ps, wid, max_match_err, Z_initial, b_initial, fov_wid, scale)
    Ex_fov.append(Ex)
    Ey_fov.append(Ey)
    Ez_fov.append(Ez)

# Plotting Measurement Errors vs Field of View Width
plt.figure(figsize=(10, 6))
plt.plot(fov_values, Ex_fov, label='Ex (Error X,Y)')
#plt.plot(fov_values, Ey_fov, label='Ey (Error Y)')
plt.plot(fov_values, Ez_fov, label='Ez (Error Z)')
plt.title('Measurement Errors vs Field of View Width')
plt.xlabel('Field of View Width (degrees)')
plt.ylabel('Measurement Error (mm)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
