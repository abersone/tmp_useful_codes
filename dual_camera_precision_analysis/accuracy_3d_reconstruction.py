import math
## Input ##
print('----Test 1: Module Config----')
# ps = 0.002 # pixel size, unit mm
# wid = 2448 # image width, unit pixel
# max_match_err = 0.5 # dualcam match error, unit pixel
# Z = 1000.0 # measure distance ,unit mm
# b = 100 # dualcam baseline, unit mm
# fov_wid = 40 # horizontal fov of camera, unit degree

ps = 0.003 # pixel size, unit mm
wid = 3072 # image width, unit pixel
max_match_err = 0.2 # dualcam match error, unit pixel
Z = 300.0 # measure distance ,unit mm
b = 230 # dualcam baseline, unit mm
#fov_wid = 40 # horizontal fov of camera, unit degree


f = 38 #mm
radians = math.atan(ps * wid / 2 / f)
fov_wid = radians * 180 / math.pi * 2

#radians = (fov_wid / 2) / 180 * math.pi # the half horizontal fov of cameara, unit radian
#f = ps * wid / 2 / math.tan(radians) # focal lens of camera

scale = 0.8
X = math.tan(radians) * Z * scale # mm
hori_range = 2.0 * X / scale

sqrt_2 = math.sqrt(2)
s = sqrt_2 * max_match_err
Ez = (Z / b) * (Z / f) * max_match_err * ps
Ex = math.sqrt(1.0 + (X / (sqrt_2 * b)) * (X / (sqrt_2 * b))) * Z * s / (f / ps)
Ey = math.sqrt(1.0 + (X / (sqrt_2 * b)) * (X / (sqrt_2 * b))) * Z * s / (f / ps)

## Output ##
#print("focus_lens:", f)
print('fov_wid:', fov_wid)
print('max horizontal measure range = ', hori_range, " mm")
print("(max err_x, err_y, err_z) = ", Ex, Ey, Ez)

