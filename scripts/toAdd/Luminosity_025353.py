import io
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.image as mpimg
import numpy as np
import os
from PIL import Image

######## 1. READ IMAGE ######
# 2025 v2012 event parameters
path = "09_21_025353_v2012/"  # UPDATE THIS PATH to match your folder structure
interval = 25.0  #### microseconds
exposure = 2.41 #### microseconds
Starttime = 324022.1  #### micro seconds (from v711 spectroscopy analysis)

# ROI parameters - UPDATE THESE after viewing an image
# These define the region of interest where luminosity is calculated
someX_big, someY_big, size_bigX, size_bigY = 800, 280,100,140  # Adjust based on your v2012 images

# Test image to verify ROI position
im_test = mpimg.imread(path + "IMG000100.tif")  # 6-digit format for 2025
im_cut = im_test[:,:]
plt.figure()
plt.title("v2012 - Check ROI position")
plt.imshow(im_cut)
currentAxis = plt.gca()
currentAxis.add_patch(Rectangle((someX_big, someY_big), size_bigX, size_bigY, edgecolor='red', fill=None, alpha=1))
plt.savefig("Luminosity_ROI_check_v2012.png")
print(f"Image shape: {im_test.shape[0]} x {im_test.shape[1]}")
plt.show()

# Background frames - use dark frames at the end or beginning
# Check your image sequence to find appropriate background frames
bg_start = 0  # Starting frame for background
bg_count = 10   # Number of frames to average

bg = np.zeros((bg_count, im_test.shape[0], im_test.shape[1]))
for i in range(0, bg_count):
    frame_num = bg_start + i
    m = "{:06d}".format(frame_num)  # 6-digit format for 2025 v2012
    try:
        bg[i,:,:] = mpimg.imread(path + "IMG" + str(m) + ".tif")
    except:
        print(f"Warning: Could not read background frame {frame_num}")
        bg_count = i
        break

bg_mean = np.median(bg[:bg_count], axis=0)

def CAL_lum(im, n, someX, someY, sizeX, sizeY):
    """Calculate luminosity in a specified region"""
    im_cut = im[:, :]
    n_time = Starttime + (n) * interval
    time_new = int(n_time)   ##### just to make the number integer
    im_lu = im_cut[someY:someY + sizeY, someX:someX + sizeX]
    im_lu = np.asarray(im_lu)
    lu = sum(im_lu)
    lu = sum(lu)
    lu_big = sum(im_lu)
    lu_big = sum(lu_big)
    return lu_big

# LOADING TIF IMAGES AND CALCULATING LUMINOSITY
os.chdir(path)
output_file = "../Luminosity_20250921_025353_v2012.txt"

with open(output_file, 'w') as f:
    f.write("%s %s \n" % ("#Time", "Lum big"))
    for file in os.listdir():
        if file.endswith(".tif"):
            im = mpimg.imread(file)
            # Extract frame number - 6 digits for 2025 v2012 (IMG000000.tif)
            n = float(file[3:9])  # "IMG" is 3 chars, then 6 digits
            print(f"Processing frame {int(n)}")
            n_time = Starttime + n * interval
            im = im - bg_mean
            lu_big = CAL_lum(im, n, someX_big, someY_big, size_bigX, size_bigY)
            f.write("%f %f \n" % (n_time, lu_big))

print(f"Luminosity data saved to {output_file}")

# Plot the luminosity
lumi = np.loadtxt(output_file)
N = -1
plt.figure(figsize=(10, 5))
plt.title("TGFs - 20250921 - 025353 (v2012)")
plt.plot(lumi[:N,0], lumi[:N,1]/np.max(lumi[:N,1]), linewidth=2, color='green', label="Luminosity")
plt.xlabel("Time ($\mu$s)")
plt.ylabel("Luminosity (a.u.)")
plt.legend()
plt.savefig("./../Luminosity_20250921_025353_v2012.png")
plt.show()