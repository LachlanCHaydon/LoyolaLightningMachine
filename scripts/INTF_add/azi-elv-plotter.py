#!/usr/bin/env python

import sys
import os
import numpy as np
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib
import matplotlib.pyplot as plt
import geopy as gp
import matplotlib.image as mpimg
from PIL import Image, ImageOps
import taTools as ta
from mpl_toolkits.axes_grid1 import make_axes_locatable

intfLat = 39.339082
intfLon = -112.700696
intfGps = gp.point.Point(intfLat, intfLon, 1.4)  # intf antenna A
intfCart = ta.gps2cart(intfGps)

###########################
# Event: Sept 21, 2025 - 02:53:53
# Calculated Distance (x1): 17.7 km
# Calculated Azimuth: 256.7 deg
# Calculated Elevation: 5.8 deg
###########################

##### Manual Settings #####
# set time limits manually (us):
tStart, tStop = 324200, 328000  ######
# Set azi and elv limits and ticks
azi_limits = (249, 260)  # Centered around 256.7
azi_ticks = range(azi_limits[0] - 1, azi_limits[1] + 2, 2)
elv_limits = (0, 10)  # Fits the 5.8 deg elevation
elv_ticks = range(elv_limits[0], elv_limits[1] + 2, 2)
alt_ticks = range(0, 5, 1)

# data offsets and scaling - do not change unless Mark says so
elv_offset = 0.3

###############################
##### Input file handling #####
detectors, times, energies, azis, cores, sources, sources_alt = ([] for i in range(7))

# UPDATED FILE PATH
inFile = "INTF/TAR_intf_250921_025353_azi_cutFeb10.dat"

# Load INTF data:
# Parsing logic updated for format: TAR_intf_250921_025353_...
filename_parts = inFile.split('/')[-1].split('_')
date = filename_parts[2]  # 250921
time = filename_parts[3]  # 025353

# Note: Using generic load, assuming file format matches standard TASD output
# If using the 'cut' file from the previous script, structure might be slightly different than 'scribble'
# Assuming columns 0-5 exist.
try:
    intfList = np.genfromtxt(inFile, skip_header=0, usecols=(0, 1, 2, 3, 4, 5))
except:
    # Fallback if header exists or columns differ in the 'cut' file
    intfList = np.genfromtxt(inFile, names=True)
    # You might need to adjust this depending on if your 'cutFeb10.dat' has a header

# Ensure it's a numpy array if not already
# ... (after loading intfList) ...

# ---------------------------------------------------------
# OPTIMIZED VECTORIZED FILTERING (Replaces the slow while loop)
# ---------------------------------------------------------

# 1. Filter by Time (Keep points between tStart and tStop)
# This replaces the entire 'while ii < len(intfList)...' block
time_mask = (intfList[:, 0] >= tStart) & (intfList[:, 0] <= tStop)
intfList = intfList[time_mask]

print(f"Points after time cut: {len(intfList)}")

# 2. Filter by Elevation (Keep points between elvMin and elvMax)
elvMin, elvMax = (0, 10)
elv_mask = (intfList[:, 2] >= elvMin) & (intfList[:, 2] <= elvMax)
intfList = intfList[elv_mask]

# 3. Filter by Azimuth (Keep points between aziMin and aziMax)
aziMin, aziMax = (250, 265)
azi_mask = (intfList[:, 1] >= aziMin) & (intfList[:, 1] <= aziMax)
intfList = intfList[azi_mask]

print(f"Points remaining for plot: {len(intfList)}")

# ---------------------------------------------------------
# Load TASD data (Continue with rest of script below...)
sdDir = "TASD/SD_0921_025353/"
# ...
# Load TASD data:
# UPDATED DIRECTORY
sdDir = "TASD/SD_0921_025353/"

# print (sdDir)
for root, dirs, files in os.walk(sdDir):  # check all TASD files in sdDir
    for filename in files:
        item = filename.split('/')[-1].split('_')
        # Logic to find SD files.
        # Your previous script used SD_0921..., so item[0] might be SD or SD_0921
        if filename.startswith('SD'):
            ff2 = open(sdDir + filename, 'r')
            for line2 in ff2:
                if line2.startswith('TASD'):
                    detectors.append(line2.split(' ')[1][:4])
                elif line2.startswith('burst times'):
                    times.append(line2.replace(',', '').split(' ')[3:])
                elif line2.startswith('burst VEM'):
                    energies.append(line2.replace(',', '').split(' ')[3:])
                elif line2.startswith('##'):
                    break
            ff2.close()

        # Logic to find Analysis files (output from your previous script)
        elif filename.startswith('Analysis'):
            source_on = True
            ff2 = open(sdDir + filename, 'r')
            for line2 in ff2:
                if line2.startswith(' elv(deg)'):
                    # tempElv=float(line2.split(' ')[2])
                    tempElv = float(line2.split('=')[1].split(' ')[0])
                    # UPDATED DISTANCE CONSTANT: 17.83 km
                    tempAlt = 17.83 * np.tan(np.pi * tempElv / 180.)
                elif line2.startswith(' azi(deg)'):
                    tempAzi = float(line2.split('=')[1].split(' ')[0])
            ff2.close()
            try:
                sources.append([tempAzi, tempElv])
                sources_alt.append([tempAzi, tempAlt])
            except:
                pass  # handle cases where file might be empty
print("Source = ", sources)

# calculate azimuth angle to each detector
ff1 = open('tasd_gpscoors.txt', 'r')
for det in detectors:
    ff1.seek(0)
    for line in ff1:
        if line[:4] == det:
            tempLat = line.split()[1]
            tempLon = line.split()[2]
            break
    tempGps = gp.point.Point(tempLat, tempLon, 1.4)
    tempCart = ta.gps2cart(tempGps)
    dx = abs(tempCart[0] - intfCart[0])
    dy = tempCart[1] - intfCart[1]
    if dy > 0:
        azis.append(270 + (180. / np.pi) * np.arctan(dy / dx))
    else:
        azis.append(270 - (180. / np.pi) * np.arctan(abs(dy) / dx))
ff1.close()

print("Azimuth value = ", azis)
# calculate approx. core lcoation for each trigger
trigDets, trigErgs, trigAzis = ([] for i in range(3))
azis = np.array(azis)
energies = np.array(energies)
trigs = len(energies[0])
trigNum = 0
while trigNum < trigs:
    tempDets, tempErgs, tempAzis, tempWeight = ([] for i in range(4))
    ii = 0
    while ii < len(detectors):
        if float(energies[ii][trigNum]) > 0.0:
            tempDets.append(detectors[ii])
            tempErgs.append(float(energies[ii][trigNum]))
            tempAzis.append(azis[ii])
            tempWeight.append(float(azis[ii]) * float(energies[ii][trigNum]))
        ii += 1
    trigDets.append(tempDets)
    trigErgs.append(tempErgs)
    trigAzis.append(tempAzis)
    tempWeight = np.array(tempWeight)
    tempErgs = np.array(tempErgs)
    cores.append(np.sum(tempWeight) / np.sum(tempErgs))
    trigNum += 1


################

# UPDATED CONSTANT TO 31.1 km
def CtoF(x):  ### elevation - height
    return 17.83 * np.tan(x * np.pi / 180)


def FtoC(x):
    return (np.arctan(x / 17.83)) * 180 / np.pi


def CxtoFx(x):  ### azimuth - distance
    # Keeping original scaling for now, but note that 3.6 was the old distance.
    # 31.1 is the new distance.
    return (x * 17.83) / 7.52 - 122.3


def FxtoCx(x):
    return (x * 7.52) / 17.83 - 255.42


fig = plt.figure(figsize=(12, 8))
ax = fig.add_axes((0.1, 0.1, 0.82, 0.77))
cmap = "jet"
Nsize = 20

# Check if intfList is empty before scattering
if len(intfList) > 0:
    sc = ax.scatter(intfList[:, 1], intfList[:, 2], c=intfList[:, 0], cmap=cmap, s=5, label='INTF')
    cbar = fig.colorbar(sc, pad=0.1)
    cbar.ax.tick_params(labelsize=18)
    ax.text(azi_limits[1] + 5.5, 3, 'Microseconds', color='k', rotation=90, size=Nsize)
else:
    print("WARNING: No INTF points found in range!")

ax.set_xticks(azi_ticks)
ax.set_xlabel('Azimuth (deg)', fontsize=Nsize)
ax.set_ylim(elv_limits[0] - 1, elv_limits[1] + 1)
ax.set_yticks(elv_ticks)
ax.set_ylabel('Elevation (deg)', fontsize=Nsize)
ax.set_xlim(azi_limits[0] - 1, azi_limits[1] + 1)
ax.tick_params(axis='both', which='major', labelsize=Nsize)

ax.set_title("TGFs on " + date + ' ' + time, fontsize=Nsize)

seaxy = ax.secondary_yaxis('right', functions=(CtoF, FtoC))
seaxy.set_ylabel('Height (km)', fontsize=Nsize)
# seaxx = ax.secondary_xaxis('top', functions=(CxtoFx, FxtoCx))
# seaxx.set_xlabel('Distance (km)', fontsize=Nsize)
altitude = [0,1,2,3]
# distance = [-3, -2, -1, 0, 1, 2, 3, ]
# seaxx.set_xticklabels(distance, fontsize=Nsize)
# seaxy.set_yticklabels(altitude, fontsize=Nsize)

############# PLOTTING TRIGGERS ##########
# Removed specific "Trig A/B/C" hardcoded text as it applied to the 2021 event.
# The loop below will plot whatever triggers are found in the SD files.

print("Total Triggers found:", trigs)
trigNum = 0
colors = ['r', 'b', 'g', 'c', 'm', 'k']  # Color cycle for triggers

while trigNum < trigs:
    color = colors[trigNum % len(colors)]

    # Check if we have source data for this trigger
    if trigNum < len(sources):
        source_azi = sources[trigNum][0]
        source_elv = sources[trigNum][1]

        ax.scatter(trigAzis[trigNum], np.zeros(len(trigAzis[trigNum])), marker='o', facecolors='none', edgecolors=color,
                   s=150 * np.log(trigErgs[trigNum]))
        ax.plot((source_azi, cores[trigNum]), (source_elv, 0), c=color, label='TGF shower axis')
        ax.plot((source_azi, np.min(trigAzis[trigNum])), (source_elv, 0), linestyle='--', c=color)
        ax.plot((source_azi, np.max(trigAzis[trigNum])), (source_elv, 0), linestyle='--', c=color)
        ax.scatter(cores[trigNum], 0, marker='o', facecolors=color, edgecolors='k', s=250, label='TGF shower core')

        print(f"Trig {trigNum + 1}, Detectors: ", trigDets[trigNum])
        print("Opening Angle: ", np.max(trigAzis[trigNum]) - np.min(trigAzis[trigNum]))
    else:
        print(f"Skipping plot for Trig {trigNum + 1} - No source analysis found yet.")

    trigNum += 1

plt.show()