#!/usr/bin/env python

import sys
import os
import numpy as np
from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib.image as mpimg
import matplotlib
import matplotlib.pyplot as plt
from decimal import Decimal
import datetime
sys.path.append('./intf-tools')
import intf_tools as it
import pandas as pd
from READ_TASD_microseconds import READ_INTF, READ_TASD_BURST


bg1 = np.zeros((20,320, 800))
for i in range (0, 20):
    m = '{:03d}'.format(i+30)
    bg1[i,:,:]= mpimg.imread("./../Cameras/stroke1_31/Img000"+str(m)+'.tif')
bg1_mean = np.mean(bg1, axis=0)

bg2 = np.zeros((20,320, 800))
for i in range (0, 20):
    m = '{:03d}'.format(i)
    bg2[i,:,:]= mpimg.imread("./../Cameras/stroke5_67/Img000"+str(m)+'.tif')
bg2_mean = np.mean(bg2, axis=0)

frame3 = [20]
Img3_file = "./../Cameras/stroke3_16/Img0000"+str(frame3[0])+".tif"
img3=mpimg.imread(Img3_file)
img3 = img3-bg2_mean
start3_time = 772874.03    #### micro seconds

frame4 = [19]
Img4_file = "./../Cameras/stroke4_15/Img0000"+str(frame4[0])+".tif"
img4=mpimg.imread(Img4_file)
img4 = img4-bg2_mean
start4_time = 772874.03    #### micro seconds

frame5 = [55]
Img5_file = "./../Cameras/stroke5_67/Img0000"+str(frame5[0])+".tif"
img5=mpimg.imread(Img5_file)
img5=img5-bg2_mean
start5_time = 772874.03    #### micro seconds


########## CHANGE INFORMATION HERE AND FROM LINE 130 ###############
# set trigger time manually:
date = '240817'
time = '16:18:44'
plotall = True
SD_limits =(0,500)

manualIntfTrig = 730069
manualFATrig = 730069
fa_filename = './../FA_INTF_LMA/TAR_20240817_161844_736616_T0/TAR_20240817_161844_736616_T0.csv'
fa = pd.read_csv(fa_filename, skiprows = [0,1,2,3,4,5])

# intfFile = "./../FA_INTF_LMA/INTF_FA_2024.08.17_16-18-44_736616/TAR_2024.08.17_16-18-44_736616_pix200_S256-I64-P4_W3_calibrated.dat"
intfFile = "./../FA_INTF_LMA/INTF_FA_2024.08.17_16-18-44_736616/TAR_intf_240817_161845_cut_3strokes_new.dat"
intfList = np.genfromtxt(intfFile, skip_header=57, usecols=(0, 1,2,3,4,5))
elevation_cuts = True ################
azimuth_cuts = True
time_cuts = True
elvMin,elvMax = (-5,40) ##### certain range of INTF data
aziMin, aziMax = (270, 330)
timeMin, timeMax = (782960, 842450)
print (timeMin, timeMax)
if time_cuts == False:
    intfList = np.delete(intfList, np.where(intfList[:, 0] < timeMin)[0], 0)
    inftList = np.delete(intfList, np.where(intfList[:, 0] > timeMax)[0], 0)
if elevation_cuts == True:
    intfList = np.delete(intfList, np.where(intfList[:, 2] < elvMin)[0], 0)
    inftList = np.delete(intfList, np.where(intfList[:, 2] > elvMax)[0], 0)
if azimuth_cuts == True:
    intfList = np.delete(intfList, np.where(intfList[:, 1] < aziMin)[0], 0)
    intfList = np.delete(intfList, np.where(intfList[:, 1] > aziMax)[0], 0)

############### intf data read #############
sLevelsTup = (1.0, 3., 7., 16.)   # S-ratio levels -and- alpha values between them
alphaTup   = (0.3, 0.7, 1.0)
intf_time = intfList[:,0]
intf_azi = intfList[:,1]
intf_elv = intfList[:,2]
intf_pkpk = intfList[:,5]

print (np.min(intfList[:,0]), np.max(intfList[:,0]))
# print (intf_time[0], intf_time[-1])
N=len(intf_pkpk)
ss = np.log10( intf_pkpk )
# print (ss)
aMin = np.min( ss )
aMax = np.max( ss )
ss = (ss-aMin)/(aMax-aMin)
ss[ss>1] = 1
s = (1 + 3*ss**2)**2
markerSz = 6*s

a05 = np.log10(sorted(intf_pkpk)[int(1*N/20)])
a95 = np.log10(sorted(intf_pkpk)[int(19*N/20)])
ss = np.log10( intf_pkpk )
ss = ss/aMax
#ss = (ss-a05)/(a95-a05)
ss[ss>1] = 1
color = it.cmap_mjet( ss )
intf_time1,intf_elv1, intf_azi1,color1,markerSz1=([[] for i in range(4)] for ii in range(5))
for itran in range( len(intf_time)-1 ):
	if ( s[itran]<=sLevelsTup[1] ):
		intf_time1[0].append(intf_time[itran])
		intf_elv1[0].append(intf_elv[itran])
		intf_azi1[0].append(intf_azi[itran])
		color1[0].append(color[itran])
		markerSz1[0].append(markerSz[itran])
	elif ( (s[itran]>sLevelsTup[1]) & (s[itran]<=sLevelsTup[2]) ):
		intf_time1[1].append(intf_time[itran])
		intf_elv1[1].append(intf_elv[itran])
		intf_azi1[0].append(intf_azi[itran])
		color1[1].append(color[itran])
		markerSz1[1].append(markerSz[itran])
	elif ( s[itran]>sLevelsTup[2] ):
		intf_time1[2].append(intf_time[itran])
		intf_elv1[2].append(intf_elv[itran])
		intf_azi1[2].append(intf_azi[itran])
		color1[2].append(color[itran])
		markerSz1[2].append(markerSz[itran])


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

##################### TASD data ####################

sdStartTime = 19136. ### Time of the first burst
TASD_shift = 36.7
dirname = "./../SD_20240817_161845/"  ### CHOOSE THE DETECTOR WITH HIGHEST ENERGY

###return allmicro, alllSignal, alluSignal, detectors
allmicro, alllSignal, alluSignal, alldec = ([] for ii in range(4))
sds = READ_TASD_BURST(dirname,date, TASD_shift, allmicro, alllSignal, alluSignal, alldec)
allmicro = sds[0]
alllSignal = sds[1]
alluSignal = sds[2]
alldec = sds[3]

######## PLOT FIGURE 3 ##########
# plt.figure(figsize=[15,15])
fig = plt.figure(figsize=(15, 8))
plt.subplots_adjust(right=0.98, left=0.05, bottom = 0.05, top = 0.9,hspace=0.3,wspace=0.15  )
host1 = host_subplot(221)
host2 = host_subplot(222)
host3 = host_subplot(223)
host4 = host_subplot(224)
Nsize = 20
host1.set_title("B)",weight="bold",fontsize=Nsize)
host2.set_title("A')",weight="bold",fontsize=Nsize)
host3.set_title("C)",weight="bold",fontsize=Nsize)
host4.set_title("D)",weight="bold",fontsize=Nsize)
N1size=16
host1.imshow(img3)
host1.text(400,50, "Stroke 3 - 16.0 kA", fontsize = 16, color="white", weight = "bold")
host1.set_xlabel("Horizontal pixels", fontsize = N1size)
host1.set_ylabel("Vertical pixels", fontsize = N1size)
host3.imshow(img4)
host3.text(400,50, "Stroke 4 - 15.9 kA", fontsize = 16, color="white", weight = "bold")
host3.set_xlabel("Horizontal pixels", fontsize = N1size)
host3.set_ylabel("Vertical pixels", fontsize = N1size)
host4.imshow(img5)
host4.text(400,50, "Stroke 5 - 67.3 kA", fontsize = 16, color="white", weight = "bold")
host4.set_xlabel("Horizontal pixels", fontsize = N1size)
host4.set_ylabel("Vertical pixels", fontsize = N1size)

for iS in range( len(sLevelsTup)-1 ):
	if (iS==2): # added to fix legend
		host2.scatter(intf_time1[iS],intf_elv1[iS],label='INTF',s=markerSz1[iS],facecolor=color1[iS],alpha=alphaTup[iS],edgecolors='k',zorder=1)
	else:
		host2.scatter(intf_time1[iS],intf_elv1[iS],s=markerSz1[iS],facecolor=color1[iS],alpha=alphaTup[iS],edgecolors='k',zorder=1)

xstart0,xstop0 = 1018900,1019900  ## range from the time TGFs begin til the return stroke happen
tick_freq0=5000
elvMin0, elvMax0 = (-5,40)
xticks0=range(xstart0,xstop0+tick_freq0,tick_freq0)
FAlimit0 = (-100,30)
host2.set_xlim(xstart0,xstop0)
host2.set_ylim(elvMin0, elvMax0)


host2.set_xlabel("Microseconds (µs)", fontsize = N1size)
# host2.set_ylabel("INTF elevation (deg)", fontsize = Nsize)
host2.axis["bottom"].label.set_fontsize(Nsize)
host2.axis["left"].label.set_fontsize(Nsize)
host2.axis["left"].label.set_color('k')
[i.set_color("k") for i in host2.yaxis.get_ticklines()]
host2.tick_params(labelsize=Nsize)
host2.tick_params(axis='y', colors='k')
################## FA PLOT  ###########################
par1 = host2.twinx()
par1.plot(fa['Time [ms]']*1000+ manualFATrig, fa['E [V/m]'], color='g', linewidth=1.5, label='Fast Antenna', zorder=2)
par1.set_ylabel("Fast Antenna (V/m)", fontsize= Nsize, color = 'g')
par1.set_ylim(FAlimit0)
par1.tick_params(labelsize=Nsize)
par1.axes.get_yaxis().set_visible(False)
par1.tick_params(axis='y', colors='g')

################ TASD plot ###################
par2 = host2.twinx()
if plotall == True:
	for i in range(0, len(allmicro)):
		par2.plot(allmicro[i], alluSignal[i], color = "m")
	par2.plot(allmicro[0], alluSignal[0], color="m",label="TASD waveform")
else:
	par2.plot(allmicro[3], alluSignal[3], label= "TASD "+alldec[3], color = "m")

par2.set_ylim(SD_limits)
par2.set_ylabel("FADC count", fontsize = Nsize)
par2.tick_params(labelsize=Nsize, color = "m")
par2.axes.get_yaxis().set_visible(False)
par2.spines["right"].set_position(("axes", 1.14))
host2.grid(linestyle=':')
host2.yaxis.label.set_color('r')
par1.yaxis.label.set_color("g")
par2.yaxis.label.set_color("m")
# host2.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5), fontsize = Nsize-5)
# host2.legend(loc=3, bbox_to_anchor=(0.5, 0., 0.5, 0.5), fontsize = Nsize-5)
host2.tick_params(axis='y', colors='r')
par1.tick_params(axis='y', colors='g')
par2.tick_params(axis='y', colors='m')

plt.savefig("./../REWRITE_PAPER/Figure_2B.png")
plt.show()



