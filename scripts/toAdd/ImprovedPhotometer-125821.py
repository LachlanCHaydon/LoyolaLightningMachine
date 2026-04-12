
import re
import sys
import numpy as np
from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib.pyplot as plt
sys.path.append('./intf-tools')
import intf_tools as it
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
from matplotlib.patches import ConnectionPatch

# Example image paths and times (replace with actual data)
image_paths = ["Event 2/20230918-125821-627603-inverted-1.png", "Event 2/20230918-125821-630003-inverted-1.png", "Event 2/20230918-125821-633003-inverted-1.png", "Event 2/20230918-125821-636003-inverted-1.png", "Event 2/20230918-125821-639003-inverted-1.png","Event 2/20230918-125821-642003-inverted-1.png","Event 2/20230918-125821-645003-inverted-1.png", "Event 2/20230918-125821-648003-inverted-1.png", "Event 2/20230918-125821-651000-inverted-1.png"]
image_times = [627603, 630003, 633003, 636003, 639003, 642003, 645003, 648003, 651003 ]  # x positions for each image on time axis

filename = '20230918-125821.15101369_PXI_P19.dat'  # Change for each event


def calfunc(rsk, G, R, A, T): #this function is used to calibrate the data
	return 1/(rsk*G*R*A*T)

#All the numbers below from line 20 to 44 are numbers determined from manufacturer information used in the calibration formula
#Additonaly for each photometer wavelength the calibration factor is calculated and printed

R = 1 #load resistance in kohms

# 337 calibration
T1 = 0.325 #Filter transmittance
rsk1 = 60.000 #Cathode Radiance Sensitivity (RSk)
A1 = 907.46 * (10**(-6)) #effective area As
G111 = 	24484.37309 #final correct Gain value

cal1 = calfunc(rsk1, G111, R, A1, T1) * (10**6)
print(cal1)

# 391 calibration
T2 = 0.3125 #Filter transmittance
rsk2 = 80.000 #Cathode Radiance Sensitivity (RSk)
A2 = 907.46 * (10**(-6)) #effective area As
G222 = 	24484.37309 #final correct Gain value

cal2 = calfunc(rsk2, G222, R, A2, T2) * (10**6)
print(cal2)

# 777 calibration
T3 = 0.475 #Filter transmittance
rsk3 = 14.531 #Cathode Radiance Sensitivity (RSk)
A3 = 586.7 * (10**(-6)) #effective area As
G333 = 	67145.67 #final correct Gain value

cal3 = calfunc(rsk3, G333, R, A3, T3) * (10**6)
print(cal3)

# Extract microseconds
microseconds = int(filename.split('.')[1].split('_')[0])

# Extract the date and time
date = filename.split('.')[0]

time_string = filename.split('-')[1].split('.')[0]
formatted_time = f"{time_string[:2]}:{time_string[2:4]}:{time_string[4:]}" #used to clearly write out eh time in pot title
print(formatted_time)

#Create variable for time spacing in time array based on samling rate
f = 30e6
Rec_length = int(30e6)  # Ensure Rec_length is an integer
dt = 1 / f

# Create the time array t to use to plot photometer data
t = np.arange(1, Rec_length + 1) * dt

# Change t to align with plot
t1 = t * 1e6 + microseconds*(10**(-2))

t_min = 626000
t_max = 652000

mask = (t1 >= t_min) & (t1 <= t_max)
t1_select = t1[mask]


# Read the CSV file, skipping the first 8 rows
FA_file = 'TAR_20230918_125821_634615_T0.csv' # Change file name for different events
M = pd.read_csv(FA_file, skiprows=8).to_numpy()



# Extracting the time difference for the FA data
with open(FA_file, 'r') as file:
	lines = file.readlines()

number = None
for line in lines:
	if line.startswith('#'):
	# if line.startswith('# T0[sec]:'):
		# Extract decimal part from the float
		# pattern = r'(\d+)\.(\d+)'  # match the float and split it
		pattern = r'\.\s*(\d+)\.' # old pattern
		match = re.search(pattern, line)
		if match:
			# number = match.group(2)  #get part after decimal
			number = match.group(1) # old number
			break
# number = 698806.817
number = int(number)
# number = number*(10**(-3))
print(number)

# Create t2 and E arrays from the FA data
t2 = M[:, 0] * 1000 + number
E = M[:, 1]

mask2 = (t2 >= t_min) & (t2 <= t_max)
t2_select = t2[mask2]
E_select = E[mask2]

# Read photometer data from binary file and calibrate it
with open(filename, 'rb') as fid:


	ch0 = np.fromfile(fid, dtype=np.float64, count=Rec_length)
	ch0 = -cal1 * (ch0 - ch0[5])

	ch1 = np.fromfile(fid, dtype=np.float64, count=Rec_length)
	ch1 = -cal2*(ch1 - ch1[5])

	ch2 = np.fromfile(fid, dtype=np.float64, count=Rec_length)
	ch2 = -cal3*(ch2 - ch2[5])

	ch3 = np.fromfile(fid, dtype=np.float64, count=Rec_length)
	# ch3 = cal3*(ch3)+30000


# set INTF trigger as number from FA data file
manualIntfTrig = number
##### Read INTF points #####
def READ_INTF(intfFile, manualIntfTrig):
	# if scribbleData: # check if scribble data format
	#     intfList=np.genfromtxt(intfFile,skip_header=2,usecols=(0,1,2,3,4,5))
	#     for item in intfList:
	#         item[0]=(item[0]%1)*(10**6)
	# else:
	# ff=open(intfFile,'r')
	# for line in ff: # get header data
	#     if line.startswith('#uSecond'):
	# 		intfTrig = float(line.split(':')[1])
	#     if line.startswith('#Time'):
	#         break
	# ff.close()
	intfList=np.genfromtxt(intfFile,skip_header=44,usecols=(0,1,2,3,4,5))
	intfList[:,0] *= 1000.
	intfTrig = manualIntfTrig
	intfList[:,0] += intfTrig
	return intfList

intfFile = "TAR_2023.09.18_12-58-21_634615_pix200_S256-I64-P4_W3.dat" # Replace with correct INTF file name
inft_read = READ_INTF(intfFile,manualIntfTrig) ############### Read INTF points #####
inft_cuts = True
inftMin,inftMax = (0,35)
if inft_cuts:
	inft_read = np.delete(inft_read, np.where(inft_read[:, 2] < inftMin)[0], 0)
	inft_read = np.delete(inft_read, np.where(inft_read[:, 2] > inftMax)[0], 0)


sLevelsTup = (1.0, 3., 7., 16.)   # S-ratio levels -and- alpha values between them
alphaTup   = (0.3, 0.7, 1.0)
#Store all the differnt elements of the INTF data
intf_time = inft_read[:,0]
intf_azi = inft_read[:,1]
intf_tasd = inft_read[:,2]
intf_pkpk = inft_read[:,5]

N=len(intf_pkpk)
ss = np.log10( intf_pkpk )
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
ss[ss>1] = 1
color = it.cmap_mjet( ss )

intf_time1,intf_tasd1, intf_azi1,color1,markerSz1=([[] for i in range(4)] for ii in range(5)) #create array with all INTF data
for itran in range( len(intf_time)-1 ):
	if ( s[itran]<=sLevelsTup[1] ):
		intf_time1[0].append(intf_time[itran])
		intf_tasd1[0].append(intf_tasd[itran])
		intf_azi1[0].append(intf_azi[itran])
		color1[0].append(color[itran])
		markerSz1[0].append(markerSz[itran])
	elif ( (s[itran]>sLevelsTup[1]) & (s[itran]<=sLevelsTup[2]) ):
		intf_time1[1].append(intf_time[itran])
		intf_tasd1[1].append(intf_tasd[itran])
		intf_azi1[0].append(intf_azi[itran])
		color1[1].append(color[itran])
		markerSz1[1].append(markerSz[itran])
	elif ( s[itran]>sLevelsTup[2] ):
		intf_time1[2].append(intf_time[itran])
		intf_tasd1[2].append(intf_tasd[itran])
		intf_azi1[2].append(intf_azi[itran])
		color1[2].append(color[itran])
		markerSz1[2].append(markerSz[itran])



# Plotting

fig, (ax1, ax22) = plt.subplots(nrows=2, ncols=1, figsize=(15, 10), gridspec_kw={'height_ratios': [4, 1]})


ax3 = ax1.twinx()# use the twin function to make multiple y axis to store the different data types

ch0_select = ch0[mask]
ch1_select = ch1[mask]
ch2_select = ch2[mask]

#photometer plotting
ax1.plot(t1_select, ch0_select, color='blue', label='337 nm')
ax1.plot(t1_select, ch1_select, color='purple', label='391 nm')
ax1.plot(t1_select, ch2_select, color='red', label='777 nm')
ax1.set_ylabel('Irradiance ($\mu$W/$m^2$)', color='black',fontsize=15)
ax1.tick_params(axis='y', labelcolor='black')
ax1.tick_params(labelsize=14)
ax1.set_xlabel('Time after ' + formatted_time + '($\mu$s)', fontsize=14)

plt.gca().xaxis.get_major_formatter().set_scientific(False) # Keep numbers withouth scientific notation
plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}')) # Make sure the full number is shown
num_ticks = 15  # Change this number to add more or less ticks
ax1.xaxis.set_major_locator(plt.MaxNLocator(num_ticks))

# Create a second y-axis for the FA data
ax2 = ax1.twinx()
ax2.plot(t2_select, E_select, color='green', label='FA')
ax2.set_ylabel('Fast Antenna (V/m)', color='green', fontsize=15)
ax2.set_ylim([-25, 5]) #use for limits on FA data if needed
ax2.tick_params(axis='y', labelcolor='black')
ax2.tick_params(labelsize=14)

#Plot the INTF data
ax3.spines['right'].set_position(('outward', 60))  # Offset the third y-axis
ax3.set_frame_on(True)
ax3.patch.set_visible(False)
for sp in ax3.spines.values():
	sp.set_visible(False)
ax3.spines['right'].set_visible(True)

# mask3 = (intf_time1 >= t_min) & (intf_time1 <= t_max)
# t2_select = t2[mask2]
# E_select = E[mask2]

for iS in range( len(sLevelsTup)-1 ):
	intf_time_arr = np.array(intf_time1[iS])
	intf_tasd_arr = np.array(intf_tasd1[iS])
	markerSz_arr = np.array(markerSz1[iS])
	color_arr = np.array(color1[iS])

	mask3 = (intf_time_arr >= t_min) & (intf_time_arr <= t_max)
	intf_t = intf_time_arr[mask3]
	intf = intf_tasd_arr[mask3]
	markerSz_sel = markerSz_arr[mask3]
	color_sel = color_arr[mask3]
	if (iS==2): # added to fix legend
		ax3.scatter(intf_t,intf,label='INTF',s=markerSz_sel,facecolor=color_sel,alpha=alphaTup[iS],edgecolors='k',zorder=1)
	else:
		ax3.scatter(intf_t,intf,s=markerSz_sel,facecolor=color_sel,alpha=alphaTup[iS],edgecolors='k',zorder=1)

Nsize = 15 #size of font for INTF y label
ax3.set_ylabel("INTF elevation (deg)", fontsize = Nsize)
ax3.yaxis.label.set_color('r')
ax3.tick_params(axis='y', colors='r')
[i.set_color("r") for i in ax1.yaxis.get_ticklines()]
ax3.tick_params(labelsize=14)


# Add grid and legend
ax1.grid(True)


# Combine legends from all y-axes
lines, labels = ax3.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax1.get_legend_handles_labels()
lines4 = lines3+lines2+lines
labels4 = labels3+labels2+labels
ax1.legend(lines4,labels4, fontsize=11,loc='best')

# Set x label and plot title
# plt.xlabel('Time after ' + formatted_time + '($\mu$s)', fontsize=14)
# plt.title('Photometer data for ' + date, fontsize=14)

# Clean up ax2 for image display
ax22.set_xlim(ax1.get_xlim())
ax22.set_ylim(0, 1)
ax22.axis('off')  # Turn off axis lines and labels

# === Add images on ax2 ===
# Number of images
N_images = len(image_paths)

# Evenly spaced x positions across the x-axis range of ax22
x_positions = np.linspace(t_min, t_max, N_images)

for img_path, xpos, xpos1 in zip(image_paths, x_positions, image_times):
	try:
		img = mpimg.imread(img_path)
		imagebox = OffsetImage(img, zoom=0.15)
		ab = AnnotationBbox(imagebox, (xpos, 0.5),
							xycoords='data',
							frameon=False,
							box_alignment=(0.5, 0.5))
		ax22.add_artist(ab)

		# Add time labels under the image
		ax22.text(xpos, 0.0, f'{int(xpos1)} µs', ha='center', va='top', fontsize=12)

		# Connect image in ax22 to x-axis of ax1 using ConnectionPatch
		# Start point in ax22 coords (xpos, 0.5)
		# End point in ax1 coords (xpos, y=0, the x-axis line)
		con = ConnectionPatch(
			xyA=(xpos, 0.951), coordsA=ax22.transData,
			xyB=(xpos1, -1410), coordsB=ax1.transData,
			arrowstyle='-', linestyle=':', color='black', linewidth=0.8)
		fig.add_artist(con)

	except FileNotFoundError:
		print(f"Image not found: {img_path}")


plt.tight_layout()
plt.show()



