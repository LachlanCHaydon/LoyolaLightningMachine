#!/usr/bin/env python

import os
import sys
import heapq
import numpy as np
import geopy as gp
import geopy.distance
import scipy.constants as sp
import matplotlib.pyplot as plt
from READ_calc_iterative_2024 import READ_INTF, READ_TASD, READ_LMA

############## CHANGE INFORMATION FOR DIFFERENT TGFs (14-32) AND LINE 132, 152

trigNum = 1# which trigger within the TGF? (1,2,3, etc)

savefile = False
if savefile:
	stdoutOrigin=sys.stdout
	sys.stdout = open("TASD/SD_0921_025353/Analysis"+ str(trigNum) + "_Feb2026.txt", "w") #### write the output in a file

print ('TRIGGER ', trigNum)
sdStartTime = 325108.### time of the first burst
hour,minute,second = 2,53,53 ##### Changes for different TGFs
time = "025353"

intfFile = 'INTF/TAR_intf_250921_025353_azi_cutFeb10.dat'
intfList = READ_INTF(intfFile) ############### Read INTF points #####

lmaFile = 'LMA/LMA_250921_025353_Febcut.txt'
lmaList=READ_LMA(lmaFile) ##### Read LMA points #####
# print (lmaList[:,0])

manualStart = True # use to manually ignore TASD data before this time
sds,detectors,t0,t0err,varList,zOld,maxList=([] for i in range(7))
sdDir = "TASD/SD_0921_025353/" ##### Read TASD data #####
sds = READ_TASD(sdDir, time, trigNum, manualStart, sdStartTime,sds) ### Read tasd data ####

tempLMA_cut = 39.5
tempIntf_cut_max = 11
tempIntf_cut_min = 1.0
tempAzi_cut_max = 302
tempAzi_cut_min = 240.
intf_plot = False
############# SAME INFORMATION FOR ALL TGFs ############
scribbleData = True #### for INTF data, we used only the scribbleData
c = sp.c * (10**-9)   # kilometers/microsecond
rEarth = 6371 + 1.4   # kilometers

manualTrig = False
sdTimingErr = 0.04 # microseconds
intfRange = 9.4 ### microseconds - range of inft data
initialAlt = 3. # initial guess in km
itLimit = 10 # Give up after this many iterations
gpsConversion = True # 'True' for spherical earth calculations, 'False' to use geopy
intfLat = 39.339082
intfLon = -112.700696
intfGps = gp.point.Point(intfLat,intfLon,1.4) # intf antenna A
lmaRange = 200 #### microseconds
gpscoors='tasd_gpscoors.txt'
year,month,day = 2024,8, 17
date = year+month+day

# ##################################################################################
# ##### Main calculations of height, r1, r2, and final shift for each detector #####
# ##### Error calculations are based on general form (dq=sqrt((dx1*dq/dx1)^2+...+(dxn*dq/dxn)^2)) #####
# # first guess: #### THE SAME FOR ALL TGFs
ii=0
#sds = np.array(sds)
for item in sds:
	tempLma = []
	tempTrig =hour*3600 + minute*60 + second + (item[-1])*(10**-6) # individual sd trigger time
	print ('TEMP TRIG 0', tempTrig)
	# pick out LMA points within range specified in manual settings:
	for item2 in lmaList: # check timing first
		diff = (tempTrig - item2[0] )
		if diff < (lmaRange*(10**-6)):
			tempLma.append(item2)
	# print (tempLma)
	tempLma = np.array(tempLma)
	lmaGps=gp.point.Point(np.mean(tempLma[:,1]),np.mean(tempLma[:,2]),1.4)
	sdGps=gp.point.Point(item[4],item[5],1.4)
	x1=gp.distance.distance(lmaGps,intfGps).km # horizontal distance D
	x2=gp.distance.distance(lmaGps,sdGps).km # horizontal distance d

	zOld.append(initialAlt) # first guess of altitude z
	diff=np.sqrt((x1)**2+(zOld[ii])**2) - np.sqrt((x2)**2+(zOld[ii])**2) # path length difference of (lma-intf)-(lma-sd) or (R-r)
	#print(' '+item[3]+': '+str(diff/c)+' us')
	t0.append(diff/c) # difference in propagation time (R-r)/c
	t0err.append(0) # initialize error list
	ii+=1
	# print ("########### ii", ii)

print ('First estimation =', np.mean(t0) )
# ###############################################
# ##### Now full iteration until consistent #####
# ###### there are 2 places needed to be changed for different TGFs which are LMA and INTF removing data
success=0
count=1
while success==0:
	ii=0
	dt,dterr,tempVar=([] for i in range(3))
	while ii < len(t0): # repeat for all detectors
		tempLma,tempIntf=([] for i in range(2))
		tempTrig = float(time[0:2])*3600 + float(time[2:4])*60 + float(time[4:6]) + ((sds[ii][-1])*(10**-6)) # individual sdTrig (s units)
		# print ('TEMP TRIG LMA ', tempTrig)
		tempTc = t0[ii] + sds[ii][-1] # shift individual sdTrig to INTF reference (us units)

		if gpsConversion: # for manual distance calculations
			sdX = (rEarth)*((sds[ii][5]*np.pi/180) - (intfLon*np.pi/180))*np.cos((sds[ii][4]+intfLat)*np.pi/360)
			sdY = (rEarth)*((sds[ii][4]*np.pi/180) - (intfLat*np.pi/180))
		else:
			sdGps = gp.point.Point(sds[ii][4],sds[ii][5],sds[ii][6])

		for item in lmaList: # remove LMA points outside of lmaRange or with high chi2
			if (item[0]<tempTrig-(lmaRange*10**-6)) or (item[0]>tempTrig+(lmaRange*10**-6)):
				continue
			else:
				tempLma.append(item)

		tempLma = np.array(tempLma)
		# print ('AFTER REMOVING ', len(tempLma))
		#print(np.mean(tempLma[:,2]))
		#print(np.mean(tempLma[:,2]) + (outSig*np.std(tempLma[:,2])))
		#ave1=np.mean(tempLma[:,1]) # lat average
		#sig1=np.std(tempLma[:,1])  # lat std dev
		#ave2=np.mean(tempLma[:,2]) # lon average
		#sig2=np.std(tempLma[:,2])  # lon std dev
		# ave3=np.mean(tempLma[:,3]) # alt average
		# sig3=np.std(tempLma[:,3])  # alt std dev
###############################################3
############################################3
###############################################
		# get new LMA coords
		# print(tempLma[:, 0])
		# print(tempLma[:, 1])
		# print(tempLma[:, 2])
		tempLma = np.array(tempLma)
		tempLma = np.delete(tempLma, np.where(tempLma[:, 1] > tempLMA_cut)[0], 0)   #### removing the LMA point
        ######## plot the figure bellow to check these data
		lmaLat=np.mean(tempLma[:,1])
		lmaLaterr=np.std(tempLma[:,1])/np.sqrt(len(tempLma))
		lmaLon=np.mean(tempLma[:,2])
		lmaLonerr=np.std(tempLma[:,2])/np.sqrt(len(tempLma))
		# plt.figure()
		# plt.title(str(ii))
		# plt.scatter(tempLma[:, 1], tempLma[:, 2])
		# plt.show()
		# remove entries outside of intfRange
		for item in intfList:
			if (item[0] < (tempTc - intfRange)) or (item[0] > (tempTc + intfRange)):
				continue
			else:
				tempIntf.append(item)
		# print('\nnumber of INTF points within +/- '+str(intfRange)+' us: '+str(len(tempIntf)))
		##		print(len(tempIntf))
		tempIntf = np.array(tempIntf)
		tempIntf = np.delete(tempIntf, np.where(tempIntf[:, 2] > tempIntf_cut_max)[0], 0)   #### removing the INTF point - different TGFs
		tempIntf = np.delete(tempIntf, np.where(tempIntf[:, 2] < tempIntf_cut_min)[0], 0)
		tempIntf = np.delete(tempIntf, np.where(tempIntf[:, 1] > tempAzi_cut_max)[0], 0)   #### removing the INTF point - different TGFs
		tempIntf = np.delete(tempIntf, np.where(tempIntf[:, 1] < tempAzi_cut_min)[0], 0)

		#######   plot the figure bellow to check the removing INTF points
		if intf_plot:
			plt.figure()
			plt.title(str(ii))
			fig = plt.scatter(tempIntf[:, 1], tempIntf[:, 2], c = tempIntf[:,0])
			cbar = plt.colorbar(fig, pad=0.1, cmap="jet")
			plt.show()

		if gpsConversion:
			lmaX = (rEarth) * ((lmaLon * np.pi / 180) - (intfLon * np.pi / 180)) * np.cos((intfLat) * np.pi / 180)
			lmaY = (rEarth) * ((lmaLat * np.pi / 180) - (intfLat * np.pi / 180))
			lmaXerr = (lmaLonerr * np.pi / 180) * np.cos(np.pi * lmaLat / 180) * (rEarth)
			lmaYerr = (lmaLaterr * np.pi / 180) * rEarth
		# print('lmaX = '+str(lmaX)+' +/- '+str(lmaXerr))
		# print('lmaY = '+str(lmaY)+' +/- '+str(lmaYerr))
		else:
			lmaGps = gp.point.Point(lmaLat, lmaLon, 1.4)
			# determine direction before finding lmaErr:
			if np.mean(tempIntf[:, 1]) > 250:
				lmaErr = gp.point.Point(lmaLat + lmaLaterr, lmaLon - lmaLonerr, 1.4)
			else:
				lmaErr = gp.point.Point(lmaLat - lmaLaterr, lmaLon - lmaLonerr, 1.4)

		# elevation from INTF points:
		elv = np.mean(tempIntf[:, 2])
		elverr = np.std(tempIntf[:, 2]) / np.sqrt(len(tempIntf))

		# azimuth from INTF points:
		azi = np.mean(tempIntf[:, 1])
		azierr = np.std(tempIntf[:, 1]) / np.sqrt(len(tempIntf))

		# horizontal distances:
		if gpsConversion:
			x1 = np.sqrt((lmaX)**2+(lmaY)**2)
			x1err = np.sqrt((lmaX*lmaXerr)**2+(lmaY*lmaYerr)**2)/x1
			x2 = np.sqrt((sdX-lmaX)**2+(sdY-lmaY)**2)
			x2err = np.sqrt(((lmaX-sdX)*lmaXerr)**2+((lmaY-sdY)*lmaYerr)**2)/x2
		else:
			x1 = gp.distance.distance(lmaGps,intfGps).km
			x1err = np.abs(gp.distance.distance(lmaErr,intfGps).km - x1)
			x2 = gp.distance.distance(lmaGps,sdGps).km
			x2err = np.abs(gp.distance.distance(lmaErr,sdGps).km - x2)

		# altitude calculation
		z = x1*np.tan(elv*np.pi/180.)
		# print("x1, elv, z", x1, elv, z)
		zerr = np.sqrt((np.tan(np.pi*elv/180.)*x1err)**2+(x1*(np.pi*elverr/180.)*(np.cos(np.pi*elv/180.))**-2)**2)
		# print('z = '+str(z)+' +/- '+str(zerr))

		# slant distances:
		r1 = np.sqrt(x1**2 + (z)**2) # source to intf
		r1err = np.sqrt((x1*x1err)**2+((z)*zerr)**2)/r1

		r2 = np.sqrt(x2**2 + (z-(sds[ii][6]-1.4))**2) # source to tasd
		if gpsConversion:
			dotp=((sdX-lmaX)*-lmaX)+((sdY-lmaY)*-lmaY)
			r2err = np.sqrt((x1*x1err*((np.tan(np.pi*elv/180.))**2))**2+((np.pi*elverr/180.)*(np.tan(np.pi*elv/180.))*(x1/np.cos(np.pi*elv/180.))**2)**2+(x2*x2err)**2+(2*dotp*x1err*x2err*(np.tan(np.pi*elv/180.))**2))/r2
		else:
			r2err = np.sqrt((x2*x2err)**2+((z-(sds[ii][6]-1.4))*zerr)**2)/r2

		# time of source onset
		ta = tempTc-(r1/c)
		taerr = np.sqrt(((r1err/c)**2)+((t0err[ii])**2)+((sdTimingErr)**2))
		#print('t_a='+str(ta))

		# time of source onset shifted to INTF frame
		tc = tempTc
		tcerr = np.sqrt(((taerr)**2)+((r2err/c)**2))
		#print(str(sds[ii][3])+': ta_err = '+str(taerr)+', tc_err = '+str(tcerr))

		# propagation time difference
		dt.append((r1-r2)/c)
		if gpsConversion:
			term1 = (r1/(x1*c)) - ((x1*(np.tan(np.pi*elv/180.))**2)/(r2*c))
			term2 = -x2/(r2*c)
			term3 = ((x1**2)/c)*((1/r1)-(1/r2))*(np.tan(np.pi*elv/180.)/((np.cos(np.pi*elv/180.))**2))
			cosAlpha = dotp/(x1*x2)
			dterr.append(np.sqrt((term1*x1err)**2+(term3*np.pi*elverr/180.)**2+(term2*x2err)**2+(2*term1*term2*cosAlpha*x1err*x2err)))
		else:
			dterr.append(np.sqrt((r1err/c)**2+(r2err/c)**2))

		# add results to data array:
		if gpsConversion:
			tempVar.append([elv,elverr,x1,x1err,z,zerr,r1,r1err,x2,x2err,r2,r2err,sds[ii][6],sds[ii][7],dt[-1],dterr[-1],ta,taerr,tc,tcerr,lmaX,lmaXerr,lmaY,lmaYerr,sdX,sdY,sds[ii][0][0],lmaLat,lmaLon])
		else:
			tempVar.append([elv,elverr,x1,x1err,z,zerr,r1,r1err,x2,x2err,r2,r2err,sds[ii][6],sds[ii][7],dt[-1],dterr[-1],ta,taerr])

		ii+=1

	# finished with all SDs
	# test consistency with previous iteration
	tempVar2=np.array(tempVar)
	# print(tempVar2[:,2])
	zOld=np.array(zOld)
	if np.array_equal(np.round(tempVar2[:,4],4),np.round(zOld,4)):
		success=1
		print('convergence!')
		if count==1:
			print('  Warning: onset time may be misreported on first iteration!')
		print('iterations = '+str(count))
		break
	else:
		if count==itLimit:
			print('consistency failed ('+str(itLimit)+' attempts)')
			print(zOld)
			print(tempVar2[:,4])
			break
		# no convergence - use results as new initial guess
		zOld = tempVar2[:,4]
		t0=dt
		t0err=dterr
		count+=1

# print ('Altitude of LMA ',tempLma[:, 0])
# print ("LMA", tempLma[:,3])
print('\nnumber of LMA points within +/- '+str(lmaRange)+' us: '+str(len(tempLma)))
print('number of INTF points within +/- '+str(intfRange)+' us: '+str(len(tempIntf[:,0])))
print(tempLma[:,0])
print(tempLma[:,3])
print('Average altitude Intf =', 11*np.tan(np.mean(tempIntf[:, 1])))
print('Average altitude LMA =', (np.mean(tempLma[:, 3])))
# print(lmaLaterr)
# print(lmaLonerr)
# print(len(tempIntf))
# print('elv,elverr,x1,x1err,z,zerr,r1,r1err,x2,x2err,r2,r2err,sds[ii][6],sds[ii][7],dt[-1],dterr[-1],ta,taerr,lmaX,lmaXerr,lmaY,lmaYerr,sdX,sdY,sdTrigger')
# print(tempVar2)

print('\nz median = '+str(np.median(tempVar2[:,4]))+' +/- '+str(np.median(tempVar2[:,5])))
print('ta median= '+str(np.median(tempVar2[:,16]))+' +/- '+str(np.median(tempVar2[:,17])))
print('tc median= '+str(np.median(tempVar2[:,18]))+' +/- '+str(np.median(tempVar2[:,19])))
print('elv median= '+str(np.median(tempVar2[:,0]))+' +/- '+str(np.median(tempVar2[:,1])))

print('\nz mean = '+str(np.mean(tempVar2[:,4]))+' +/- '+str(np.mean(tempVar2[:,5])))
print('ta mean= '+str(np.mean(tempVar2[:,16]))+' +/- '+str(np.mean(tempVar2[:,17])))
print('tc mean= '+str(np.mean(tempVar2[:,18]))+' +/- '+str(np.mean(tempVar2[:,19])))

maxList = []
for item in heapq.nlargest(2,tempVar2[:,13]):
		maxList.append(tempVar2[np.where(tempVar2[:,13]==item)])
maxList=np.array(maxList)
print('\nz 2high = '+str(np.mean(maxList[:,:,4]))+' +/- '+str(np.mean(maxList[:,:,5])))
print('ta 2high= '+str(np.mean(maxList[:,:,16]))+' +/- '+str(np.mean(maxList[:,:,17])))
print('tc 2high= '+str(np.mean(maxList[:,:,18]))+' +/- '+str(np.mean(maxList[:,:,19])))

maxList = []
for item in heapq.nsmallest(2,tempVar2[:,8]):
		maxList.append(tempVar2[np.where(tempVar2[:,8]==item)])
maxList=np.array(maxList)
print('\nz 2close = '+str(np.mean(maxList[:,:,4]))+' +/- '+str(np.mean(maxList[:,:,5])))
print('ta 2close= '+str(np.mean(maxList[:,:,16]))+' +/- '+str(np.mean(maxList[:,:,17])))
print('tc 2close= '+str(np.mean(maxList[:,:,18]))+' +/- '+str(np.mean(maxList[:,:,19])))

maxList = []
for item in heapq.nsmallest(2,tempVar2[:,26]):
		maxList.append(tempVar2[np.where(tempVar2[:,26]==item)])
maxList=np.array(maxList)
print('\nz earliest = '+str(np.mean(maxList[:,:,4]))+' +/- '+str(np.mean(maxList[:,:,5])))
print('ta earliest= '+str(np.mean(maxList[:,:,16]))+' +/- '+str(np.mean(maxList[:,:,17])))
print('tc earliest= '+str(np.mean(maxList[:,:,18]))+' +/- '+str(np.mean(maxList[:,:,19])))

weightingZ,weightingZerr,weightingTa,weightingTaerr,weightingTc,weightingTcerr=(0 for i in range(6))
totDep = sum(tempVar2[:,13])
for item in tempVar:
	weightingZ += (item[4] * item[13])
	weightingZerr += (item[5] * item[13])
	weightingTa += (item[16] * item[13])
	weightingTaerr += (item[17] * item[13])
	weightingTc += (item[18] * item[13])
	weightingTcerr += (item[19] * item[13])
weightedZ = weightingZ / totDep
weightedZerr = weightingZerr / totDep
weightedTa = weightingTa / totDep
weightedTaerr = weightingTaerr / totDep
weightedTc = weightingTc / totDep
weightedTcerr = weightingTcerr / totDep
print('\nz weighted = '+str(weightedZ)+' +/- '+str(weightedZerr))
print('ta weighted= '+str(weightedTa)+' +/- '+str(weightedTaerr))
print('tc weighted= '+str(weightedTc)+' +/- '+str(weightedTcerr))

# sds = np.array(sds)
dt = np.array(dt)
tempVar=np.array(tempVar)
ii=0
print('\n')
while ii<len(sds):
	print(str(sds[ii][3])+':')
	print('   dt(us)='+str(dt[ii])+' +/- '+str(dterr[ii]) + ' or '+str(100*dterr[ii]/dt[ii])+' %')
	print('    z(km)='+str(tempVar[ii][4])+' +/- '+str(tempVar[ii][5]) + ' or '+str(100*tempVar[ii][5]/tempVar[ii][4])+' %')
	print(' elv(deg)='+str(tempVar[ii][0])+' +/- '+str(tempVar[ii][1]) + ' or '+str(100*tempVar[ii][1]/tempVar[ii][0])+' %')
	print('   x1(km)='+str(tempVar[ii][2])+' +/- '+str(tempVar[ii][3]) + ' or '+str(100*tempVar[ii][3]/tempVar[ii][2])+' %')
	print('   r1(km)='+str(tempVar[ii][6])+' +/- '+str(tempVar[ii][7]) + ' or '+str(100*tempVar[ii][7]/tempVar[ii][6])+' %')
	print('   ta(us)='+str(tempVar[ii][16])+' +/- '+str(tempVar[ii][17]) + ' or '+str(100*tempVar[ii][17]/tempVar[ii][16])+' %')
	print('   tc(us)='+str(tempVar[ii][18])+' +/- '+str(tempVar[ii][19]) + ' or '+str(100*tempVar[ii][19]/tempVar[ii][18])+' %')

	ii+=1

print('\nmedians:')
print('   dt(us)='+str(np.median(dt))+' +/- '+str(np.median(dterr)) + ' or '+str(100*np.median(dterr)/np.median(dt))+' %')
print('    z(km)='+str(np.median(tempVar2[:,4]))+' +/- '+str(np.median(tempVar2[:,5])) + ' or '+str(100*np.median(tempVar2[:,5])/np.median(tempVar2[:,4]))+' %')
print(' elv(deg)='+str(np.median(tempVar2[:,0]))+' +/- '+str(np.median(tempVar2[:,1])) + ' or '+str(100*np.median(tempVar2[:,1])/np.median(tempVar2[:,0]))+' % ')
print('   x1(km)='+str(np.median(tempVar2[:,2]))+' +/- '+str(np.median(tempVar2[:,3])) + ' or '+str(100*np.median(tempVar2[:,3])/np.median(tempVar2[:,2]))+' %')
print('   r1(km)='+str(np.median(tempVar2[:,6]))+' +/- '+str(np.median(tempVar2[:,7])) + ' or '+str(100*np.median(tempVar2[:,7])/np.median(tempVar2[:,6]))+' %')
print('   ta(us)='+str(np.median(tempVar2[:,16]))+' +/- '+str(np.median(tempVar2[:,17])) + ' or '+str(100*np.median(tempVar2[:,17])/np.median(tempVar2[:,16]))+' %')
print('   tc(us)='+str(np.median(tempVar2[:,18]))+' +/- '+str(np.median(tempVar2[:,19])) + ' or '+str(100*np.median(tempVar2[:,19])/np.median(tempVar2[:,18]))+' %')

print('\nmeans:')
print('   dt(us)='+str(np.mean(dt))+' +/- '+str(np.mean(dterr)) + ' or '+str(100*np.mean(dterr)/np.mean(dt))+' %')
print('    z(km)='+str(np.mean(tempVar2[:,4]))+' +/- '+str(np.mean(tempVar2[:,5])) + ' or '+str(100*np.mean(tempVar2[:,5])/np.mean(tempVar2[:,4]))+' %')
print(' elv(de)='+str(np.mean(tempVar2[:,0]))+' +/- '+str(np.mean(tempVar2[:,1])) + ' or '+str(100*np.mean(tempVar2[:,1])/np.mean(tempVar2[:,0]))+' %')
# print(' azi(deg)='+str(newIntfAziMean)+' +/- '+str(newIntfAziStd)+' or '+str(100*newIntfAziStd/newIntfAziMean)+' %')
print(' azi(deg)='+str(azi)+' +/- '+str(azierr)+' or '+str(100*azierr/azi)+' %')
print('   x1(km)='+str(np.mean(tempVar2[:,2]))+' +/- '+str(np.mean(tempVar2[:,3])) + ' or '+str(100*np.mean(tempVar2[:,3])/np.mean(tempVar2[:,2]))+' %')
print('   r1(km)='+str(np.mean(tempVar2[:,6]))+' +/- '+str(np.mean(tempVar2[:,7])) + ' or '+str(100*np.mean(tempVar2[:,7])/np.mean(tempVar2[:,6]))+' %')
print('   ta(us)='+str(np.mean(tempVar2[:,16]))+' +/- '+str(np.mean(tempVar2[:,17])) + ' or '+str(100*np.mean(tempVar2[:,17])/np.mean(tempVar2[:,16]))+' %')
print('   tc(us)='+str(np.mean(tempVar2[:,18]))+' +/- '+str(np.mean(tempVar2[:,19])) + ' or '+str(100*np.mean(tempVar2[:,19])/np.mean(tempVar2[:,18]))+' %')
print('   source@'+str(np.mean(tempVar2[:,27]))+', '+str(np.mean(tempVar2[:,28]))+' (deg)')
if savefile:
	sys.stdout.close()
	sys.stdout=stdoutOrigin


