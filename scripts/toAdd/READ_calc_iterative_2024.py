import os
import sys
import heapq
import numpy as np
import geopy as gp
import geopy.distance
import scipy.constants as sp

def READ_TASD(sdDir, time, trigNum, manualStart, sdStartTime, sds):
    ##### Find all relevant SD numbers/locations and pull data #####
    # with open("students.csv", "r") as ff:
    ff = open('tasd_gpscoors.txt', 'r')

    for root, dirs, files in os.walk(sdDir):  # check all TASD files in sdDir
        for filename in files:
            if filename.endswith(".txt"):
                # print(filename)
                item = filename.split('/')[-1].split('_')
                print (item[1])
                if (item[1] == time):  # confirm file time
                    # print ("Ny")
                    ff.seek(0)  # return to beginning of tasd_gpscoors.txt
                    # print("Ny")
                    for line in ff:
                        line = line.strip()
                        columns = line.split()
                        # print ("item -1", item[-1][:4])
                        if columns[0] == item[-1][:4]:  # look for detector gps coords
                            ff2 = open(sdDir + "/" + filename, 'r')
                            # print ("file file file")
                            # print (sdDir+"/"+filename)
                            sdTime, sdSigL, sdSigU = ([] for i in range(3))
                            header = True
                            sigCount = 0
                            for line2 in ff2:  # loop through lines for timing calibration
                                if line2.startswith('###'):
                                    continue
                                if header:  # get header data
                                    if line2.startswith('burst times'):
                                        trigs = np.array(line2.replace(',', '').split(' ')[3:])
                                        trigs = trigs.astype(int)
                                        trig = trigs[0]
                                        continue
                                    if line2.startswith('burst VEM'):
                                        deps = np.array(line2.replace(',', '').split(' ')[3:])
                                        deps = deps.astype(float)
                                        dep = deps[trigNum - 1]

                                        continue
                                    if line2.startswith('pedestal'):
                                        if len(line2.split(' ')) == 4:  # check pedestal header formatting
                                            sigPedL = float(line2.split(' ')[2].strip(','))
                                            sigPedU = float(line2.split(' ')[3])
                                        else:
                                            sigPedL = float(line2.split(' ')[2])
                                            sigPedU = sigPedL
                                        continue
                                    if line2.startswith('us'):
                                        header = False
                                        continue
                                else:  # get waveform data
                                    columns2 = line2.split(', ')
                                    if line2 == '\n':
                                        break
                                    if columns2[1] == '0.0' and columns2[2] == '0.0':  # ignore waveform separators
                                        continue
                                    if manualStart:  # ignoring signals before manual start time, if turned on
                                        if trig + float(columns2[0]) < sdStartTime:
                                            continue
                                    if (trig + float(columns2[0]) < trigs[trigNum - 1]):  # ignore data before trigNum
                                        # print(str(trig+float(columns2[0]))+' < '+str(trigs[trigNum-1]))
                                        continue
                                    if (trigNum < len(trigs)) and (
                                            trig + float(columns2[0]) > trigs[trigNum]):  # ignore data after trigNum
                                        # print(str(trig+float(columns2[0]))+' > '+str(trigs[trigNum]))
                                        break
                                    sdTime.append(trig + float(columns2[0]))
                                    sdSigL.append(int(float(columns2[1])))
                                    sdSigU.append(int(float(columns2[2])))
                                    if (
                                            sigCount < 15):  # check for individual sd trigger time (integrated 15 FADC counts above pedestal)
                                        intSigL = sdSigL[-1] - sigPedL
                                        if intSigL < 0:
                                            intSigL = 0
                                        intSigU = sdSigU[-1] - sigPedU
                                        if intSigU < 0:
                                            intSigU = 0
                                        aveSig = (intSigL + intSigU) / 2
                                        sigCount += aveSig  # count integrated signal
                                        if (sigCount >= 10):
                                            sdTrig = sdTime[-1]
                                            # print ("sdTime ", sdTime)
                            # add all data to TASD data array:
                            if len(sdTime) > 1:
                                sds.append([sdTime, sdSigL, sdSigU, item[-1][:4], float(columns[1]), float(columns[2]),
                                            0.001 * float(columns[3]), dep, trig, sdTrig])
                                # print (float(columns[2]), sdTrig)
                            ff2.close()
                            break  # continue to next detector
    ff.close()
        # print(np.array(sds)[:, -1])
    return sds

##### Read INTF points #####
def READ_INTF(intfFile ):
    intfList = np.genfromtxt(intfFile, skip_header=2, usecols=(0, 1, 2, 3, 4, 5))
    # for item in intfList:
    #     item[0] = (item[0] % 1) * (10 ** 6)
    return intfList

def READ_LMA(lmaFile):
    lmaList = np.loadtxt(lmaFile)
    return lmaList