import os
import sys
import heapq
import numpy as np
import geopy as gp
import geopy.distance
import scipy.constants as sp

def READ_TASD_BURST(dirname,date, TASD_shift, allmicro, alllSignal, alluSignal, alldec):
    os.chdir(dirname)
    print (dirname)
    i = 0
    sds = []
    for file in os.listdir():
        if file.startswith("SD"+date):
            i = i + 1
            print(i, file, TASD_shift)
            # allmicro, alllSignal, alluSignal = ([] for i in range(3))
            dec = file[-8:-4]
            # print (dec)
            SD_data = open(file, 'r')
            micro, lSignal, uSignal = ([] for i in range(3))
            header = True
            sigCount = 0
            for line in SD_data:
                if line.startswith('###'):
                    continue
                if header:  # get header data
                    if line.startswith('burst times'):
                        trigs = np.array(line.replace(',', '').split(' ')[3:])
                        trigs = trigs.astype(int)
                        trig = trigs[0]
                        continue
                    if line.startswith('burst VEM'):
                        deps = np.array(line.replace(',', '').split(' ')[3:])
                        deps = deps.astype(float)
                        dep = np.sum(deps)
                        continue
                    if line.startswith('pedestal'):
                        if len(line.split(' ')) == 4:  # check pedestal header formatting
                            sigPedL = float(line.split(' ')[2].strip(','))
                            sigPedU = float(line.split(' ')[3])
                        else:
                            sigPedL = float(line.split(' ')[2])
                            sigPedU = sigPedL
                        continue
                    if line.startswith('us'):
                        header = False
                        continue
                else:
                    columns = line.split(', ')
                    if line == '\n':
                        break
                    if columns[1] == '0.0' and columns[2] == '0.0':  # ignore waveform separators and drop signal
                        continue
                    # print ("Ny")
                    time = trig + TASD_shift + float(columns[0])
                    # time = (time/1000) + 1000 #### Depend on the plot whether for 3 stroke together - change to milliseconds scale
                    time = time+1000000 #### microsecond but after 16:18:44
                    micro.append(time)
                    lSignal.append(int(float(columns[1])))
                    uSignal.append(int(float(columns[2])))
                    if (
                            sigCount < 15):  # check for individual sd trigger time (integrated 15 FADC counts above pedestal)
                        intSigL = lSignal[-1] - sigPedL
                        if intSigL < 0:
                            intSigL = 0
                        intSigU = uSignal[-1] - sigPedU
                        if intSigU < 0:
                            intSigU = 0
                        aveSig = (intSigL + intSigU) / 2
                        sigCount += aveSig  # count integrated signal
                        if (sigCount >= 10):
                            sdTrig = micro[-1]
            SD_data.close()
            allmicro.append(micro)  #### return
            alllSignal.append(lSignal)  #### return
            alluSignal.append(uSignal)  #### return l
            alldec.append(dec)
            # print (alluSignal)
            # print (detectors)
    return allmicro, alllSignal, alluSignal, alldec

##### Read INTF points #####
def READ_INTF(intfFile,manualTrig, manualIntfTrig, scribbleData):
    if scribbleData: # check if scribble data format
        intfList=np.genfromtxt(intfFile,skip_header=2,usecols=(0,1,2,3,4,5))
        for item in intfList:
            item[0]=(item[0]%1)*(10**6)
    else:
        ff=open(intfFile,'r')
        for line in ff: # get header data
            if line.startswith('#uSecond'):
                intfTrig = float(line.split(':')[1])
            if line.startswith('#Time'):
                break
        ff.close()

        intfList=np.genfromtxt(intfFile,skip_header=44,usecols=(0,1,2,3,4,5))
        intfList[:,0] *= 1000.
        if manualTrig: # REMOVE THIS LATER - INTF TRIGGER TIME CALIBRATION?
            intfTrig = manualIntfTrig
        intfList[:,0] += intfTrig
    return intfList
