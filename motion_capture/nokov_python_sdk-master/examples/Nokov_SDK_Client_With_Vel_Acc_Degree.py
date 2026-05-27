__author__ = 'duguguang'

from nokov.nokovsdk import *
import time,math
import sys, getopt
from Utility import *

preFrmNo = 0
curFrmNo = 0
FingertripVelocityArray = SlideFrameArray()
FingertripAccelerationArray = SlideFrameArray()

def py_data_func(pFrameOfMocapData, pUserData):
    if pFrameOfMocapData == None:  
        print("Not get the data frame.\n")
    else:
        frameData = pFrameOfMocapData.contents
        global preFrmNo, curFrmNo 
        curFrmNo = frameData.iFrame
        if curFrmNo == preFrmNo:
            return

        preFrmNo = curFrmNo
        print( "FrameNo: %d\tTimeStamp:%Ld" % (frameData.iFrame, frameData.iTimeStamp))					
        print( "nMarkerset = %d" % frameData.nMarkerSets)

        for iMarkerSet in range(frameData.nMarkerSets):
            markerset = frameData.MocapData[iMarkerSet]
            print( "Markerset%d: %s [nMarkers Count=%d]\n" % (iMarkerSet+1, markerset.szName, markerset.nMarkers))
            print("{\n")

            for iMarker in range(markerset.nMarkers):
                print("\tMarker%d: %3.2f,%3.2f,%3.2f\n" %(	
				iMarker,
				markerset.Markers[iMarker][0],
				markerset.Markers[iMarker][1],
				markerset.Markers[iMarker][2]))
            print( "}\n")

            # calculate Finger (assume the finger is the firstmarkerset which has only contains 3 markers)
            if (0 == iMarkerSet):
                Fingertrip = Point(markerset.Markers[0][0], markerset.Markers[0][1], markerset.Markers[0][2], "Fingertrip")
                FingerMiddle = Point(markerset.Markers[1][0], markerset.Markers[1][1], markerset.Markers[1][2], "FingerMiddle")
                FingerRoot = Point(markerset.Markers[2][0], markerset.Markers[2][1], markerset.Markers[2][2], "FingerRoot")

                # print points
                print(Fingertrip)
                print(FingerMiddle)
                print(FingerRoot)

                # calculate angle
                angle = calculate_angle(Fingertrip, FingerMiddle, FingerRoot)
                print(f"The angle between the lines is: {angle} degrees\n")

                # Caution: Actually, you cat get velocity of frame 2 after you get frame 3's position, so the velocity and acceleration belongs last frame
                # calculate Velocity of Fingertrip 
                global FingertripVelocityArray,FingertripAccelerationArray
                FingertripVelocityArray.cache(Fingertrip)
                FingertripAccelerationArray.cache(Fingertrip)

                method = CalculateVelocity(60, 3); # FPS:60 FrameFactor:3 which means the first 2 frame has None
                print(FingertripVelocityArray.try_to_calculate(method))

                # calculate Accel of Fingertrip 
                method2 = CalculateAcceleration(60, 3)
                print(FingertripAccelerationArray.try_to_calculate(method2))
        
        print("\n")

def py_msg_func(iLogLevel, szLogMessage):
    szLevel = "None"
    if iLogLevel == 4:
        szLevel = "Debug"
    elif iLogLevel == 3:
        szLevel = "Info"
    elif iLogLevel == 2:
        szLevel = "Warning"
    elif iLogLevel == 1:
        szLevel = "Error"
  
    print("[%s] %s" % (szLevel, cast(szLogMessage, c_char_p).value))

def py_forcePlate_func(pFocePlates, pUserData):
    if pFocePlates == None:  
        print("Not get the forcePlate frame.\n")
        pass
    else:
        ForcePlatesData = pFocePlates.contents
        print("iFrame:%d" % ForcePlatesData.iFrame)
        for iForcePlate in range(ForcePlatesData.nForcePlates):
            print("Fxyz:[%f,%f,%f] xyz:[%f,%f,%f] MFree:[%f]" % (
                ForcePlatesData.ForcePlates[iForcePlate].Fxyz[0],
                ForcePlatesData.ForcePlates[iForcePlate].Fxyz[1],
                ForcePlatesData.ForcePlates[iForcePlate].Fxyz[2],
                ForcePlatesData.ForcePlates[iForcePlate].xyz[0],
                ForcePlatesData.ForcePlates[iForcePlate].xyz[1],
                ForcePlatesData.ForcePlates[iForcePlate].xyz[2],
                ForcePlatesData.ForcePlates[iForcePlate].Mfree
            ))

def main(argv):
    serverIp = '10.1.1.198'

    try:
        opts, args = getopt.getopt(argv,"hs:",["server="])
    except getopt.GetoptError:
        print('NokovrSDKClient.py -s <serverIp>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('NokovrSDKClient.py -s <serverIp>')
            sys.exit()
        elif opt in ("-s", "--server"):
            serverIp = arg

    print ('serverIp is %s' % serverIp)
    print("Started the Nokovr_SDK_Client Demo")
    client = PySDKClient()

    ver = client.PyNokovVersion()
    print('NokovrSDK Sample Client 2.4.0.5428(NokovrSDK ver. %d.%d.%d.%d)' % (ver[0], ver[1], ver[2], ver[3]))

    client.PySetVerbosityLevel(0)
    client.PySetMessageCallback(py_msg_func)
    client.PySetDataCallback(py_data_func, None)

    print("Begin to init the SDK Client")
    ret = client.Initialize(bytes(serverIp, encoding = "utf8"))

    if ret == 0:
        print("Connect to the Nokovr Succeed")
    else:
        print("Connect Failed: [%d]" % ret)
        exit(0)


    serDes = ServerDescription()
    client.PyGetServerDescription(serDes)
    
    #Give 5 seconds to system to init forceplate device
    ret = client.PyWaitForForcePlateInit(5000)
    if (ret != 0):
        print("Init ForcePlate Failed[%d]" % ret)
        exit(0)

    client.PySetForcePlateCallback(py_forcePlate_func, None)

    while(input("Press q to quit\n") != "q"):
        pass
 
if __name__ == "__main__":
   main(sys.argv[1:])