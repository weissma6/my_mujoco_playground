__author__ = 'duguguang'

from nokov.nokovsdk import *
import time,math
import sys, getopt
from Utility import *

preFrmNo = 0
curFrmNo = 0
# 为每个标记点存储速度和加速度计算数组
velocity_arrays = {}
acceleration_arrays = {}

def py_data_func(pFrameOfMocapData, pUserData):
    if pFrameOfMocapData == None:  
        print("Not get the data frame.\n")
    else:
        frameData = pFrameOfMocapData.contents
        global preFrmNo, curFrmNo, velocity_arrays, acceleration_arrays
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

            # 计算每个标记点的速度和加速度
            for iMarker in range(markerset.nMarkers):
                # 创建唯一的标记点标识符
                marker_key = f"{iMarkerSet}_{iMarker}"
                
                # 初始化该标记点的速度和加速度数组（如果尚未存在）
                if marker_key not in velocity_arrays:
                    velocity_arrays[marker_key] = SlideFrameArray()
                    acceleration_arrays[marker_key] = SlideFrameArray()
                
                # 创建当前标记点
                marker_point = Point(
                    markerset.Markers[iMarker][0],
                    markerset.Markers[iMarker][1],
                    markerset.Markers[iMarker][2],
                    f"MarkerSet{iMarkerSet+1}_Marker{iMarker+1}"
                )
                
                # 缓存当前标记点位置用于计算
                velocity_arrays[marker_key].cache(marker_point)
                acceleration_arrays[marker_key].cache(marker_point)
                
                # 计算速度
                vel_method = CalculateVelocity(60, 3)  # FPS:60 FrameFactor:3
                velocity = velocity_arrays[marker_key].try_to_calculate(vel_method)
                
                # 计算加速度
                acc_method = CalculateAcceleration(60, 3)
                acceleration = acceleration_arrays[marker_key].try_to_calculate(acc_method)
                
                print(f"\tMarker{iMarker+1}(mm) \tx:{markerset.Markers[iMarker][0]:6.2f}"\
                    f"\ty:{markerset.Markers[iMarker][1]:6.2f}\tz:{markerset.Markers[iMarker][2]:6.2f}")
                
                # 打印速度和加速度
                if velocity:
                    print(f"\t  V(mm/s)\t{velocity}")
                else:
                    print(f"\t  V(mm/s)\tVx: 0.00\t Vy: 0.00\t Vz: 0.00")
                if acceleration:
                    print(f"\t  A(mm/s^2):\t{acceleration}")
                else:
                    print(f"\t  A(mm/s^2):\tAx: 0.00\t Ay: 0.00\t Az: 0.00")
            
            print("}")
                

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
    print('NokovrSDK Sample Client 4.1.0.5873(NokovrSDK ver. %d.%d.%d.%d)' % (ver[0], ver[1], ver[2], ver[3]))

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