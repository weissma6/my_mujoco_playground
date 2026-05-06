__author__ = 'Lika'

import os,platform,re
from ctypes import *
from enum import Enum

if platform.system().lower() == 'linux':
    System = "linux"
else:
    System = "windows"



if platform.architecture()[0].lower() == '64bit':
    lib_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib' + os.path.sep)
else:
    lib_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib32' + os.path.sep)

curr_dir_before = os.getcwd()
os.chdir(lib_dir_path)
lib_path = os.path.join(lib_dir_path, 'libnokov_sdk.so')

Machine = platform.machine()
if (System == "linux"):
    if (re.search('aarch64', Machine)):
        lib_path = os.path.join(lib_dir_path, 'libnokov_sdk_aarch64.so')
    elif (re.search('arm.*l', Machine) or re.search('arm.*hf', Machine)):
        lib_path = os.path.join(lib_dir_path, 'libnokov_sdk_armhf.so')
    else:
        lib_path = os.path.join(lib_dir_path, 'libnokov_sdk.so')
else:
    lib_path = os.path.join(lib_dir_path, 'nokov_sdk.dll')

print(f"Check System:{System} Machine:{Machine} arch:{platform.architecture()[0].lower()}")

nokovLib = CDLL(lib_path) 
print("Loaded nokovsdk")
os.chdir(curr_dir_before)

#Define Const Number
MAX_MODELS               =  200     # maximum number of MarkerSets 
MAX_RIGIDBODIES          =  1000    # maximum number of RigidBodies
MAX_NAMELENGTH           =  256     # maximum length for strings
MAX_MARKERS              =  200     # maximum number of markers per MarkerSet
MAX_RBMARKERS            =  20       # maximum number of markers per RigidBody
MAX_SKELETONS            =  100     # maximum number of skeletons
MAX_SKELRIGIDBODIES      =  200     # maximum number of RididBodies per Skeleton
MAX_LABELED_MARKERS      =  1000    # maximum number of labeled markers per frame
MAX_UNLABELED_MARKERS    =  1000    # maximum number of unlabeled (other) markers per frame
MAX_MSG_LENGTH			 =	100		# maximum number of message
MAX_CAMERA_NUM           =  100     # maximum number of cameras

MAX_FORCEPLATES          =  8       # maximum number of force plates
MAX_ANALOG_CHANNELS      =  32      # maximum number of data channels (signals) per analog/force plate device
MAX_ANALOG_CHANNELSEX    =  80      # maximum number of data channels (signals) per analog device
MAX_ANALOG_SUBFRAMES     =  30      # maximum number of analog/force plate frames per mocap frame
MAX_PACKETSIZE			 =  300000	 # max size of packet (actual packet size is dynamic)

#Define Enum
class NotifyAction(Enum):
    ActionAdd       =   1
    ActionRemove    =   2
    ActionCover     =   3
    ActionUpdate    =   4

class NotifyType(Enum):
    RigidBodyChange     =   1
    SkeletonChange      =   2		
    DataChange          =   3		
    StartPlay           =   4				
    PausePlay           =   5
    FrameRateChange     =   6		

class ErrorCode(Enum):
    ErrorCode_OK = 0
    ErrorCode_Internal = 1
    ErrorCode_External = 2
    ErrorCode_Network = 3
    ErrorCode_Other = 4
    
class DataDescriptors(Enum):
    Descriptor_MarkerSet        =   0
    Descriptor_RigidBody        =   1
    Descriptor_Skeleton         =   2
    Descriptor_ForcePlate       =   3
    Descriptor_MarkerSetEx      =   4
    Descriptor_Param            =   5
    Descriptor_CameraParam      =   6


class MocapDataType(Enum):
    MocapDataFrameNO            =   1
    MocapDataMarkerSetNum       =   2
    MocapDataRigidBodyNum       =   3
    MocapDataSkeletonNum        =   4
    MocapDataLabeledMarkerNum   =   5
    MocapDataOtherMarkerNum     =   6
    MocapDataAnalogChNum        =   7
    MocapDataParam              =   8

class ExtendDataType(Enum):
    ExtendDataRigidBody = 1
    ExtendDataEndType = 10

# Custom Structure
class InfoStructure(Structure):
    # Dump to Dict
    def dump_dict(self):
        info = {}
        # Get The _fields_
        # Check the Field Type
        # Support Recursion and iteration
        for k, v in self._fields_:
            av = getattr(self, k)
            if type(v) == type(Structure):
                av = av.dump_dict()
            elif type(v) == type(Array):
                av = cast(av, c_char_p).value
            else:
                pass
            info[k] = av
        return info

    def __str__(self):
        info = self.dump_dict()
        return repr(info)

    # Print Info
    def show(self):
        from pprint import pprint
        pprint(self.dump_dict())


# for ctypes
class ServerDescription(InfoStructure):
    _fields_ = [
        ("HostPresent", c_bool),
        ("szHostComputerName", c_char * MAX_NAMELENGTH),
        ("HostComputerAddress", c_ubyte * 4),
        ("szHostApp",  c_char * MAX_NAMELENGTH),
        ("HostAppVersion", c_ubyte * 4),
        ("HostApNokovSDKVersionpVersion", c_ubyte * 4),
    ]

class Marker(InfoStructure):
    _fields_ = [
        ("ID", c_int),
        ("x", c_float),
        ("y", c_float),
        ("z", c_float),
        ("size", c_float),
        ("params", c_short),
    ]

class MarkerSetDescription(InfoStructure):
    _fields_ = [
        ("szName", c_char * MAX_NAMELENGTH),
        ("nMarkers", c_int),
        ("szMarkerNames", POINTER(c_char_p)),
    ]

class MarkerSetData(InfoStructure):
    _fields_ = [
        ("szName", c_char * MAX_NAMELENGTH),
        ("nMarkers", c_int),
        ("Markers", POINTER(c_float * 3)),
    ]

class RigidBodyDescription(InfoStructure):
    _fields_ = [
        ("szName", c_char * MAX_NAMELENGTH),
        ("ID", c_int),
        ("parentID", c_int),
        ("offsetx", c_float),
        ("offsety", c_float),
        ("offsetz", c_float),
        ("qx", c_float),
        ("qy", c_float),
        ("qz", c_float),       
        ("qw", c_float),       
    ]

class RigidBodyData(InfoStructure):
    _fields_ = [
        ("ID", c_int),
        ("x", c_float),
        ("y", c_float),
        ("z", c_float),
        ("qx", c_float),
        ("qy", c_float),
        ("qz", c_float),
        ("qw", c_float),
        ("nMarkers", c_int),
        ("Markers", POINTER(c_float * 3)),
        ("MarkerIDs", POINTER(c_int)),
        ("MarkerSizes", POINTER(c_float)),
        ("MeanError", c_float),
        ("params", c_short),
    ]

class SkeletonData(InfoStructure):
    _fields_ = [
        ("skeletonID", c_int),
        ("nRigidBodies", c_int),
        ("RigidBodyData", POINTER(RigidBodyData)),
    ]

class SkeletonDescription(InfoStructure):
    _fields_ = [
        ("szName", c_char * MAX_NAMELENGTH),
        ("skeletonID", c_int),
        ("nRigidBodies", c_int),
        ("RigidBodies", RigidBodyDescription * MAX_SKELRIGIDBODIES),
    ]

class ForcePlateData(InfoStructure):
    _fields_ = [
        ("Fxyz", c_float * 3),
        ("xyz", c_float * 3),
        ("Mfree", c_float),
    ]

class ForcePlates(InfoStructure):
    _fields_ = [
        ("iFrame", c_int),
        ("nForcePlates", c_int),
        ("ForcePlates", ForcePlateData * MAX_FORCEPLATES),
    ]

class DataParam(InfoStructure):
    _fields_ = [
        ("nFrameRate", c_int),
        ("nReserve1", c_int),
        ("nReserve2", c_int),
        ("nReserve3", c_int),
        ("nReserve4", c_int),
    ]

class CameraInfo(InfoStructure):
    _fields_ = [
        ("cameraId", c_int),
        ("tx", c_double),
        ("ty", c_double),
        ("tz", c_double),
        ("rx", c_double),
        ("ry", c_double),
        ("rz", c_double),
    ]

class CameraDescriptions(InfoStructure):
    _fields_ = [
        ("nCameras", c_int),
        ("Cameras", CameraInfo * MAX_CAMERA_NUM),
    ]

class ForcePlateDescription(InfoStructure):
    _fields_ = [
        ("ID", c_int),
        ("scale", c_int),
        ("fWidth", c_float),
        ("fLength", c_float),
        ("Position", c_float * 3),
        ("Electrical", c_float * 3),
        ("Orientation", (c_float * 3) * 3),
        ("fCalMat", (c_float * 8) * 8),
        ("fCorners", (c_float * 3) * 4),
        ("iPlateType", c_int),
        ("iChannelDataType", c_int),
        ("nChannels", c_int),
        ("szChannelNames", (c_char * MAX_NAMELENGTH) * MAX_ANALOG_CHANNELS),
    ]

class Data(Union):
    _fields_ = [
        ("MarkerSetDescription", POINTER(MarkerSetDescription)),
        ("RigidBodyDescription", POINTER(RigidBodyDescription)),
        ("SkeletonDescription", POINTER(SkeletonDescription)),
        ("ForcePlateDescription", POINTER(ForcePlateDescription)),
        ("MarkerSetData", POINTER(MarkerSetData)),
        ("DataParam", POINTER(DataParam)),
        ("cameraDescriptions", POINTER(CameraDescriptions)),
    ]

class DataDescription(InfoStructure):
    _fields_ = [
        ("type", c_int),
        ("Data", Data),
    ]

class DataDescriptions(InfoStructure):
    _fields_ = [
        ("nDataDescriptions", c_int),
        ("arrDataDescriptions", DataDescription * MAX_MODELS),
    ]

class RigidBodyExtendData(InfoStructure):
    _fields_ = [
        ("ID", c_int),
        ("roll", c_float),
        ("pitch", c_float),
        ("yaw", c_float),
        ("rollVel", c_float),
        ("pitchVel", c_float),
        ("yawVel", c_float),
        ("rollAvel", c_float),
        ("pitchAvel", c_float),
        ("yawAvel", c_float),
        ("xVel", c_float),
        ("yVel", c_float),
        ("zVel", c_float),
        ("rVel", c_float),
        ("xAvel", c_float),
        ("yAvel", c_float),
        ("zAvel", c_float),
        ("rAvel", c_float),
        ("reserve", c_int * 4),
    ]

class ExtendDataUnion(Union):
    _fields_ = [
        ("RigidBodyExtendData", RigidBodyExtendData*MAX_RIGIDBODIES),
        ("void", c_void_p),
    ]

class ExtendData(InfoStructure):
    _fields_ = [
        ("type", c_int),
        ("number", c_int),
        ("length", c_int),
        ("ExtendDataUnion", ExtendDataUnion),
    ]

class FrameExtendData(InfoStructure):
    _fields_ = [
        ("nExtendDataNum", c_int),
        ("extendData", ExtendData * ExtendDataType.ExtendDataEndType.value),
    ]

class FrameOfMocapData(InfoStructure):
    _fields_ = [    
        ("iFrame", c_int),
        ("nMarkerSets", c_int),
        ("MocapData", MarkerSetData * MAX_MODELS),
        ("nOtherMarkers", c_int),
        ("OtherMarkers", POINTER(c_float * 3)),
        ("nRigidBodies", c_int),
        ("RigidBodies", RigidBodyData * MAX_RIGIDBODIES),
        ("nSkeletons", c_int),
        ("Skeletons", SkeletonData * MAX_SKELETONS),
        ("nLabeledMarkers", c_int),
        ("LabeledMarkers", Marker * MAX_LABELED_MARKERS),
        ("nAnalogdatas", c_int),
        ("Analogdata", c_float * MAX_ANALOG_CHANNELS),
        ("fLatency", c_float),
        ("Timecode", c_uint),
        ("TimecodeSubframe", c_uint),
        ("iTimeStamp", c_longlong),
        ("params", c_short),
        ("FrameExtendData", FrameExtendData),
    ]

class NotifyMsg(InfoStructure):
    _fields_ = [    
        ("nType", c_int),
        ("nValue", c_int),
        ("nParam1", c_int),
        ("nParam2", c_int),
        ("nParam3", c_int),
        ("nParam4", c_int),
        ("nTimeStamp", c_ulonglong),
        ("sMsg", c_char * MAX_MSG_LENGTH), 
    ]

class FrameOfAnalogChannelData(InfoStructure):
    _fields_ = [
        ("iFrame", c_int),
        ("nAnalogdatas", c_int),
        ("nSubFrame", c_int),
        ("Analogdata", (c_float * MAX_ANALOG_SUBFRAMES) * MAX_ANALOG_CHANNELSEX ),
        ("Timecode", c_uint),
        ("TimecodeSubframe", c_uint),
        ("iTimeStamp", c_longlong),
        ("params", c_short),
    ]

# CallBack
MSGFUNC = CFUNCTYPE(None, c_int, POINTER(c_char))
DATAHANDLEFUNC = CFUNCTYPE(None, POINTER(FrameOfMocapData), c_void_p)
FORCEPLATEFUNC = CFUNCTYPE(None, POINTER(ForcePlates), c_void_p)
NOTIFYFUNC = CFUNCTYPE(None, POINTER(NotifyMsg), c_void_p)
ANALOGCHFUNC = CFUNCTYPE(None, POINTER(FrameOfAnalogChannelData), c_void_p)

# for test
def py_msg_func(iLogLevel, szLogMessage):
    print(iLogLevel, szLogMessage)

def py_data_func(frameData, userData):
    print(frameData)

def py_force_plate_func(forceData, userData):
    print(forceData)

def py_notify_func(notify, userData):
    print(notify)

def py_analog_channel_func(analogData, userData):
    print(analogData)

msg_func = MSGFUNC(py_msg_func)
data_func = DATAHANDLEFUNC(py_data_func)
forcePlate_func = FORCEPLATEFUNC(py_force_plate_func)
notify_func =  NOTIFYFUNC(py_notify_func)
analog_func = ANALOGCHFUNC(py_analog_channel_func)

class PySDKClient():
    __pClient = c_void_p()

    def __init__(self):
        retVal = self.__CreateClient()
        if (retVal != 0):
            raise ValueError('__CreateClient Failed')
            
    def __del__(self):
        self.__Uninitialize()
        self.__DestroyClient()

    def __CreateClient(self):
        nokovLib.CreateClient.argtypes = [c_void_p]
        return nokovLib.CreateClient(byref(self.__pClient))

    def Initialize(self, szServerAddress):
        nokovLib.Initialize.argtypes = [c_void_p,  c_char_p]
        return nokovLib.Initialize(self.__pClient, szServerAddress)

    def __DestroyClient(self):
        nokovLib.DestroyClient.argtypes = [c_void_p]
        return nokovLib.DestroyClient(self.__pClient)

    def __Uninitialize(self):
        nokovLib.Uninitialize.argtypes = [c_void_p]
        return nokovLib.Uninitialize(self.__pClient)

    def PyNokovVersion(self):
        nokovLib.NokovVersion.argtypes = [c_void_p, POINTER(c_ubyte)]
        Version = (c_ubyte * 4)(0,0,0,0)
        nokovLib.NokovVersion(self.__pClient, Version)
        return Version

    def PySetVerbosityLevel(self, ilevel):
        nokovLib.SetVerbosityLevel.argtypes = [c_void_p, c_int]
        return nokovLib.SetVerbosityLevel(self.__pClient, ilevel)

    def PySetDataCallback(self, dataHandler, pUserData):
        nokovLib.SetDataCallback.argtypes = [c_void_p, DATAHANDLEFUNC, c_void_p]
        if dataHandler != None:
            global data_func
            data_func = DATAHANDLEFUNC(dataHandler)
        return nokovLib.SetDataCallback(self.__pClient, data_func, pUserData)

    def PySetForcePlateCallback(self, dataHandler, pUserData):
        nokovLib.SetForcePlateCallback.argtypes = [c_void_p, FORCEPLATEFUNC, c_void_p]
        if dataHandler != None:
            global forcePlate_func
            forcePlate_func = FORCEPLATEFUNC(dataHandler)
        return nokovLib.SetForcePlateCallback(self.__pClient, forcePlate_func, pUserData)        

    def PyWaitForForcePlateInit(self, time):
        nokovLib.WaitForForcePlateInit.argtypes = [c_void_p, c_long]
        return nokovLib.WaitForForcePlateInit(self.__pClient, time)

    def PySetMessageCallback(self, msgHandler):
        nokovLib.SetMessageCallback.argtypes= [c_void_p, MSGFUNC]
        if msgHandler != None:
            global msg_func
            msg_func = MSGFUNC(msgHandler)
        return nokovLib.SetMessageCallback(self.__pClient, msg_func)

    def PySetNotifyMsgCallback(self, notifyHandler, pUserData):
        nokovLib.SetNotifyMsgCallback.argtypes = [c_void_p, NOTIFYFUNC, c_void_p]
        if notifyHandler != None:
            global notify_func
            notify_func = NOTIFYFUNC(notifyHandler)
        return nokovLib.SetNotifyMsgCallback(self.__pClient, notify_func, pUserData)

    def PySetAnalogChFunc(self, analogHandler, pUserData):
        nokovLib.SetAnalogChCallback.argtypes = [c_void_p, ANALOGCHFUNC, c_void_p]
        if ANALOGCHFUNC != None:
            global analog_func
            analog_func = ANALOGCHFUNC(analogHandler)
        return nokovLib.SetAnalogChCallback(self.__pClient, analog_func, pUserData)

    def PySendMessage(self, szMessage):
        nokovLib.SendMessage.argtypes = [c_void_p, c_char_p]
        return nokovLib.SendMessage(self.__pClient, szMessage)

    def PyGetServerDescription(self, pServerDescription):
        nokovLib.GetServerDescription.argtypes = [c_void_p, POINTER(ServerDescription)]
        return nokovLib.GetServerDescription(self.__pClient, pServerDescription)

    def PyGetDataDescriptions(self, phDataDescriptions):
        nokovLib.GetDataDescriptions.argtypes = [c_void_p, POINTER(POINTER(DataDescriptions))]
        return nokovLib.GetDataDescriptions(self.__pClient, byref(phDataDescriptions))

    def PyGetDataDescriptionsEx(self, phDataDescriptions, handle):
        nokovLib.GetDataDescriptionsEx.argtypes = [c_void_p, POINTER(DataDescriptions), c_void_p]
        return nokovLib.GetDataDescriptionsEx(self.__pClient, byref(phDataDescriptions), byref(handle))
    
    def PyTposeDataDescriptions(self, name, phDataDescriptions):
        nokovLib.GetTposeDataDescriptions.argtypes = [c_void_p, c_char_p, POINTER(DataDescriptions)]
        return nokovLib.GetTposeDataDescriptions(self.__pClient, name, byref(phDataDescriptions))

    def PyFreeDataDescriptionsEx(self, handle):
        nokovLib.FreeDataDescriptionsEx.argtypes = [c_void_p, c_void_p]
        return nokovLib.FreeDataDescriptionsEx(self.__pClient, handle)

    def PySetMulticastAddress(self, szMulticast):
        nokovLib.SetMulticastAddress.argtypes = [c_void_p, c_char_p]
        return nokovLib.SetMulticastAddress(self.__pClient, szMulticast)

    def PyDecodeTimecode(self, inTimecode, inTimecodeSubframe, hour, minute, second, frame, subframe):
        nokovLib.DecodeTimecode.argtypes = [c_void_p, c_uint, c_uint, POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int), POINTER(c_int)]
        return nokovLib.DecodeTimecode(self.__pClient, inTimecode, inTimecodeSubframe, hour, minute, second, frame, subframe)

    def PyTimecodeStringify(self, inTimecode, inTimecodeSubframe, Buffer, BufferSize):
        nokovLib.TimecodeStringify.argtypes = [c_void_p, c_uint, c_uint, c_char_p, c_int]
        return nokovLib.TimecodeStringify(self.__pClient, inTimecode, inTimecodeSubframe, Buffer, BufferSize)

    def PyGetLastFrameOfMocapData(self):
        nokovLib.GetLastFrameOfMocapData.argtypes = [c_void_p]
        nokovLib.GetLastFrameOfMocapData.restype = POINTER(FrameOfMocapData)
        return nokovLib.GetLastFrameOfMocapData(self.__pClient)

    def PyNokovFreeFrame(self, pFrameOfMocapData):
        nokovLib.NokovFreeFrame.argtypes = [c_void_p, POINTER(FrameOfMocapData)]
        return nokovLib.NokovFreeFrame(self.__pClient, pFrameOfMocapData)

    def PyGetLastFrameDataByType(self, data_type):
        out_value = c_int()
        nokovLib.GetLastFrameDataByType.argtypes = [c_void_p, c_int, POINTER(c_int)]
        nokovLib.GetLastFrameDataByType.restype = ErrorCode
        result = nokovLib.GetLastFrameDataByType(self.__pClient, data_type, byref(out_value))
        return result, out_value.value

    def PyGetLastFrameLatency(self):
        out_value = c_float()
        nokovLib.GetLastFrameLatency.argtypes = [c_void_p, POINTER(c_float)]
        nokovLib.GetLastFrameLatency.restype = ErrorCode
        result = nokovLib.GetLastFrameLatency(self.__pClient, byref(out_value))
        return result, out_value.value

    def PyGetLastFrameSubframe(self):
        out_value = c_uint()
        nokovLib.GetLastFrameSubframe.argtypes = [c_void_p, POINTER(c_uint)]
        nokovLib.GetLastFrameSubframe.restype = ErrorCode
        result = nokovLib.GetLastFrameSubframe(self.__pClient, byref(out_value))
        return result, out_value.value

    def PyGetLastFrameTimecode(self):
        out_value = c_uint()
        nokovLib.GetLastFrameTimecode.argtypes = [c_void_p, POINTER(c_uint)]
        nokovLib.GetLastFrameTimecode.restype = ErrorCode
        result = nokovLib.GetLastFrameTimecode(self.__pClient, byref(out_value))
        return result, out_value.value

    def PyGetLastFrameTimeStamp(self):
        out_value = c_longlong()
        nokovLib.GetLastFrameTimeStamp.argtypes = [c_void_p, POINTER(c_longlong)]
        nokovLib.GetLastFrameTimeStamp.restype = ErrorCode
        result = nokovLib.GetLastFrameTimeStamp(self.__pClient, byref(out_value))
        return result, out_value.value

    def PyGetLastFrameMarkerByName(self, markersetName, markerName):
        data = POINTER(c_float)()
        nokovLib.GetLastFrameMarkerByName.argtypes = [c_void_p, c_char_p, c_char_p, POINTER(POINTER(c_float))]
        nokovLib.GetLastFrameMarkerByName.restype = ErrorCode
        result = nokovLib.GetLastFrameMarkerByName(self.__pClient, markersetName.encode('utf-8'), markerName.encode('utf-8'), byref(data))
        return result, data

    def PyGetLastFrameRigidBodyByName(self, name):
        rigid_body_data_instance = RigidBodyData()
        data = POINTER(RigidBodyData)(rigid_body_data_instance)
        nokovLib.GetLastFrameRigidBodyByName.argtypes = [c_void_p, c_char_p, POINTER(POINTER(RigidBodyData))]
        nokovLib.GetLastFrameRigidBodyByName.restype = ErrorCode
        result = nokovLib.GetLastFrameRigidBodyByName(self.__pClient, name.encode('utf-8'), byref(data))
        return result, data

    def PyGetLastFrameSkeletonByName(self, markerSetName, skeName):
        data = POINTER(RigidBodyData)()
        nokovLib.GetLastFrameSkeletonByName.argtypes = [c_void_p, c_char_p, c_char_p, POINTER(POINTER(RigidBodyData))]
        nokovLib.GetLastFrameSkeletonByName.restype = ErrorCode
        result = nokovLib.GetLastFrameSkeletonByName(self.__pClient, markerSetName.encode('utf-8'), skeName.encode('utf-8'), byref(data))
        return result, data

    def PyGetLastFrameLabeledMarker(self):
        marker = POINTER(Marker)()
        nokovLib.GetLastFrameLabeledMarker.argtypes = [c_void_p, POINTER(POINTER(Marker))]
        nokovLib.GetLastFrameLabeledMarker.restype = ErrorCode
        result = nokovLib.GetLastFrameLabeledMarker(self.__pClient, byref(marker))
        return result, marker

    def PyGetLastFrameUndefined(self, index):
        point = POINTER(c_float)()
        nokovLib.GetLastFrameUndefined.argtypes = [c_void_p, c_int, POINTER(POINTER(c_float))]
        nokovLib.GetLastFrameUndefined.restype = ErrorCode
        result = nokovLib.GetLastFrameUndefined(self.__pClient, index, byref(point))
        return result, point

    def PyGetLastFrameAnalogdata(self, chName):
        data = c_float()
        nokovLib.GetLastFrameAnalogdata.argtypes = [c_void_p, c_char_p, POINTER(c_float)]
        nokovLib.GetLastFrameAnalogdata.restype = ErrorCode
        result = nokovLib.GetLastFrameAnalogdata(self.__pClient, chName.encode('utf-8'), byref(data))
        return result, data.value

    def PyGGetLastFrameExtendData(self):
        data = ExtendData()
        nokovLib.GetLastFrameExtendData.argtypes = [c_void_p, POINTER(ExtendData)]
        nokovLib.GetLastFrameExtendData.restype = ErrorCode
        result = nokovLib.GetLastFrameExtendData(self.__pClient, byref(data))
        return result, data
