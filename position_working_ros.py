# Working version


import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from vicon_dssdk import ViconDataStream
import argparse
import sys
from digi.xbee.devices import XBeeDevice
from digi.xbee.devices import RemoteZigBeeDevice

from digi.xbee.models.address import XBee64BitAddress
from digi.xbee.models.status import NetworkDiscoveryStatus
from digi.xbee.devices import XBeeDevice, RemoteXBeeDevice

from time import time
from threading import Thread, Event
from time import sleep
from threading import Lock
from scipy.signal import buttap, lp2hp_zpk, bilinear_zpk, zpk2tf, butter, filtfilt
import matplotlib.pyplot as plt
import math
from scipy.spatial.transform import Rotation as R
import time

event = Event()
data_lock = Lock()

# TODO: Replace with the serial port where your local module is connected to.
PORT = "COM4"
# TODO: Replace with the baud rate of your local module.
BAUD_RATE = 115200


def callback_discovery_finished(status):
    if status == NetworkDiscoveryStatus.SUCCESS:
        print("  Discovery process finished successfully.")
    else:
        print("  There was an error discovering devices: %s" % status.description)


def cb_network_modified(event_type, reason, node):
    print("  >>>> Network event:")
    print("         Type: %s (%d)" % (event_type.description, event_type.code))
    print("         Reason: %s (%d)" % (reason.description, reason.code))
    if not node:
        return
    print("         Node:")
    print("            %s" % node)


def print_nodes(xb_net):
    print("\n  Current network nodes:\n    ", end='')
    if xb_net.has_devices():
        print("%s" % '\n    '.join(map(str, xb_net.get_devices())))
    else:
        print("None")


def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread

    return wrapper


def vicon_init():
    # Initialise Vicon, get initial pose of all available agents
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('host', nargs='?', help="Host name, in the format of server:port", default="localhost:801")
    args = parser.parse_args()
    client = ViconDataStream.Client()

    print(" Connecting to Vicon host")
    client.Connect(args.host)
    client.EnableSegmentData()

    hasFrame = False
    timeout = 1000

    while not hasFrame:
        print('.')
        try:
            if client.GetFrame():
                hasFrame = True
            timeout = timeout - 1
            if timeout < 0:
                print('Failed to get frame')
                sys.exit()
        except ViconDataStream.DataStreamException as e:
            client.GetFrame()

    client.SetStreamMode(ViconDataStream.Client.StreamMode.EServerPush)
    print('Get Frame Pull', client.GetFrame(), client.GetFrameNumber())
    print('Segments', client.IsSegmentDataEnabled())

    subjectName = client.GetSubjectNames()
    idata = dict()

    print(subjectName)

    for subName in subjectName:
        segmentName = client.GetSegmentNames(subName)
        print(segmentName)
        print(subName)

        print(segmentName, 'has global translation', client.GetSegmentGlobalTranslation(subName, segmentName[0]))
        pos = client.GetSegmentGlobalTranslation(subName, segmentName[0])[0]

        print(segmentName, 'has global rotation( EulerXYZ )',
              client.GetSegmentGlobalRotationEulerXYZ(subName, segmentName[0]))
        rot = client.GetSegmentGlobalRotationEulerXYZ(subName, segmentName[0])[0]
        # data.update({subName: [pos, rot]})

        idata.update({subName: np.array([pos, rot, pos, rot])})

    idata.update({'dt': 0})

    # Start Vicon
    sleep(0.1)
    print(idata)

    print('Vicon Initialised')  # Press Ctrl+F8 to toggle the breakpoint.
    return client, segmentName, subjectName, idata


def get_Vframe(client):
    return client.GetFrame()


def xbee_init(names):
    xbee_network = None

    xbee = XBeeDevice(PORT, BAUD_RATE)

    try:
        xbee.open()
        xbee.set_sync_ops_timeout(1)

        xbee_network = xbee.get_network()
        remote_devicess = []

        for name in names:
            remote_devicess.append(xbee_network.discover_device(name))

        a = list(map(str, remote_devicess))
        b = []

        for i in range(len(a)):
            b.append(a[i][-8:])

        print(remote_devicess)
        print(b)

    finally:
        if xbee_network is not None:
            xbee_network.del_discovery_process_finished_callback(callback_discovery_finished)
            xbee_network.del_network_modified_callback(cb_network_modified)

    return xbee, remote_devicess, b


###### filters
def lowpass_vel_filter(f, m):


    # order 6
    # this one somewhat working
    a1 = 0.02957
    a2 = 0.04844
    a3 = 0.06826
    a4 = 0.07608
    a5 = 0.06826
    a6 = 0.04844
    a7 = 0.02957

    y0 = a1 * m + a2 * f[0, :] + a3 * f[1, :] + a4 * f[2, :] + a5 * f[3, :]
    y0 = a1 * m + a2 * f[0, :] + a3 * f[1, :] + a4 * f[2, :] + a5 * f[3, :] + a6 * f[4, :] + a7 * f[5, :]
    f[5, :] = f[4, :]
    f[4, :] = f[3, :]
    f[3, :] = f[2, :]
    f[2, :] = f[1, :]
    f[1, :] = f[0, :]
    f[0, :] = m

    return y0, f


def derivative_vel_filter(f, m):
    # freq vec=[0.05 0.3 0.5 1] best one
    a1 = -0.1533
    a2 = 0.1950
    a3 = -0.1904
    a4 = -0.0098
    a5 = 0.2457
    a6 = 0.2415
    a7 = 0
    a8 = -0.2415
    a9 = -0.2457
    a10 = 0.0098
    a11 = 0.1904
    a12 = -0.1950
    a13 = 0.1533

    y0 = 100 * (a1 * m + a2 * f[0, :] + a3 * f[1, :] + a4 * f[2, :] + a5 * f[3, :] + a6 * f[4, :] + a7 * f[5,
                                                                                                         :] + a8 * f[6,
                                                                                                                   :] + a9 * f[
                                                                                                                             7,
                                                                                                                             :] + a10 * f[
                                                                                                                                        8,
                                                                                                                                        :] + a11 * f[
                                                                                                                                                   9,
                                                                                                                                                   :] + a12 * f[
                                                                                                                                                              10,
                                                                                                                                                              :] + a13 * f[
                                                                                                                                                                         11,
                                                                                                                                                                         :])
    f[11, :] = f[10, :]
    f[10, :] = f[9, :]
    f[9, :] = f[8, :]
    f[8, :] = f[7, :]
    f[7, :] = f[6, :]
    f[6, :] = f[5, :]
    f[5, :] = f[4, :]
    f[4, :] = f[3, :]
    f[3, :] = f[2, :]
    f[2, :] = f[1, :]
    f[1, :] = f[0, :]
    f[0, :] = m

    return y0, f


def om_filter(f, m):
    y1 = f[0, :]
    y2 = f[1, :]
    x00 = m
    x11 = f[2, :]
    x22 = f[3, :]

    a1 = 0
    a2 = 7.8387
    a3 = -7.8387
    b1 = 1.0000
    b2 = - 1.5622
    b3 = 0.6413

    if abs(x00[2] - x11[2]) > 1.5 * math.pi:
        if x00[2] > x11[2]:
            x0 = x00 - 2 * math.pi
        elif x00[2] < x11[2]:
            x0 = x00 + 2 * math.pi
    else:
        x0 = x00

    if abs(x11[2] - x22[2]) > 1.5 * math.pi:
        if x11[2] > x22[2]:
            x1 = x11 - 2 * math.pi
        elif x11[2] < x22[2]:
            x1 = x11 + 2 * math.pi
    else:
        x1 = x11

    y0 = 1 * (-b2 * y1 - b3 * y2 + a1 * x0 + a2 * x1 + a3 * x22)

    f = np.array([y0, y1, x00, x11])
    f[:, :2] = 0
    return f


############


class Server:

    def __init__(self):

        self.cutoff_freq = 10
        self.sample_time = 0.01
        self.frame = 0
        self.fl = 0
        self.data = ","
        # vicon is the client, subjectNames are the agent names, Segment names are subgroups of each agent
        self.vicon, self.segmentName, self.subjectNames, self.mover = vicon_init()
        self.xbee, self.remote_devicess, self.remote_names = xbee_init(
            self.subjectNames)  # This returns the xbee device,remote network, agents on the network
        self.plot = True

        self.ref = dict()
        self.vfilter = dict()
        self.rfilter = dict()
        self.pd_vals = dict()
        self.integral_vals = dict()
        self.pdata = dict()
        self.error_vals = dict()

        # perform a check for matching vicon and xbee agents
        self.t0 = time.time()
        self.active_agents = None
        self.nagents = len(self.subjectNames)
        self.t = 0
        self.data = dict()
        self.packet = None
        self.init_var()

        self.stop = False
        # self.live_plot(True)
        self.t = np.zeros((1, 50), dtype=float)

        self.johnny_state_update()
        self.johnny_velocity_control()

        self.filtercycle()

        self.pos = 0
        self.rot = 0
        self.vf = 0
        self.pd = 0
        self.vflp = 0
        self.vd = 0

        self.cycle_update()
        self.cycle_control()

    def johnny_state_update(self):  # updating the state

        self.t[0, :-1] = self.t[0, 1:]
        self.t[0, -1] = time.time() - self.t0
        self.vicon.GetFrame()

        for subName in self.subjectNames:
            pos = np.asarray(self.vicon.GetSegmentGlobalTranslation(subName, subName)[0])
            rot = np.asarray(self.vicon.GetSegmentGlobalRotationEulerXYZ(subName, subName)[0])

            # ref_vrot = self.ref[subName][1]
            # ref_vel = self.ref[subName][0]
            frame_rate = self.vicon.GetFrameRate()
            [vf, pd] = derivative_vel_filter(self.pdata[subName], pos)
            [vflp, vd] = lowpass_vel_filter(self.vfilter[subName], vf)
            self.vf = vf
            self.pd = pd
            self.vflp = vflp
            self.vd = vd
            self.pdata.update({subName: pd})
            pdata = self.pdata
            # pdata_array=pdata[subName] #from mm to cm
            self.vfilter.update({subName: vd})
            self.rfilter.update({subName: om_filter(self.rfilter[subName], rot)})
            self.mover[subName] = np.array([pos, rot, vflp, self.rfilter[subName][0]])
            dt = np.mean(np.diff(self.t))
            # print('update cycle'+' dt: '+str(dt))
            # pdata=pdata/10 #from mm to cm
            # distance=np.linalg.norm(np.array(pdata_array[0,0],pdata_array[0,1])-np.array(pdata_array[11,0],pdata_array[11,1]))

            # print(',   Vel_lowpass : '+ str(np.linalg.norm(vflp[:2])))
            # print(',   Vel_first_order : ' + str(distance*16.67))
            # print(' Pos: ' + str(pos)+ ' Rot: ' + str(rot*180/np.pi))
            # #print(' Pos history: ' + str(pdata))
            # print(' Frame rate : ' + str(frame_rate))

    def johnny_velocity_control(self):  # rate controller
        # sleep(0.02)
        self.t[0, :-1] = self.t[0, 1:]
        self.t[0, -1] = time.time() - self.t0
        self.vicon.GetFrame()

        for subName in self.subjectNames:

            ref_vrot = self.ref[subName][1]
            ref_vel = self.ref[subName][0]
            pos = self.mover[subName][0]
            rot = self.mover[subName][1]
            vflp = self.mover[subName][2]

            T = self.t[0]

            Rz = R.from_euler('z', rot[2], degrees=False).as_matrix()
            v = vflp / 1  # v is 3D vector (u v w)
            vr = self.rfilter[subName][0]

            Ev = self.error_vals[subName][0]
            Ew = self.error_vals[subName][1]

            Ev[:-1] = Ev[1:]
            Ew[:-1] = Ew[1:]
            Ev[-1] = ref_vel[0] - np.linalg.norm(v[:2])


            count = T[-1] % 20




            #############################################################
            # This section is doing the translation from single integrator to unicycle velocities
            # input (u,v) in cm/s
            # output (u,w) in cm/s deg/s

            P_des = np.array([75.4, -8.9, 8.6])  # desired position in cm
            # P_des = self.mover['Johnny07'][0]/10
            P = pos[0:2] / 1000  # in m

            v_des = (P_des - 0.1 * pos)
            v_des = v_des[0:2]  # the third dimension is z
            v_des = v_des



            ref_vel_comm = 0.5 * np.array([np.cos(rot[2]), np.sin(rot[2])]) @ v_des.transpose()
            if ref_vel_comm >= 10:
                ref_vel_comm = 10
            elif ref_vel_comm <= 0:
                ref_vel_comm = 0


            ref_vrot_comm = 0.4 * math.atan2(np.array([-np.sin(rot[2]), np.cos(rot[2])]) @ v_des.transpose(),
                                             np.array([np.cos(rot[2]), np.sin(rot[2])]) @ v_des.transpose()) / (
                                        np.pi / 2)
            ref_vrot_comm = 180 / np.pi * ref_vrot_comm
            if ref_vrot_comm >= 20:
                ref_vrot_comm = 20
            elif ref_vrot_comm <= -20:
                ref_vrot_comm = -20



            ################################



            ref_vrot = self.ref[subName][1]
            ref_vel = self.ref[subName][0]
            print(subName + '  Linear_velocity: ' + str(ref_vel[0]) + '   Angular velocity:  ' + str(
                ref_vrot[2]) + '  Pos  ' + str(pos / 10) + '  P_des  ' + str(P_des))


            # velocity open loop control (neutral input=100)
            data_v = 14.58 * ref_vel[0] + 122

            if abs(data_v - 100) <= 22:
                data_v = 100

            data_v = round(data_v)

            # angular rate open loop control (neutral input=500)
            if abs(ref_vrot[2]) <= 5:
                data_rz = 500
            elif ref_vrot[2] >= 0:
                data_rz = 0.9172 * ref_vrot[2] + 521.1
                data_rz = round(data_rz)
            else:
                data_rz = 0.9337 * ref_vrot[2] + 479.2
                data_rz = round(data_rz)
            # cw rot=1.069*data_rz-512.5 deg/s
            # data_rz = 500
            self.error_vals.update({subName: np.array([Ev, Ew])})
            # proportional
            Kpv = 10
            Kpw = 0.05

            # derivative
            Kdv = 0.1
            Kdw = 0.5

            # integral
            Kiv = 0.1  # 0.01
            Kiw = 0.01  # 0.005

            Kkv = 50
            Kkw = 1

            Kpv = 10
            Kpw = 6  # 2

            # derivative
            Kdv = 0.1
            Kdw = 0.3  # 0.3

            # integral
            Kiv = 0.1  # 0.01
            Kiw = 0.05  # 0.05

            Kkv = 50
            Kkw = 1

            Kpv = 10
            # Kpw = 2

            # derivative
            Kdv = 0.1
            # Kdw = 0.3

            # integral
            Kiv = 0.1  # 0.01
            # Kiw = 0.05 #0.005

            Kkv = 50
            Kkw = 1

            Ew[-1] = (ref_vrot[2] - vr[2])  # transfer error to degree

            # velocity controller
            # data_v = data_v     +     (Kpv*Ev[-1] + Kdv * (Ev[-1] - Ev[-2]) + Kiv * sum(Ev))/500

            # angular rate controller (normal reference value + )
            data_rz = data_rz + Kpw * Ew[-1] + Kdw * (Ew[-1] - Ew[-2]) + np.minimum(Kiw * 1, Kiw * sum(Ew))
            data_rz = round(data_rz)

            if (data_v > 260):
                data_v = 260

            if (data_rz > 600):
                data_rz = 600

            if (data_rz < 400):
                data_rz = 400

            # print('control cycle')
            self.data.update({subName: [data_v, data_rz]})

    @threaded
    def get_agents(self):
        # get agent names
        return self.subjectNames

    def get_frame(self):
        print('Vicon frame')

    def send_data(self):
        # broadcast
        i = 0
        for name in self.remote_names:
            # print('iter' )
            # print(i)
            # print(self.mover[name])
            # print(name)
            # d = str(self.mover[name][0])+ ',' + str(self.mover[name][1])
            d = str(self.data[name][0]) + "," + str(self.data[name][1])
            print(name + '  d= ', d)
            self.xbee.send_data_async(self.remote_devicess[i], d)
            # print('DATA')
            # print(name)
            # print(self.remote_devicess[i])
            # print(d)
            i = i + 1

    @threaded  # this thread is for updating the data
    def cycle_update(self):
        while (True):
            self.johnny_state_update()
            self.send_data()
            if self.stop == True:
                break

    @threaded  # this thread is only for control and sending the data
    def cycle_control(self):
        while (True):
            self.johnny_velocity_control()
            if self.stop == True:
                break

    def get_estimate(self):
        # print('')
        # return np.array([self.t, self.mover])
        return self.mover

    def end(self):

        print('Motor switched off')

    def start(self):
        print('Motor switched on')

    def battery_status(self):
        print('Battery Value')

    def init_var(self):
        for name in self.subjectNames:
            # Robot.ref[name] = np.array([[0,0,0],[0,0,0]])
            self.ref.update({name: np.array([[0, 0, 0], [0, 0, 0]])})
            self.vfilter.update({name: np.zeros((10, 3))})
            self.pdata.update({name: np.zeros((20, 3))})
            self.rfilter.update({name: np.zeros((4, 3))})

            # store proportional and derivative errors
            self.error_vals.update({name: np.zeros((2, 50))})

            # store integral errors
            self.integral_vals.update({name: np.zeros((2, 50))})

    def filtercycle(self):
        for i in range(100):
            self.johnny_state_update()
            self.johnny_velocity_control()

        for name in self.subjectNames:
            self.mover[name] = np.zeros((4, 3))

        veps = 1
        reps = 1
        while veps > 0.1 and reps > 0.1:
            # eps = 0
            self.johnny_state_update()
            self.johnny_velocity_control()

            for name in self.subjectNames:
                # eps = eps + np.linalg.norm(self.vfilter[name][0]) + np.linalg.norm(self.rfilter[name][0])
                veps = np.linalg.norm(self.vfilter[name][0])
                reps = np.linalg.norm(self.rfilter[name][0])
                # print(veps)
                # print(reps)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('PyCharm')

    Robot = Server()

    pos = np.array([[0, 0, 0, 0, 0, 0]])
    vel = np.array([[0, 0, 0, 0, 0, 0]])

    rot = np.array([[0, 0, 0, 0, 0, 0]])
    rvel = np.array([[0, 0, 0, 0, 0, 0]])

    sp = []

    eps = 100
    t0 = time.time()
    print("start")
    print(t0 - time.time())
    D = 1000
    T_history = [0]
    # data = np.zeros((1,10000))
    counter = 1
    while (time.time() - t0 < D):
        t = time.time()
        T = time.time() - t0

        counter += 1

        for name in Robot.subjectNames:


            a = Robot.get_estimate()
            # print(a)
            p = a[name][0][:2]
            v = a[name][2][0]  # convert to m/s
            r = a[name][1][2]
            w = a[name][3][2]

            # print(v)

            # print(r)

            Kv = 1
            Kw = 1
            ep = [0, 0] - p
            ref_v = Kv * np.linalg.norm(ep)
            ref_w = Kw * (np.arctan2(ep[1], ep[0]) - r)

            # circle
            wd = 0
            # vx = 0.5*wd*math.sin(wd*T)
            # vy = 0.5*wd*math.cos(wd*T)
            v = 0
            # ref_v = 1000
            ref_v = 0
            ref_w = 0
            # data[:,counter] = T
            # print('t '+str(T),' T list   ' + str(np.transpose(T_history)))


            # the first three elements are the linear velocities and the last three are the angular velocities
            u = 0
            w = 10
            Robot.ref['Johnny07'] = np.array([[u, u, u], [w, w, w]]) # ex: [u u u],[w w w]

            # print(type(name))
            # Robot.ref['Johnny08'] = np.array([[0, 0.0, 0.0], [0.0, 0.0, 0]])
    Robot.plot = False
    Robot.stop = True

    # np.savetxt("file.txt",data)

    print("average", np.mean(np.diff(Robot.t[1:])))

    # stop the robot
    for name in Robot.subjectNames:
        Robot.ref[name] = np.array([[0, 0.0, 0.0], [0.0, 0.0, 0.0]])

    t = time.time()
    while (time.time() - t < 1):
        Robot.johnny_state_update()
        Robot.johnny_velocity_control()
        Robot.send_data()

        # live_update_demo(False) # 28 fps
