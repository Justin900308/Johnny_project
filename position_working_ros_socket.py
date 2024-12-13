# Working version
import pickle
import socket
import matplotlib.pyplot as plt
from pynput import mouse
import numpy as np
from scipy import signal
from matplotlib import animation
from numpy import linalg as LA
from vicon_dssdk import ViconDataStream
import argparse
import sys
import cvxpy as cp
from digi.xbee.devices import XBeeDevice
from digi.xbee.devices import RemoteZigBeeDevice
import matplotlib

import matplotlib.pyplot as plt
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
        self.t = np.zeros((1, 50), dtype=float)
        self.dt = []
        self.johnny_state_update()
        self.johnny_velocity_control()
        self.position_transmission()

        self.pos = 0
        self.rot = 0
        self.vf = 0
        self.pd = 0
        self.vflp = 0
        self.vd = 0
        self.P_des = np.array([75.4, -8.9, 8.6])
        self.Coord = []
        self.data_transmission()
        # sleep(2)
        # self.live_plot(True)
        # sleep(2)
        self.filtercycle()

        self.cycle_update()
        self.cycle_control()

        # self.cycle_plot()
        # self.johnny_plot(True)

    def position_transmission(self):
        # Create a socket object (IPv4, TCP)
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Server IP address (replace with the actual server's IP)
        server_ip = '192.168.0.38'  # Replace with server's IP address
        port = 12345

        Pos = self.mover
        try:
            # Connect to the server
            client_socket.connect((server_ip, port))
            #print("Connected to the server.")

            for name in self.subjectNames:

                DATA = self.mover

                # Pos=DATA[0,:]

                # Create a NumPy array to send
                array_to_send = DATA
                # array_to_send = Pos
                # print(f"Sending array:\n{array_to_send}")

                # Serialize the NumPy array

                data = pickle.dumps(array_to_send)

                # Send the serialized data
                client_socket.sendall(data)

                # Receive the server's response
                response = b""
                while True:
                    packet = client_socket.recv(4096)
                    if not packet:
                        break
                    response += packet

                aaa = 3
                # Send a message to the server
                # Deserialize the response
                result = pickle.loads(response)
                self.P_des = pickle.loads(response) / 10
                # print(f"Received result from server: {result}")






        except Exception as e:
            print(f"Error: {e}")
        finally:
            # Close the connection
            client_socket.close()

    def johnny_state_update(self):  # updating the state

        self.t[0, :-1] = self.t[0, 1:]
        self.t[0, -1] = time.time() - self.t0
        self.vicon.GetFrame()
        self.dt.append(self.t[-1] - time.time())
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
            # print(subName + '  Linear_velocity: ' + str(ref_vel[0]) + '   Angular velocity:  ' + str(
            #     ref_vrot[2]) + '  Pos  ' + str(pos / 10) + '  P_des  ' + str(P_des))

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

    def onclick(self, event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        self.Coord = np.array([event.xdata, event.ydata, 8.6])
        print(self.Coord)

    @threaded
    def live_plot(self, blit=False):
        t = np.linspace(0, 50., num=100)
        Vx = np.zeros(t.shape)
        Vr = np.zeros(t.shape)
        x = np.zeros(t.shape)
        y = np.zeros(t.shape)
        th = np.zeros(t.shape)

        fig = plt.figure(figsize=(5, 5))
        ax1 = fig.add_subplot(1, 1, 1)

        line1, = ax1.plot([], 'g.', lw=3)
        line2, = ax1.plot([], 'ro', lw=3)
        ax1.set_xlim([-800, 1500])
        ax1.set_ylim([-800, 1500])
        ax1.set_xlabel('X')
        ax1.set_xlabel('Y')

        fig.canvas.draw()  # note that the first draw comes before setting data

        if blit:
            # cache the background
            ax1background = fig.canvas.copy_from_bbox(ax1.bbox)

        plt.show(block=False)

        # t_start = time.time()
        k = 0.

        # for i in np.arange(10000):
        while (self.plot == True):
            from scipy.ndimage import shift
            # sleep(0.05)

            v = self.mover[self.subjectNames[0]][2]  # convert to 10cm/s
            r0 = self.mover[self.subjectNames[0]][3]

            xx0 = self.mover[self.subjectNames[0]][0]
            th0 = self.mover[self.subjectNames[0]][1]
            mdi = fig.canvas.mpl_connect('button_press_event', self.onclick)
            if self.Coord == []:
                # print("List is empty")
                Pdes = self.P_des * 10

            else:
                # print(self.P_des)
                Pdes = self.Coord[0:2]
                self.P_des = self.Coord
            # Pdes = self.P_des*10
            # print(self.Coord)
            Vx = np.concatenate((Vx[1:], [np.linalg.norm(v)]))
            Vr = np.concatenate((Vr[1:], [r0[2]]))
            x = np.concatenate((x[1:], [xx0[0]]))
            y = np.concatenate((y[1:], [xx0[1]]))
            th = np.concatenate((th[1:], [th0[2]]))

            line1.set_data(x, y)
            line2.set_data(Pdes[0], Pdes[1])

            k += 0.11
            if blit:
                # restore background
                fig.canvas.restore_region(ax1background)

                # redraw just the points
                ax1.draw_artist(line1)
                ax1.draw_artist(line2)

                # coords = plt.ginput(5)
                # fill in the axes rectangle
                fig.canvas.blit(ax1.bbox)


            else:

                fig.canvas.draw()

            fig.canvas.flush_events()

    @threaded
    def get_agents(self):
        # get agent names
        return self.subjectNames

    @threaded
    def on_click(x, y, button, pressed):
        if pressed:
            print(f"Mouse clicked with {button}")
        else:
            print(f"Mouse released with {button}")

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
            # print(name + '  d= ', d)
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

    @threaded
    def data_transmission(self):
        while (True):
            self.position_transmission()
            if self.stop == True:
                break

    @threaded  # this thread is only for control and sending the data
    def cycle_control(self):
        while (True):
            self.johnny_velocity_control()
            if self.stop == True:
                break

    def trajectory_gen(self, P_des, P_ini, obs_center, R, T, Num_agent, Num_obs):

        # Define system matrices
        A = np.array([[0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
        B = np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
        C = np.eye(4)
        D = np.zeros((4, 2))
        Ts = 0.1
        T_end = 12
        # Continuous-time system
        sys = signal.StateSpace(A, B, C, D)

        # Discretize the system
        sysd = sys.to_discrete(Ts)
        Ad = sysd.A
        Bd = sysd.B

        # Initialize state and input arrays
        count = 0
        X = np.zeros((4 * Num_agent, 1))  # State trajectory
        X[:, 0] = np.array([P_ini[0, 0], P_ini[0, 1], 0, 0])  # Initial state

        u = np.empty((2 * Num_agent, 0))  # Initialize u as an empty 2x0 array to store control inputs

        # Compute the initial trajectory
        for t in range(T):
            P_err = P_des - X[0:2, t].T
            u_des = 0.05 * P_err
            u = np.concatenate((u, u_des.T), axis=1)  # Append control input [1, 1]
            # Compute the next state
            X = np.hstack((X, Ad @ X[:, count].reshape(-1, 1) + Bd @ u[:, count].reshape(-1, 1)))
            count += 1

        # Plot the trajectory
        plt.plot(X[0, :], X[1, :], '.')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Trajectory Plot')
        plt.grid(True)
        # plt.show()

        #######################################################################

        alpha = 1
        r_default = 0.5

        lambda_param = 10000

        # Time length N and trajectory state X
        N = X.shape[1]

        # Convert u to a numpy array for proper matrix operations
        u = np.array(u)
        u = np.reshape(u, (2, int(T_end / Ts + 1)))

        # Plot the obstacle (circle)
        theta = np.linspace(0, 2 * np.pi, 201)

        for i in range(int(T)):
            for j in range(Num_agent):
                plt.plot(X[4 * j:j + 1, i], X[4 * j + 1:j + 2, i], 'r.', markersize=2)
                for k in range(Num_obs):
                    x_theta = R[k] * np.cos(theta)
                    y_theta = R[k] * np.sin(theta)
                    plt.plot(obs_center[i, 2 * k:2 * k + 1] + x_theta, obs_center[i, 2 * k + 1:2 * k + 2] + y_theta)

        # Start the iterative optimization process
        linear_cost = np.zeros((201, 1))
        for interation in range(30):

            # Define variables for optimization
            w = cp.Variable((2 * Num_agent, N - 1))
            v = cp.Variable((4 * Num_agent, N - 1))
            d = cp.Variable((4 * Num_agent, N))

            s = cp.Variable((1 * Num_obs, N - 1))

            # Define the cost function
            Linear_cost = 1 * cp.norm(((u + w)), 1) + 1 * lambda_param * cp.sum(
                cp.sum(cp.abs(v))) + 1 * lambda_param * cp.sum(cp.pos(s))

            # Define constraints
            constraints = [d[:, 0] == np.zeros(4)]

            E = np.eye(4)

            for i in range(N - 1):

                for j in range(Num_agent):
                    constraints.append(
                        X[4 * j:j + 4, i + 1] + d[4 * j:j + 4, i + 1] == (
                                Ad @ X[4 * j:j + 4, i] + Ad @ d[4 * j:j + 4, i]) + (
                                Bd @ u[2 * j:j + 2, i] + Bd @ w[2 * j:j + 2, i]) + E @ v[4 * j:j + 4, i])

                    # constraints.append(cp.abs(w[0, i]) <= r_default)
                    constraints.append(w[2 * j:2 * j + 1, i] <= r_default)
                    constraints.append(-r_default <= w[0, i])
                    constraints.append(w[2 * j + 1:2 * j + 2, i] <= r_default)
                    constraints.append(-r_default <= w[1, i])

                    # Obstacle avoidance constraint
                    for k in range(Num_obs):
                        constraints.append(
                            2 * R[k] - cp.norm(X[4 * j:j + 2, i] - obs_center[i, 2 * k:+2 * k + 2], 2) - (
                                    X[4 * j:j + 2, i] - obs_center[i, 2 * k:+2 * k + 2]).T @ (
                                    X[4 * j:j + 2, i] + d[4 * j:j + 2, i] - obs_center[i, 2 * k:+2 * k + 2]) /
                            cp.norm(X[4 * j:j + 2, i] - obs_center[i, 2 * k:+2 * k + 2], 2) <= s[k:k + 1, i + T * j])

                        constraints.append(s[k:k + 1, i + T * j] >= 0)

            # Terminal condition
            constraints.append(X[:, N - 1] + d[:, N - 1] == np.array([P_des[0, 0], P_des[0, 1], 0, 0]))

            # Define the problem
            problem = cp.Problem(cp.Minimize(Linear_cost), constraints)

            # Solve the optimization problem
            problem.solve(solver=cp.CLARABEL)

            # Update the variables after solving
            w_val = w.value
            v_val = v.value
            d_val = d.value
            s_val = s.value
            # U_val = U.value
            j = 1

            linear_cost[interation, 0] = (1 * LA.norm(((u + w_val) * Ts), ord=1) +
                                          1 * lambda_param * np.sum(np.sum(np.abs(v_val))) +
                                          1 * lambda_param * np.sum(s_val))

            rho0 = 0
            rho1 = 0.25
            rho2 = 0.7
            if interation >= 1:
                delta_L = (linear_cost[interation, 0] - linear_cost[interation - 1, 0]) / linear_cost[interation, 0]
            else:
                delta_L = 1
            print(np.abs(delta_L))
            if np.abs(delta_L) <= rho0:
                r_default = np.max((r_default, 0.5))
                X = X + d_val
                u = u + w_val
            elif np.abs(delta_L) <= rho1:
                r_default = r_default
                X = X + d_val
                u = u + w_val
            elif np.abs(delta_L) <= rho2:
                r_default = r_default / 3.2
                X = X + d_val
                u = u + w_val
            else:
                X = X + d_val
                u = u + w_val
                r_default = 0.5

            # Update the trajectory

            # Plot the updated trajectory
            for i in range(Num_agent):
                plt.plot(X[4 * i:i + 1, :], X[4 * i + 1:i + 2, :], 'b.', markersize=2)
                plt.pause(0.001)
            # plt.clf()
            ss = np.zeros((1 * Num_obs, T))

            ss_max = np.array([0])
            for i in range(T):
                for j in range(Num_agent):
                    for k in range(Num_obs):
                        ss[k:k + 1, i + T * j] = LA.norm(X[4 * j:j + 2, i] - obs_center[i, 2 * k:+2 * k + 2], 2) - R[k]

            if (np.min(ss) > 0) and (interation > 10):
                break
            # print(np.min(ss) )
            print('Iteration:  ', interation + 1)

        # Final trajectory plot
        plt.clf()
        return (X, u)

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
            # self.mover[subName] = np.array([pos, rot, vflp, self.rfilter[subName][0]])
            # a is 4*3 matrix [pos],[rot],[vflp],[self.rfilter[subName][0]]]
            p = a[name][0][:2]
            v = a[name][2][0]  # convert to m/s
            r = a[name][1][2]
            w = a[name][3][2]

            # print(counter)

            # if np.mod(counter,100)==0:
            #     # Create a listener for mouse events
            #     with mouse.Listener(on_click=Robot.on_click) as listener:
            #         listener.join()

            # with mouse.Listener(on_click=Robot.on_click) as listener:
            #     listener.join()

            # P_des = np.array([75.4, -8.9, 8.6])

            Pos = a[name][0][:]
            Rot = a[name][1][2]
            P = Pos[0:2] / 1000  # in m
            # Robot.P_des= P_des
            # Robot.position_comm()

            P_des = Robot.P_des
            print('Desired position: '+   str(P_des))

            # obstacle avoidance with APF
            obs_center = np.array([238, 585]) / 1000  # in m
            obs_radi = 0.05  # in m

            # print('rho:    '+ str(rho*100) +'    v_rep: ' + str(v_rep))
            v_des = (P_des - 0.1 * Pos)
            v_des = v_des[0:2]  # the third dimension is z
            v_des = v_des

            # transfer the desired velocity from cartesian coordinate into polar coordinate for unicycle control
            # linear velocity control
            ref_vel_comm = 0.5 * np.array([np.cos(Rot), np.sin(Rot)]) @ v_des.T
            if ref_vel_comm >= 10:
                ref_vel_comm = 10
            elif ref_vel_comm <= 0:
                ref_vel_comm = 0

            # angular velocity control
            ref_vrot_comm = 0.4 * math.atan2(np.array([-np.sin(Rot), np.cos(Rot)]) @ v_des.T,
                                             np.array([np.cos(Rot), np.sin(Rot)]) @ v_des.T) / (
                                    np.pi / 2)
            ref_vrot_comm = 180 / np.pi * ref_vrot_comm
            if ref_vrot_comm >= 20:
                ref_vrot_comm = 20
            elif ref_vrot_comm <= -20:
                ref_vrot_comm = -20

            # print(name + '  Linear_velocity: ' + str(ref_vel_comm) + '   Angular velocity:  ' + str(
            #     ref_vrot_comm) + '  Pos  ' + str(Pos / 10) + '  P_des  ' + str(P_des))

            # print(Rot)

            # # velocity open loop control (neutral input=100)
            # ref_vel_comm = 0
            # ref_vrot_comm = 0

            # ref_v = 1000
            ref_v = 0
            ref_w = 0

            # the first three elements are the linear velocities and the last three are the angular velocities
            # Robot.ref['Johnny07'] = np.array([[ref_v, ref_v, ref_v], [ref_w, ref_w, ref_w]]) # ex: [u u u],[w w w]
            Robot.ref['Johnny08'] = np.array([[ref_vel_comm, ref_vel_comm, ref_vel_comm],
                                              [ref_vrot_comm, ref_vrot_comm, ref_vrot_comm]])  # ex: [u u u],[w w w]
            # print(p)
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
