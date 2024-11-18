from time import time
from threading import Thread, Event
from time import sleep
from threading import Lock
import matplotlib.pyplot as plt
import socket
import numpy as np
import pickle


def threaded(fn):
    def wrapper(*args, **kwargs):
        thread = Thread(target=fn, args=args, kwargs=kwargs)
        thread.start()
        return thread

    return wrapper


class Server:

    def __init__(self):

        # self.Johnny_name = ['Johnny01', 'Johnny02', 'Johnny03', 'Johnny04', 'Johnny05',
        #                     'Johnny06', 'Johnny07', 'Johnny08', 'Johnny09', 'Johnny10', ]
        self.Johnny_name = []
        # Create a socket for binding
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Bind to an available IP and port
        self.host = '0.0.0.0'  # Listen on all available network interfaces
        self.port = 12345

        self.stop = False
        self.P_des = np.array([75.4, -8.9, 8.6])
        self.mover = []
        self.Des_coord = np.array([750.4, -80.9, 80.6])

        self.Get_data()
        sleep(5)
        #self.DATA_transmission()
        #self.test()
        self.plot = True
        self.live_plot()
        #sleep(2)

    # self.live_plot(True)

    def onclick(self,event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata))
        self.Des_coord = np.array([event.xdata, event.ydata, 8.6])
        print(self.Des_coord)

    @threaded
    def test(self):

        while (True):
            #sleep(1)
            print('ll')
            if self.stop == True:
                break

    @threaded
    def Get_data(self):

        self.server_socket.bind((self.host, self.port))

        # Start listening for connections (max 5 connections in queue)
        self.server_socket.listen(5)
        print(f"Server listening on {self.host}:{self.port}...")
        while (True):
            #sleep(1)
            #  Accept a new client connectio
            client_socket, client_address = self.server_socket.accept()

            # Receive data from the client
            data = b""
            packet = client_socket.recv(4096)
            data += packet

            # Deserialize the received data into a NumPy array
            self.mover = pickle.loads(data)
            #print(f"Current pos:{self.mover}")
            # received_array = pickle.loads(data)
            # print(f"Current pos:{received_array}")

            # Process the array (e.g., calculate the sum)

           # print(f"Pos to be sent: {self.mover}")
            # result = received_array
            # print(f"Pos to be sent: {result}")

            # Send the result back to the client
            client_socket.send(pickle.dumps(self.Des_coord))

            # Close the client connection
            client_socket.close()

            self.Johnny_name = list(self.mover)[0]

            v = self.mover[self.Johnny_name][2]
            #print(v)
            if self.stop == True:
                break

    @threaded
    def live_plot(self, blit=False):
        t = np.linspace(0, 50., num=500)
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

            v = self.mover[self.Johnny_name][2]  # convert to 10cm/s
            r0 = self.mover[self.Johnny_name][3]

            xx0 = self.mover[self.Johnny_name][0]
            th0 = self.mover[self.Johnny_name][1]
            mdi = fig.canvas.mpl_connect('button_press_event', self.onclick)
            Pdes = self.Des_coord[0:2]
            print(self.Des_coord)
            Vx = np.concatenate((Vx[1:], [np.linalg.norm(v)]))
            Vr = np.concatenate((Vr[1:], [r0[2]]))
            x = np.concatenate((x[1:], [xx0[0]]))
            y = np.concatenate((y[1:], [xx0[1]]))
            th = np.concatenate((th[1:], [th0[2]]))

            line1.set_data(x, y)
            line2.set_data([Pdes[0]], [Pdes[1]])

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


if __name__ == "__main__":
    Jonhhy = Server()
