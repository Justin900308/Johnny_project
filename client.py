# Credit: pymotw.com/2/socket/tcp.html


import socket
import sys
import numpy as np
import csv
import pandas as pd
# Create a TCP/IP socket; Parameters include address family and socket type
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


# Connect the socket to the port where the server is listening
server_address = ('192.168.0.114', 23)

print(sys.stderr, 'connecting to %s port %s' % server_address)
# Parameter include host and port; connect the client to the socket
sock.connect(server_address)

try:

    # # Send data
    # message = 'This is the message.  It will be repeated.'
    # print(sys.stderr, 'sending "%s"' % message)
    # # Code changed from message to message.encode() so that type is bytes and not a string
    # sock.sendall(message.encode())
    #
    # # Look for the response
    # amount_received = 24
    # amount_expected = len(message)
    #
    Data_PNT = np.array([[0, 0, 0, 0, 0, 0]])
    count = 0
    while True:
# # while amount_received < amount_expected:

        # Recieve data
        data = 0
        data = sock.recv(1000000)
        # amount_received += len(data)
        #print(sys.stderr, 'received "%s"' % data)
        out=data.decode()
        modified_string = out.replace('\r', ' ').replace('\n', '').replace(',', '')
        str_list = modified_string.split()
        #print(modified_string)
        int_list = [int(num) for num in str_list]
        #print(str_list)
        arr = np.array(int_list)
        arr = arr[:6]
        arr = np.array([arr])
        if arr.size == 0:
            arr=np.array([[0, 0, 0, 0, 0, 0]])
        print(arr)
        Data_PNT = np.concatenate((Data_PNT,arr),axis=0)
        #connection, client_address = sock.accept()
        # Break if there is no more data being sent
        #connection = 0
        #connection, client_address = sock.accept()
        # if connection == 0:
        #     print('Connection Break')
        #     break
        count = count +1
        if count >= 5000:
            break



finally:
    print(sys.stderr, 'closing socket')
    sock.close()
    dT = pd.DataFrame(Data_PNT)
    dT.to_csv('DATA1001.csv', index=False)
