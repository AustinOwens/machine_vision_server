#!/usr/bin/env python3
"""
Created on Jun 17, 2022

@author: Austin
"""

import pickle
import socket
import struct
import threading

import cv2

# Enter IP address of device running the machine_vision_server.py
IP_ADDRESS = "<IP_ADDRESS>"


class SocketClientThread(threading.Thread):
    def __init__(self, ip, port, name):
        threading.Thread.__init__(self, name=name)

        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((ip, port))

        self.kill_thread = False

    def kill(self):
        self.kill_thread = True
        self.s.shutdown(socket.SHUT_RDWR)
        self.s.close()


img_config_thread = SocketClientThread(
    ip=IP_ADDRESS, port=9001, name="Img_Config_Thread"
)
img_thread = SocketClientThread(ip=IP_ADDRESS, port=9002, name="Img_Thread")


# ROI Trackbars
def nothing(x):
    pass


cv2.namedWindow("roi_selector", cv2.WINDOW_NORMAL)

cv2.createTrackbar("erosion", "roi_selector", 0, 10, nothing)
cv2.createTrackbar("dilation", "roi_selector", 0, 10, nothing)

# Mouse Click Event
draw_hsv_roi = False
show_hsv_roi_cnt = 0
x1, y1, x2, y2 = 0, 0, 0, 0
roi_x1, roi_y1, roi_x2, roi_y2 = 0, 0, 0, 0


def roi_click_event(event, x, y, flags, params):
    global draw_hsv_roi
    global show_hsv_roi_cnt
    global x1, y1, x2, y2
    global roi_x1, roi_y1, roi_x2, roi_y2

    # Checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1 = x, y
        draw_hsv_roi = True

    # Checking if previously clicked left mouse and is held down
    if draw_hsv_roi:
        # Counter for how long hsv roi box should stay on screen after letting go of mouse
        show_hsv_roi_cnt = 15

        x2, y2 = x, y

    # Checking if let go of left mouse
    if event == cv2.EVENT_LBUTTONUP:

        # Boundary check
        if y1 > y2:
            tmp = y2
            y2 = y1
            y1 = tmp

        if x1 > x2:
            tmp = x2
            x2 = x1
            x1 = tmp

        if y1 == y2:
            y2 += 1

        if x1 == x2:
            x2 += 1

        roi_x1 = x1
        roi_y1 = y1
        roi_x2 = x2
        roi_y2 = y2

        draw_hsv_roi = False


cv2.setMouseCallback("roi_selector", roi_click_event)

# Img Data for STEP 1
data = b""
payload_size = struct.calcsize("L")

# Previous data send over socket for STEP 2
prev_send_data = [0] * 6

while True:

    ### STEP 1: RETRIEVE IMG OVER SOCKET FROM SERVER ###
    # Retrieve message size
    while len(data) < payload_size:
        data += img_thread.s.recv(4096)

    packed_msg_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack("L", packed_msg_size)[0]

    # Retrieve all data based on message size
    while len(data) < msg_size:
        data += img_thread.s.recv(4096)

    frame_data = data[:msg_size]
    data = data[msg_size:]

    # Extract frame and display
    server_img = pickle.loads(frame_data)

    ### STEP 2: SEND IMG CFG DATA OVER SOCKET TO SERVER ###
    erosion = cv2.getTrackbarPos("erosion", "roi_selector")
    dilation = cv2.getTrackbarPos("dilation", "roi_selector")

    # Only send data over socket to server if it is new data
    send_data = [erosion, dilation, roi_x1, roi_y1, roi_x2, roi_y2]
    if send_data != prev_send_data:
        img_config_thread.s.sendall(struct.pack(">BBIIII", *send_data))
    prev_send_data = send_data

    ### STEP 3: DRAW ROI HSV FILTER BOX ###
    if show_hsv_roi_cnt >= 0:
        cv2.rectangle(
            server_img, (x1, y1), (x2, y2), (0, 0, 255), int(show_hsv_roi_cnt / 4)
        )

        # If not draw_hsv_roi, then slowly fade the rectangle out of existance
        if not draw_hsv_roi:
            show_hsv_roi_cnt -= 5

    cv2.imshow("roi_selector", server_img)

    if cv2.waitKey(1) == 27:
        break

img_thread.kill()
img_config_thread.kill()
