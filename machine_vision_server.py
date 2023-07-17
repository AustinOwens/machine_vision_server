#!/usr/bin/env python3
"""
Created on Jun 17, 2022

@author: Austin
"""

import math as m
import pickle
import signal
import socket
import struct
import threading

import cv2
import numpy as np


class SocketServerThread(threading.Thread):
    def __init__(self, ip, port, name):
        threading.Thread.__init__(self, name=name)

        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.s.bind((ip, port))
        self.s.listen(5)

        self.reestablishing_conn = False
        self.kill_thread = False

    def accept(self):
        conn, addr = None, None

        try:
            conn, addr = self.s.accept()

            # Set blocking to non-blocking after the first blocking accept() so that once
            # connection from client is lost, server will continue to process images without
            # requireing a client connection.
            self.s.setblocking(False)

        except BlockingIOError:
            pass

        except OSError as e:
            # Errno 22 is Invalid arg. It happens when I shutdown the socket with SIGINT.
            if e.args[0] == 22:
                pass

            # Errno 9 is Bad file descriptor. It can happen when I shutdown the socket with SIGINT.
            elif e.args[0] == 9:
                pass

            else:
                raise

        return conn, addr

    def run(self):
        # Wait for client to connection
        print("{}: Starting".format(self.name))
        conn, addr = self.accept()

        while not self.kill_thread:
            try:
                self.do_work_func(conn, addr)

            except (ConnectionResetError, BrokenPipeError):
                self.reestablishing_conn = True
                print("{}: Waiting for new connection...".format(self.name))

            # Once connection from client is lost, set socket to non-blocking so server can continue to process images without client.
            if self.reestablishing_conn:
                conn, addr = self.accept()
                if conn != None:
                    self.reestablishing_conn = False

        print("{}: Killed".format(self.name))

    def kill(self):
        self.kill_thread = True
        self.s.shutdown(socket.SHUT_RDWR)
        self.s.close()


class SocketVideoCaptureThread(SocketServerThread):
    def __init__(self, ip, port, video_id=0):
        SocketServerThread.__init__(self, ip, port, name="Video_Capture_Thread")

        # Connection to camera
        self.cap = cv2.VideoCapture(video_id)

        # Initialize Camshift Track Window
        self.track_window = (0, 0, 1, 1)  # x, y, width, height

        # HSV Vals
        self.min_hue = 0
        self.max_hue = 180
        self.min_sat = 0
        self.max_sat = 255
        self.min_val = 0
        self.max_val = 255

    def do_work_func(self, conn, addr):
        # Read camera frame
        ret, img = self.cap.read()

        if ret:
            ### STEP 1: ROI Selection ###
            # If new ROI points are available, update HSV filter
            if sock_img_cfg_thread.new_roi_data:
                # Get ROI based on points
                roi_bgr = img[
                    sock_img_cfg_thread.y1 : sock_img_cfg_thread.y2,
                    sock_img_cfg_thread.x1 : sock_img_cfg_thread.x2,
                ]

                # Convert to HSV color scale
                roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

                # Find the min and max HSV within the ROI
                min_hsv = np.amin((np.amin(roi_hsv, 1)), 0)
                max_hsv = np.amax((np.amax(roi_hsv, 1)), 0)

                # Widend boundaries of min and max HSV by some tolerance
                tolerance = 3
                self.min_hue = min_hsv[0] - tolerance if min_hsv[0] >= tolerance else 0
                self.min_sat = min_hsv[1] - tolerance if min_hsv[1] >= tolerance else 0
                self.min_val = min_hsv[2] - tolerance if min_hsv[2] >= tolerance else 0
                self.max_hue = (
                    max_hsv[0] + tolerance if max_hsv[0] <= 180 - tolerance else 180
                )
                self.max_sat = (
                    max_hsv[1] + tolerance if max_hsv[1] <= 255 - tolerance else 255
                )
                self.max_val = (
                    max_hsv[2] + tolerance if max_hsv[2] <= 255 - tolerance else 255
                )

                # Set new_roi_data in sock_img_cfg_thread back to False
                sock_img_cfg_thread.new_roi_data = False

            ### STEP 2: HSV Filtering ###
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            lower_bound = np.array([self.min_hue, self.min_sat, self.min_val])
            upper_bound = np.array([self.max_hue, self.max_sat, self.max_val])
            mask_img = cv2.inRange(
                hsv_img, lower_bound, upper_bound
            )  # Binary image (0 or 255)

            ### STEP 3: EROSION & DILATION ###
            kernel = np.ones((5, 5), np.uint8)
            mask_img = cv2.erode(
                mask_img, kernel, iterations=sock_img_cfg_thread.erosion
            )
            mask_img = cv2.dilate(
                mask_img, kernel, iterations=sock_img_cfg_thread.dilation
            )

            # Adding third axis to make RGB compatible and masking orig img with hsv
            hsv_mask_img = np.bitwise_and(img, mask_img[:, :, np.newaxis])

            ### STEP 4: CAMSHIFT ###
            term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
            track_box, self.track_window = cv2.CamShift(
                mask_img, self.track_window, term_crit
            )
            # print("{}: Camshift: {}".format(self.name, track_box))

            # Makes the ellipse on the image
            cv2.ellipse(hsv_mask_img, track_box, (0, 0, 255), 2)
            cv2.putText(
                hsv_mask_img,
                "({:.1f}, {:.1f})".format(track_box[0][0], track_box[0][1]),
                (int(track_box[0][0]), int(track_box[0][1])),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (0, 255, 0),
            )
            cv2.putText(
                hsv_mask_img,
                "({:.1f}, {:.1f})".format(track_box[1][1], track_box[1][0]),
                (int(track_box[0][0]), int(track_box[0][1] + 15)),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (0, 255, 0),
            )
            cv2.putText(
                hsv_mask_img,
                "({:.1f})".format(track_box[2]),
                (int(track_box[0][0]), int(track_box[0][1] + 30)),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (0, 255, 0),
            )

            ### STEP 5: CALCULATE X, Y, Z ###
            track_box_x = track_box[0][0]
            track_box_y = track_box[0][1]
            track_box_width = track_box[1][1]
            track_box_height = track_box[1][0]
            track_box_orientation = track_box[2]

            # rot_angle = m.radians(track_box_orientation-90) # Subtract 90 deg since 90 degrees is baseline for camshift width
            # roi_w = abs(track_box_width*m.cos(rot_angle)) + abs(track_box_height*m.sin(rot_angle))
            # roi_h = abs(track_box_width*m.sin(rot_angle)) + abs(track_box_height*m.cos(rot_angle))

            a0, a1, a2 = 12, 0, 0  # 12, 12 # Inches
            x_obj_width, y_obj_width = 2.55906, 6.062988  # Inches
            cx, cy = int(img.shape[1] / 2), int(img.shape[0] / 2)
            fx, fy = 635, 633
            c1, c2 = 1500, 200

            if track_box_width >= 1 and track_box_height >= 1:

                Z1 = (x_obj_width / track_box_width) * c1
                Z2 = (y_obj_width / track_box_height) * c2

                Z = (Z1 + Z2) / 2.0
                if Z >= a0 + a1:
                    Z = a0 + a1

                X = (Z / fx) * (track_box_x - cx)
                Y = (Z / fy) * (track_box_y - cy)

                theta1 = m.atan2(Y, X)
                theta2 = (X * m.cos(theta1) + Y * m.sin(theta1)) / -a0
                theta3 = m.asin(Z / a0)

                print(
                    "{}: X: {:.2f}, Y: {:.2f}, Z: {:.2f}, Z1: {:.2f}, Z2: {:.2f}, theta1: {:.2f} theta2: {:.2f} theta3: {:.2f}".format(
                        self.name,
                        X,
                        Y,
                        Z,
                        Z1,
                        Z2,
                        m.degrees(theta1),
                        m.degrees(theta2),
                        m.degrees(theta3),
                    )
                )

            ### STEP 6: PACKAGE IMG AND SEND OVER SOCKET ###
            if conn:
                # Serialize frame
                data = pickle.dumps(hsv_mask_img)

                # Send message length with data
                message_size = struct.pack("L", len(data))
                conn.sendall(message_size + data)


class SocketImgConfigThread(SocketServerThread):
    def __init__(self, ip, port):
        SocketServerThread.__init__(self, ip, port, name="Img_Config_Thread")

        self.erosion = 0
        self.dilation = 0

        self.x1 = 0
        self.y1 = 0
        self.x2 = 0
        self.y2 = 0

        self.new_roi_data = False

    def do_work_func(self, conn, addr):
        if conn:
            img_config_data = conn.recv(4096)

            if img_config_data == b"":
                raise ConnectionResetError

            img_cfg_data_unpacked = struct.unpack(">BBIIII", img_config_data)

            self.erosion = img_cfg_data_unpacked[0]
            self.dilation = img_cfg_data_unpacked[1]

            # If there is new ROI data, update state and set self.new_roi_data flag to True
            if (
                self.x1 != img_cfg_data_unpacked[2]
                or self.y1 != img_cfg_data_unpacked[3]
                or self.x2 != img_cfg_data_unpacked[4]
                or self.y2 != img_cfg_data_unpacked[5]
            ):
                self.new_roi_data = True
                self.x1 = img_cfg_data_unpacked[2]
                self.y1 = img_cfg_data_unpacked[3]
                self.x2 = img_cfg_data_unpacked[4]
                self.y2 = img_cfg_data_unpacked[5]

            print(
                "{}: Erosion: {}, Dilation: {}, x1: {}, y1: {}, x2: {}, y2: {}".format(
                    self.name,
                    self.erosion,
                    self.dilation,
                    self.x1,
                    self.y1,
                    self.x2,
                    self.y2,
                )
            )


# Start Socket Threads
sock_img_cfg_thread = SocketImgConfigThread(ip="", port=9001)
sock_img_cfg_thread.start()

sock_vid_cap_thread = SocketVideoCaptureThread(ip="", port=9002)
sock_vid_cap_thread.start()

# Signal Interrupt
def signal_handler(sig, frame):
    print("{}: Killing Machine Vision Server...".format(__name__))

    sock_vid_cap_thread.kill()
    sock_img_cfg_thread.kill()


signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    print("{}: Waiting for threads to finish...".format(__name__))
    sock_vid_cap_thread.join()
    sock_img_cfg_thread.join()
    print("{}: Killed".format(__name__))
