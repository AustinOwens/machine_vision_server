# Machine Vision Server

This repo consists of two applications:

- machine_vision_client.py
- machine_vision_server.py

## machine_vision_client.py

The app typically runs on a users laptop/desktop and is responsible for sending machine
vision config params to the machine_vision_server.py application. When ran, it will
pop up a window that allows the user to click and drag anywhere on the screen to select
a region of interest that should be tracked. The tracking is based on HSV thresholding.

This app needs to be ran after the machine_vision_server.py is already running on the
desired target.

## machine_vision_server.py

The machine vision typically runs on a headless Linux server (i.e. a Linux OS that
only has shell acess). This server runs the OpenCV camshifting algorithm, inverse 
kinematics algorithms, and controls the ODrive motor controllers. This application
was tested on the Zybo Z7-20 board.
