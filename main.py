from timeit import default_timer as timer
from imageai.Detection import VideoObjectDetection
import serial
import os
import cv2

# get current file directory
execution_path = os.getcwd()

# initialize camera
camera = cv2.VideoCapture(0)

video_detector = VideoObjectDetection()
video_detector.setModelTypeAsYOLOv3()
video_detector.setModelPath(os.path.join(execution_path, "yolo.h5"))
video_detector.loadModel()

# create custom objects
objects = video_detector.CustomObjects(person=True, car=True, motorcycle=True, bus=True, truck=True)

# initialize car detected to False
car_present = False

# initialize start time
start_time = 0

# initialize the time it takes to send a message to the authorities
valid_time = 5.0

# initialize arduino communication
arduino = serial.Serial(port='COM4', baudrate=9600, timeout=.1)

def forFrame(frame_number, output_array, output_count, returned_frame):

    global car_present
    global valid_time
    global start_time

    # get frame size
    height, width, n = returned_frame.shape
    half_size = int(width/2)

    # save frame 
    cv2.imwrite('car-frame.jpg', returned_frame)
    
    # count number of cars present in 'No Parking' Area
    car_count = 0

    # get all the objects detected on the frame
    for eachCar in output_array:
        # get the points of the bounding box of the detected object
        x1, y1, x2, y2 = eachCar['box_points']

        if x1 < half_size and x2 < half_size:
            car_count += 1

    if car_count != 0 and car_present:
        end_time = timer()
        elapsed_time = end_time - start_time
        if elapsed_time > valid_time:
            print('Message sent')
            arduino.write(bytes('1', 'utf-8'))
    elif car_count != 0 and not(car_present):
        start_time = timer()
        car_present = True
    else:
        start_time = 0
        car_present = False
        arduino.write(bytes('0', 'utf-8'))

# initialize video object detection
video_detector.detectObjectsFromVideo(camera_input=camera, 
                                            custom_objects=objects, 
                                            save_detected_video=False,  
                                            frames_per_second=5, 
                                            per_frame_function=forFrame,  
                                            minimum_percentage_probability=30, 
                                            return_detected_frame=True)