import cv2
import serial

from datetime import datetime
from timeit import default_timer as timer

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

# load model
model = cv2.dnn_DetectionModel(frozen_model, config_file)

classLabels = []
file_name = 'labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')
    
model.setInputSize(320, 320)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# load camera
camera = cv2.VideoCapture(0)

# set frame size
width = 1280
height = 720
camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# line properties
vertical = int(width / 2)
horizontal = int(height * 0.13)
line_color = (0, 255, 0) # green
line_thickness = 3

if not camera.isOpened():
    raise IOError('Cannot Open Camera')

font_scale = 2
font = cv2.FONT_HERSHEY_PLAIN

# initialize car detected to False
car_present = False

# initialize start time
start_time = 0.0

# initialize the time it takes to send a message to the authorities
valid_time = 170

message_sent = False

# number of frames
frame_count = 0

# initialize video saving
size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('car-detection.avi', fourcc, 20.0, size)

# initialize arduino communication
arduino = serial.Serial(port='/dev/ttyUSB0', baudrate=9600, timeout=.1)

print('Car Detection Activated')

while True:
    ret, frame = camera.read()
    
    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.5)
    
    car_count = 0
    if (len(ClassIndex) != 0):
        for index, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            # Index Equivalent
            # 3 - car
            # 4 - motorbike
            # 6 - bus
            # 8 - truck
            if ((index <= 80) and (index == 1 or index == 3 or index == 4 or index == 6 or index == 8)):
                x1, y1, x_width, y_height = boxes
                x2 = x1 + x_width
                y2 = y1 + y_height
                
                if x2 < vertical and y1 > horizontal:
                    # display bounding box
                    cv2.rectangle(frame, boxes, (0, 0, 255), 2)
                    
                    # display label
                    cv2.putText(frame, classLabels[index-1], (x1+10, y1+40), font, fontScale=font_scale, color=(0, 255, 255))
                    
                    # increment car count
                    car_count += 1
    
    # if cars are detected on the 'No Parking'area and previously there are no cars on it,
    # start the timer and set car_present = True
    if car_count != 0 and not(car_present):
        print('Start Detecting ....')
        start_time = timer()
        car_present = True
        
    # if there are still cars detected on the 'No Parking' area,
    # check if the time elapsed is > than the valid_time
    elif car_count != 0 and car_present:
        # get current time
        current_time = timer()
        # get elapsed time
        elapsed_time = current_time - start_time
        elapsed_time = round(elapsed_time, 2)
        msg = f'Elapsed Time: {elapsed_time}s'
        print(msg)
        # display elapsed time
        cv2.putText(frame, msg, (vertical + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        if elapsed_time > valid_time and not(message_sent):
            message_sent = True
            # display 'Message Sent'
            cv2.putText(frame, 'Message Sent', (vertical + 20, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            # send some data to arduino through serial communication
            arduino.write(bytes('1', 'utf-8'))
        
    # reset timer
    else:
        print('No Car(s) Detected')
        start_time = 0
        car_present = False
        message_sent = False
        
    if car_count != 0:
        # display warning
        cv2.putText(frame, 'Ilegally parked car(s) detected', (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        
    today = datetime.now()
    curr_time = today.strftime('%m %d, %Y   %I:%M:%S %P')
    
    # display date and time
    cv2.putText(frame, curr_time, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # display vertical line
    cv2.line(frame, (vertical, 0), (vertical, height), line_color, line_thickness)
    
    # display horizontal line
    cv2.line(frame, (0, horizontal), (width, horizontal), line_color, line_thickness)
    
    # save frame into image
    # image_name = 'frame%d.jpg' % (frame_count)
    # cv2.imwrite(image_name, frame)
    # frame_count += 1
    
    # save video
    out.write(frame)
    
    # cv2.imshow('Car Detection', frame)
    
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
    
camera.release()
cv2.destroyAllWindows()