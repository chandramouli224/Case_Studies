import numpy as np
import cv2
import mobilenet
import os
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
import keras
from keras.applications import imagenet_utils
import threading
import time
import logging
import random
import queue
from threading import Thread
import time
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing import image_dataset_from_directory
import pandas as pd
#initializing buf size
BUF_SIZE = 10
q = queue.Queue(BUF_SIZE)
final_results = []



#this line is used to load pretrained yolo algorithm
net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')
# net = cv2.dnn.readNet('yolov3-tiny.weights','yolov3-tiny.cfg')
classes = []

#this will load mobile net model
model = mobilenet.train_model()

#this line reads all class names for yolo algorithm
with open('coco_classes.txt','r') as f:
  classes = [line.strip() for line in f.readlines()]


#reading all layers of yolo
layer_names = net.getLayerNames()
#reading output layers
outputlayers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers() ]

#below code reads images from train and validation folders as train and validation datasets
train_dataset = image_dataset_from_directory("./data/",
                                             shuffle=True,
                                             batch_size=32,
                                             image_size=(224,224))
validation_dataset = image_dataset_from_directory("./validation/",
                                                  shuffle=True,
                                                  batch_size=32,
                                                  image_size=(224,224))
class_names = train_dataset.class_names

#below code is used to train mobileNet model and save it as model.h5
# model.fit(train_dataset, epochs=25,validation_data=validation_dataset)
# model.save('model.h5')
model = keras.models.load_model('model.h5')


#below code reads ground truth file.
df = pd.read_excel("Groundtruth.xlsx")
SUV_GT = df["SUV"].values
SEDAN_GT = df["Sedan"].values
TOTAL_GT = df["Total"].values
FRAMES = df["Frame#"].values

suv_t=[]
sedan_t=[]
total_t=[]

## Create a video capture object
cap = cv2.VideoCapture('assignment-clip.mp4')
##
### Check video properties
print('FPS: \t\t'+str(cap.get(cv2.CAP_PROP_FPS)))
print('No. of Frames: \t'+str(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
print('Frame width: \t'+str(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
print('Frame height: \t'+str(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))


##Object detector function detects the object using pretrained yolo model
def object_detector(frame):
    class_ids = []
    confidences = []
    boxes = []
    height, width, channels = frame.shape
    img = frame
    # blobFromImage function is used to preprocess(mean subtraction, normalizing, and channel swapping) the images before sending it to NN
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    # this out list of lists that contains the detected objects 1st four elements are center(x,y), height and width of bounding box of detected object
    # rest of the elements are scores for each class
    outs = net.forward(outputlayers)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                # calculating object center(x,y), w, h with respect to actual size of the image
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width + 20)
                h = int(detection[3] * height + 20)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return boxes,confidences,class_ids,frame
font = cv2.FONT_HERSHEY_PLAIN

#object classifier is used for classifying the objects that are detected by object detector
#it takes boxes for each object, confidence score, class_ids, indexs from NMSBoxes function as input and returns number of sedan and suv cars present in a particular frame
def object_classifier(boxes,confidences,class_ids,indexes,frame):
    suv = 0
    sedan = 0
    img=frame
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            # if label=='car':
            if label == 'car':
                # detected objects are cropped so that it we feed that particular object as input to mobilenet for object classification
                frame = img[y:y + h + 10, x:x + w + 10]
                if frame.size != 0:
                    # 224,224 is one of the standard img size that mobilenet supports
                    # print(frame.size)
                    frame = cv2.resize(frame, (224, 224))
                    preprocessed_image = mobilenet.prepare_image(frame)
                    # predicting the class of the cropped object
                    predictions = model.predict(preprocessed_image)
                    # print(predictions)
                    if predictions > 0.5:
                        results = 'SEDAN'
                        sedan = sedan + 1
                    else:
                        results = "SUV"
                        suv = suv + 1
                    #below lines are used to create rectangles and text on detected objects with their confidence
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    cv2.putText(img, results+" :"+str(confidences[i]), (x, y + 30), font, 1, (0, 0, 250))
    return suv,sedan

#producer function initiates object detector and puts the output of each frame into queue
def producer(frame):
    q.put(object_detector(frame))

#below consumer function will process the items in queue that are processed by producer.
#it takes detected objects params and triggers object_classifier to classify objects in our case cars.
def consumer():
    while True:
        #reading items in queue
        boxes,confidences, class_ids, frame  = q.get()
        #NMSBoxes will reduce number of boxes on single object to one box
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        #object classifier is called here
        suv, sedan = object_classifier(boxes, confidences, class_ids, indexes,frame)
        #total number of cars detected text is printed on each frame
        cv2.putText(frame, str(sedan + suv) + " cars detected", (10, 10), font, 1, (0, 0, 250))
        #suv, sedan and total number of cars detected in a single frame are stored for further analysis
        suv_t.append(suv)
        sedan_t.append(sedan)
        total_t.append(suv+sedan)
        cv2.imshow('test', frame)
        print("frame: " + str(j) + " sedan: " + str(sedan) + " suv: " + str(suv) + " total: " + str(sedan + suv))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        q.task_done() #end of consumer method

#initiating timer to calculate throughput
t1 = time.time()

#initiating 7 threads for consumers
for i in range(7):
    t = Thread(target=consumer)
    t.daemon = True
    t.start()

#this loops over each frame in the video
for j in range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
# for j in range(0, 10):
    # Capture a frame
    ret, frame = cap.read()
    #producer is triggered
    producer(frame)
    q.join()

t2 = time.time()
print("throughput : ",(t2 - t1))

from sklearn.metrics import f1_score,accuracy_score
print(TOTAL_GT)
fig, ax = plt.subplots()
ax.plot(total_t, label="predictions")
ax.plot(TOTAL_GT,label = "ground truth")
ax.legend()
plt.show()
output = pd.DataFrame(
    {'Sedan': sedan_t,
     'SUV': suv_t,
     'total': total_t
    })
writer = pd.ExcelWriter('output1.xlsx')
output.to_excel(writer, 'output')
writer.save()

