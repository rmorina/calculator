import neural_network as network
import copy
import numpy
import cv2
import time
from segmentation import get_segments
from PIL import Image, ImageOps

FRAME_FREQUENCY = 1000
last_frame_time = 0

cap = cv2.VideoCapture(0)
count = 0
segment_boxes = []

while(True):
    ret, camera_frame = cap.read()
    frame = copy.deepcopy(camera_frame)
    cv2.rectangle(camera_frame, (400,300), (900,600), (255,255,255),3)
    for seg in segment_boxes:
        cv2.rectangle(camera_frame, (400+seg[1], 300+seg[0]), (400+seg[3], 300+seg[2]), (0,0,255),2)
    frame_time = int(round(time.time() * 1000))
    rgb = cv2.cvtColor(camera_frame, cv2.COLOR_BGR2BGRA)
    cv2.imshow('camera_frame', rgb)
    if frame_time - last_frame_time > FRAME_FREQUENCY:
        cropped = frame[300:600, 400:900]
        gray_cropped = Image.fromarray(cropped).convert('L')
        row_avg = numpy.average(numpy.asarray(gray_cropped))
        avg = numpy.average(row_avg)
        print avg
        if avg > 165:
            count += 1
        if avg > 165 and count > 20:
            """
            thread this!
            """
            cv2.imwrite('capture.png', cropped)
            resized_images, segments, images = get_segments('capture.png')
            print segments
            segment_boxes = segments
            for i in xrange(len(images)):
                cv2.imwrite('segment-'+str(i)+'.png', numpy.asarray(images[i]))
            last_frame_time = frame_time
            count = 0
            net = network.Network([784, 30, 13])
            net.load_parameters()
            for i in resized_images:
                print net.recognize(i)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
