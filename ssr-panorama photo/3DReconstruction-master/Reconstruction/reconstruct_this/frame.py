import numpy as np
import cv2
import random
import time
import glob
import imutils

cap = cv2.VideoCapture(0)

def getFrame(sec,frameRate):

    time.sleep(frameRate)
    path_img_name = "/home/nicho/Downloads/ssr-20211022T203842Z-001/ssr-panorama photo/3DReconstruction-master/Reconstruction/database/frame_{}.jpg".format(sec)
    hasFrames, frame = cap.read()

    img = cv2.imread(path_img_name)
    if hasFrames:
        cv2.imwrite(path_img_name, frame)     # save frame as JPG file
        cv2.imshow("img", frame)
    return hasFrames



sec = 0
frameRate = (float(1)/float(10)) #//it will capture image in each 0.5 second
success = getFrame(sec,frameRate)

while success:

    sec = sec + frameRate
    sec = round(sec, 2)
    
    if sec >= (60*frameRate):
        sec = 0

    success = getFrame(sec, frameRate)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()