'''
Created on Nov 7, 2020

@author: Luis, carlos, yoshi, jack
'''
import cv2
import numpy as np

faceCascade = cv2.CascadeClassifier("../sources/haarcascade_frontalface_default.xml")


class FaceDetector:

    def __init__(self):
        self.intitial = True

    def crop_img(self, img, rect):
        if len(rect) > 0:
            x = rect[0]
            y = rect[1]
            w = rect[2]
            h = rect[3]
            return img[y:y+h, x:x+w]
        return img

    def face(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return faceCascade.detectMultiScale(img_gray, 1.1, 2, minSize=(130, 130))

    def apply_skin(self, img):
        lower = np.array([0, 48, 80], dtype="uint8")
        upper = np.array([20, 255, 255], dtype="uint8")

        converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        skin_mask = cv2.inRange(converted, lower, upper)
        skin = cv2.bitwise_and(img, img, mask=skin_mask)
        return skin

    # for (x, y, w, h) in self.rect:
        # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # fourthx = int((w) / 4)

        # draws face region of interest
        # cv2.rectangle(img, (x + fourthx, y), (x + 3 * fourthx, y + fourthx), (255, 0, 0), 2)
        # cv2.rectangle(img, (x + int(fourthx / 2), y + fourthx * 2), (int(x + fourthx * 1.5), int(y + fourthx * 3)),
        #              (255, 0, 0), 2)
        # cv2.rectangle(img, (int(x + fourthx * 2.5), y + fourthx * 2),
        #              (int(x + fourthx * 2.5 + fourthx), int(y + fourthx * 3)), (255, 0, 0), 2)