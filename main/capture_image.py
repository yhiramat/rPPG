'''
Created on Nov 7, 2020

@author: Luis, carlos, yoshi, jack
'''

import cv2
import time
from face_detection import FaceDetector
# from motion_tracking import MotionTracker
# from multiprocessing import Process, Pipe
import numpy as np
from threading import Thread

# __init__ and the __call__ are the only methods that can be called from outside of this class instance
class RetrieveIMG:
    def __init__(self, batch_size=270, tracking_on=False):
        self.size = batch_size
        self.face_detector = FaceDetector()
        self.faces = []
        self.img = None
        self.images = []
        self.stop = False

        # variables used for motion tracking
        self.tracking_on = tracking_on
        self.bbox = None
        self.tracker = None

    def __call__(self, pipe, src):
        self.pipe = pipe
        # self.pipe_to_track = pipe_to_track
        self.videoc = cv2.VideoCapture(src)
        self.frame_count = 0
        self.start_time = None  # Updated every 30 frames in __count_frame()

        masking_thread = Thread(target=self.process_frame)
        masking_thread.start()

        print("bef rn")
        self.run()
        masking_thread.join()

    # Must be called only by __call__()
    def run(self):
        time.sleep(1)   # What's this for?
        self.start_time = time.time()
        success, _ = self.videoc.read()
        while success:
            k = cv2.waitKey(1)
            if k != -1:
                self.clean()
                break
            if k & 0xff == ord('q'):
                self.pipe.send(None)
                break
            success, _img = self.videoc.read()
            if success:
                self.count_frame()
                # self.process_frame()
                cv2.imshow('image', _img)
                self.images.append(_img)
            else:
                print("RetrieveIMG.run(): Failed to read. Or end of the src video.")
        self.clean()

    def process_frame(self):
        while not self.stop:
            # print("process_frame whle loop")
            if len(self.images) == 0:
                time.sleep(.01)
                continue
            self.img = self.images.pop(0)
            self.get_faces()

            masked_img = self.mask_img()

            # Send the image to Image Processor thru pipe. Display the image for user.
            self.pipe.send([masked_img])

        # for (x, y, w, h) in self.faces:
        #     cv2.rectangle(self.img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #     break
        # self.img = cv2.resize(self.img, (455, 255))

        #cv2.imshow('image', self.img)

    def get_faces(self):
        if len(self.faces) == 0: #or self.frame_count % 1 == 270:
            # This block must be processed at the very first frame, before else-block
            #print("facedetect")
            self.faces = self.face_detector.face(self.img)
        elif self.tracking_on:
            success, self.bbox = self.tracker.update(self.img)  # bbox includes float
            if self.tracking_on and success:
                x, y, w, h = int(self.bbox[0]), int(self.bbox[1]), int(self.bbox[2]), int(self.bbox[3])
                #print("success motion tracking at: ", np.asarray([[x, y, w, h], ]), " for ", self.bbox)
                self.faces = np.asarray([[x, y, w, h], ])
        #print(self.faces, "\n", len(self.faces))

    def mask_img(self):
        # masked_img = self.img
        masked_img = np.zeros((256, 256, 3))
        if len(self.faces) > 0:
            if self.tracker is None and self.tracking_on:
                self.start_motion_tracking(self.faces[0])
            cropped_img = self.face_detector.crop_img(self.img, self.faces[0])
            masked_img = cv2.resize(self.face_detector.apply_skin(cropped_img), (256, 256), cv2.INTER_AREA)
            # cv2.imshow('cropped', cropped_img)
        else:
            masked_img = cv2.resize(self.face_detector.apply_skin(self.img), (256, 256), cv2.INTER_AREA)
        cv2.imshow("masked", masked_img)
        return masked_img

    def start_motion_tracking(self, target=None):
        x, y, w, h = target
        self.bbox = (x, y, w, h)
        # self.tracker = cv2.TrackerTLD_create()        # FPS = 7.9, BPM = 54
        self.tracker = cv2.TrackerMOSSE_create()      # FPS = 31, BPM = 54 ~ 95, still works best
        # self.tracker = cv2.TrackerMIL_create()        # FPS = 10+, BPM = -
        # self.tracker = cv2.TrackerMedianFlow_create() # FPS = 32, BPM = -, error
        # self.tracker = cv2.TrackerKCF_create()        # FPS = 12, BPM = 54 ~ 88
        # self.tracker = cv2.TrackerCSRT_create()       # FPS <= 10, also causes error in the middle
        self.tracker.init(self.img, self.bbox)

    def count_frame(self):
        if self.frame_count % 30 == 29:
            print(f'\rFPS: {30 / (time.time() - self.start_time)}')
            self.start_time = time.time()
        self.frame_count += 1

    def clean(self):
        self.pipe.send(None)
        cv2.destroyAllWindows()
        self.videoc.release()
        self.stop = True
