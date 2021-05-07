import cv2
from threading import Thread


# TypeError: cannot pickle 'cv2.VideoCapture' object
#   i.e. Only single process can access to the capture (cv2.VideoCapture())
#   Thus, multiple cv2.VideoCapture object cannot exist
class MotionTracker:
    def __init__(self, img=None, target=None):
        print("MotionTracker __init__()")
        # unpack face (type: ndarray from capture_image.py) for bbox (must be tuple)
        x, y, w, h = target
        self.bbox = (x, y, w, h)
        self.img = img

        # Instantiate and initialize the tracker
        self.tracker = cv2.TrackerMOSSE_create()
        self.tracker.init(self.img, self.bbox)

    def __call__(self):
        print("MotionTracker __call__()")
        thread = Thread(target=self.run)
        thread.start()

    def draw_box(self):
        x, y, w, h = int(self.bbox[0]), int(self.bbox[1]), int(self.bbox[2]), int(self.bbox[3])
        # print(x, y, w, h)
        cv2.rectangle(self.img, (x, y), (x + w, y + h), (255, 0, 255), 3, 1)
        cv2.putText(self.img, "Tracking", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def run(self):
        while True:
            print("Tracking motion")
            timer = cv2.getTickCount()
            success, self.img = self.cap.read()

            success, self.bbox = self.tracker.update(self.img)

            if success:
                self.draw_box()
                # return np.array(self.bbox)
            else:
                cv2.putText(self.img, "Lost", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # return NULL

            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            cv2.putText(self.img, str(int(fps)), (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Tracking", self.img)

            if cv2.waitKey(1) & 0xff == ord('q'):
                self.clean_up()
                break

    def clean_up(self):
        self.cap.release()
        cv2.destroyAllWindows()
