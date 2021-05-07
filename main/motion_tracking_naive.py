import cv2


class MotionTrackerNaive:
    def __init__(self):
        # Init the video capture and take the first image
        self.cap = cv2.VideoCapture(0)
        _, self.img = self.cap.read()

        # Manually draw a bounding box in the img then press Enter
        # type(bbox) = <class 'tuple'>, NOT a list
        # bbox = (x-coord of origin, y-coord or origin, width, height)
        self.bbox = cv2.selectROI("Tracking", self.img, False)

        # Instantiate and initialize the tracker
        self.tracker = cv2.TrackerMOSSE_create()
        self.tracker.init(self.img, self.bbox)

    def draw_box(self):
        x, y, w, h = int(self.bbox[0]), int(self.bbox[1]), int(self.bbox[2]), int(self.bbox[3])
        print(x, y, w, h)
        cv2.rectangle(self.img, (x, y), (x + w, y + h), (255, 0, 255), 3, 1)
        cv2.putText(self.img, "Tracking", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def run(self):
        while True:
            timer = cv2.getTickCount()
            success, self.img = self.cap.read()

            success, self.bbox = self.tracker.update(self.img)

            if success:
                self.draw_box()
            else:
                cv2.putText(self.img, "Lost", (75, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            cv2.putText(self.img, str(int(fps)), (75, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Tracking", self.img)

            if cv2.waitKey(1) & 0xff == ord('q'):
                self.clean_up()
                break

    def clean_up(self):
        self.cap.release()
        cv2.destroyAllWindows()
