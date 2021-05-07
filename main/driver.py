'''
Created on Nov 7, 2020

@author: Luis, carlos, yoshi, jack
'''
import cv2
#from main import extract_signal
from process_image import ProcessImage
import time
from multiprocessing import Process, Pipe
from capture_image import RetrieveIMG


def get_source():
    ans = int(input('enter 1 for live, enter 2 for file name : '))
    while ans not in [1, 2]:
        print("invalid input:")
        ans = int(input('enter 1 for live, enter 2 for file name : '))
    if ans == 1:
        return 0
    else:
        return input("enter file destination : (default = '../Helper/Luis_20sec_day.mp4')") or \
               "../Helper/Luis_20sec_day.mp4"


if __name__ == '__main__':
    source = get_source()
    time1 = time.time()

    # Connect two processes, capturing image and processing it, with pipe
    cap_conn_end, proc_conn_end = Pipe()
    process_img = ProcessImage()
    mask_processor = Process(target=process_img, args=(proc_conn_end, source,), daemon=True)
    mask_processor.start()

    capture = RetrieveIMG(tracking_on=True)
    capture(cap_conn_end, source)
    # capture(cap_conn_end, cap_conn_end2, source)

    mask_processor.join()
    # mt_process.join()
    time2 = time.time()
    print(f'time {time2-time1}')
