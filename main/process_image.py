'''
Created on Nov 7, 2020

@author: Luis, carlos, yoshi, jack
'''

import numpy as np
import multiprocessing as mp
import time
from threading import Thread, Lock
import matplotlib.pyplot as plt
from extract_signal import Extractor
# import helper_functions as hf

class ProcessImage:

    def __init__(self, sz=270, fs=30, bs=30):
        self.stop = False
        self.batches = []
        self.batch_mean = []
        self.signal_size = sz
        self.batch_size = bs
        self.signal = np.zeros((sz, 3))
        self.extractSignal = Extractor(fs, sz, bs)
        self.rPPG_raw = []
        self.heart_rates = []

    def __call__(self, pipe, source):
        self.pipe = pipe
        self.source = source
        avg_thread = Thread(target=self.compute_avg)
        avg_thread.start()
        pulse_thread = Thread(target=self.get_signal)
        pulse_thread.start()

        self.rec_frames()
        avg_thread.join()
        # self.plot_graph()

    def rec_frames(self):
        while True and not self.stop:
            data = self.pipe.recv()

            if data is None:
                self.stop = True
                break
            batch = data[0]
            self.batches.append(batch)

    def compute_avg(self):
        current_index = 0
        batch = None
        a = None
        while not self.stop:
            if len(self.batches) == 0:
                time.sleep(.01)
                continue
            mask = self.batches.pop(0)
            if batch is None:   # initialize array with zeroes
                batch = np.zeros((self.batch_size, mask.shape[0], mask.shape[1], mask.shape[2]))

            if current_index < (self.batch_size-1):
                # add to batch until filled
                batch[current_index] = mask
                current_index += 1
                continue
            batch[current_index] = mask
            current_index = 0

            # mask that is 1 wherever batch != 0 and where it is
            pix_mask = np.where(batch != 0, 1, 0)
            non_zero_pix = pix_mask.sum(axis=(1, 2)) #axis 0 across frames, axis 1 row(y), axis 2 col(x)
            total_pixels = batch.shape[1] * batch.shape[2]
            avg_skin_pixels = non_zero_pix.mean()

            m = np.zeros((self.batch_size, 3))
            if (avg_skin_pixels + 1) / (total_pixels) < 0.1:
                print("no face")
                continue
            else:
                m = np.true_divide(batch.sum(axis=(1, 2)), non_zero_pix)
            # print(m)
            self.batch_mean.append(m)

    def get_signal(self):
        extracted_count = 0
        while not self.stop:
            if len(self.batch_mean) == 0:
                time.sleep(.01)
                continue

            avg_batch = self.batch_mean.pop(0)

            if extracted_count >= self.signal_size: # roll over
                # define length of array and input
                size = self.signal.shape[0]
                bsize = avg_batch.shape[0]
                # adjust - getting rid of oldest input and adding new average at the end
                self.signal[0:size - bsize] = self.signal[bsize:size]
                self.signal[size - bsize:] = avg_batch

                pulse_signal = self.extractSignal.pulse(self.signal) # still in time

                # for display purposes
                x = np.arange(0, pulse_signal.shape[0]/30, 1/30)

                self.heart_rates.append(self.extractSignal.hr_fft(pulse_signal))
                print("BPM = ", ((self.heart_rates[-1]) * 60))
                plt.plot(x, pulse_signal, x, np.sin(x*2*np.pi*(self.heart_rates[-1])) * 100)
                plt.show()
            else:
                self.signal[extracted_count: extracted_count + avg_batch.shape[0]] = avg_batch
            extracted_count += avg_batch.shape[0]

    def plot_graph(self):
        # print(self.batch_mean)
        y_r = []
        y_g = []
        y_b = []
        for (r, g, b) in self.signal:
            # print(r)
            y_r.append(r)
            y_g.append(g)
            y_b.append(b)
        # print(y_r)
        # plt.plot(y_g[0:60], 'g', y_r[0:60], 'r', y_b[0:60], 'b')
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
        fig.suptitle('RGB intensity')
        ax1.plot(y_g, 'g', y_r, 'r', y_b, 'b')
        ax1.set_title("all colors")
        ax2.plot(y_g, 'g')
        ax2.set_title('green')
        ax3.plot(y_r, 'r')
        ax3.set_title('red')
        ax4.plot(y_b, 'b')
        ax4.set_title('blue')

        plt.show()
        # plt.pause(.0001)



    #def fast_fourier_np(self, rPPG):
    #    self.rPPG_raw = rPPG
    #    rPPG_fft = np.fft.fft(self.rPPG_raw)
    #    freq = np.fft.fftfreq(60 + 1)
    #    return rPPG_fft, freq