'''
Created on Nov 7, 2020

@author: Luis, carlos, yoshi, jack
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import sqrt
# research show python math sqrt function faster than numpy

#lower = np.array([0, 48, 80], dtype="uint8")
#upper = np.array([20, 255, 255], dtype="uint8")


class Extractor:
    MinFreq = .8 #48 bpm
    MaxFreq = 3.5 #210 bpm

    def __init__(self, fr, s_s, b_s):
        self.framerate = fr
        self.signal_size = s_s
        self.batch_size = b_s

# CDF algorithm from Color-Distortion Filtering for Remote
    # Photoplethysmography by Wenjin Wang et al. "color distortion filter for [rPPG]"
    def CDF(self, C, B):
        # standard_c = diag(mean(c,2))^-1 * c - 1
        seg_avg = np.diag(np.mean(C, axis=1))
        seg_avg_inv = np.linalg.inv(seg_avg)
        C_S = np.dot(seg_avg_inv, C) - 1
        # Fourier = fft(C_S,[],2)
        F = np.fft.fft(C_S)
        # S = [-1,2,-1]/sqrt(6) * F
        t = np.array([[-1 / sqrt(6), 2 / sqrt(6), -1 / sqrt(6)]])
        S = np.dot(t, F)
        # W = (S * conj(S)) / sum(f*conj(f) and W(:,1:B(1)-1) = 0; W(:,B(2)+1:end)) = 0
        W = np.real((S * S.conj()) / np.sum((F * F.conj()), 0)[None, :])
        W[:, 0:B[0]] = 0
        W[:, B[1]+1:] = 0
        # F_ = F * repmat(W,[3,1])
        F_ = F * W
        iF = (np.fft.ifft(F_)+1).real
        C_ = np.dot(seg_avg, iF)
        return C_.astype(np.float)


    def pulse(self, avg_signal):
        # 3 second period averaging
        seg_batch = int(self.framerate * 3)
        ret_sig = np.zeros(self.signal_size)
        #print(avg_signal)
        #frames_per_beat_range = [int(self.framerate // self.MinFreq), int(self.framerate // self.MaxFreq)]
        beat_per_seg_range = [int(self.MinFreq // (self.framerate/seg_batch)),int(self.MaxFreq // (self.framerate/seg_batch))]
        #avg_signal[0:90].T combines 90 frames into a three arrays containing 90 element
        for i in range(self.signal_size - seg_batch):
            # transpose (90,3) --> (3,90)[ [90 red], [90 green], [90 blue] ]
            group_RGB_seg = avg_signal[i:i+seg_batch, :].T

            group_RGB_seg = self.CDF(group_RGB_seg, beat_per_seg_range)

            # create [[avg_red], [avg_green], [avg_blue]]
            #seg_avg = np.mean(group_RGB_seg, axis=1)
            # create [[avg_red, 0, 0],[0, avg_green, 0], [0 , 0 , avr_blue]]
            # standardize by dividing by respective average using inverse, dot product,
            # and subtracting result by one
            #seg_avg_inv_matrix = np.linalg.inv(np.diag(seg_avg))
            #norm_values = np.dot(seg_avg_inv_matrix, group_RGB_seg)-1

            #print(norm_values[1])

            # cdf

            # z-score standardization
            seg_avg_z = np.mean(group_RGB_seg, axis=1, keepdims=True)
            deviation = np.std(group_RGB_seg, axis=1)
            dev_inv_matrix = np.linalg.inv(np.diag(deviation))
            #print(group_RGB_seg[:, :10])
            #print(seg_avg_z)
            #print(dev_inv_matrix)
            #print(np.diag(deviation))

            norm_values_z = group_RGB_seg - seg_avg_z
            #print(norm_values_z[:, :6])
            norm_values_z = np.dot(dev_inv_matrix, norm_values_z)
            #print(norm_values_z[:, :6])


            proj_matrix = np.array([[0, 1, -1], [-2, 1, 1]])
            # prject norm value 3x90 into 2x90 where first row mainly dependent on green
            s = np.dot(proj_matrix, norm_values_z)
            # bring 2x90 down to 1X90 or 90 element array
            # using the full array of the top row
            # and a proportional adjustment from
            # the standard deviations for the second row.
            std = np.array([1, np.std(s[0, :]) / np.std(s[1, :])])
            p = np.dot(std, s)
            # print(p)
            ret_sig[i:i + seg_batch] = ret_sig[i:i + seg_batch] + (p - p.mean())
            # plt.plot( norm_values[0], 'r', norm_values[2], 'b', norm_values[1], 'g', p, 'y')
            # plt.show()
        # using convolution for moving avg info at
        # https://waterprogramming.wordpress.com/2018/09/04/implementation-of-the-moving-average-filter-using-convolution/
        #plt.plot(ret_sig)
        #plt.show()
        avg_conv = np.ones(6)/6
        ret_sig = np.convolve(ret_sig, avg_conv, 'valid')
        return ret_sig;

    def hr_fft(self, signal):
        signal_size = signal.shape[0]
        fft_signal = np.fft.rfft(signal) #complex number
        fft_signal = np.abs(fft_signal) #sqrt(a^2+b^2) seen on numpy documentation for absoulte of complex numbers
        freq = np.fft.rfftfreq(signal_size, 1./(self.framerate))
        #plt.plot(freq, fft_signal)
        #plt.show()
        #print(fft_signal)
        #print(freq)
        index = np.where((freq < self.MinFreq) | (freq > self.MaxFreq))[0]
        #print(index)
        fft_signal[index] = 0
        #print(fft_signal)
        max_index = np.argmax(fft_signal)
        #plt.plot(freq, fft_signal)
        #plt.show()
        print(freq[max_index])
        return freq[max_index];