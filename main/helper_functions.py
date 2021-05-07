'''
Created on Nov 7, 2020

@author: Luis, carlos, yoshi, jack
'''

#def create_zero_array(args):
#    if len(args)==4:
#        a = [[[[0. for x in range(args[3])] for x in range(args[2])] for x in range(args[1])] for x in range(args[0])]
#    if len(args)==2:
#        a = [[0. for x in range(args[1])] for x in range(args[0])]
#    return a

#def sum_RGB_frame(a, num_pix):
    

#def sum_RGB_frame_array(a, num_pix):
#    ret_arr = create_zero_array((len(a), 3))
#    for frame in range(len(a)):
#        ret_arr[frame] = sum_RGB_frame(a[frame],num_pix)
#    return ret_arr
def localmax(signal):
    for i in range(1, len(signal)-1):
