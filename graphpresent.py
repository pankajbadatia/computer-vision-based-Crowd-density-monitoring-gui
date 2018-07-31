#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 16:16:50 2018

@author: pankaj
"""

import matplotlib

import matplotlib.cm
import imageio
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.misc


import matplotlib.animation as animation
import time


def generate_plot_video(data_directory):
    sns.set_style('darkgrid')
    labels = np.load(os.path.join(data_directory, 'predicted_labels.npy'), mmap_mode='r')
    rois = np.load(os.path.join(data_directory, 'rois.npy'), mmap_mode='r')
    length = labels.shape[0]
    roi_labels = labels * rois
    counts = roi_labels.sum(axis=(1, 2))
    counts = savitzky_golay(counts, 7, 3)
    plot_writer = imageio.get_writer(os.path.join(data_directory, 'plot.mp4'), fps=50)
    for index in range(length):
        print('\rPlotting {}...'.format(index), end='')
        figure, axes = plt.subplots(dpi=300)
        axes.set_xlim(0, length)
        axes.set_ylim(0, counts.max())
        axes.plot(np.arange(index), counts[:index], color=sns.color_palette()[0])
        axes.set_ylabel('Person Count')
        axes.set_xlabel('Frame')
        figure.canvas.draw()
        image = np.fromstring(figure.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        image = image.reshape(figure.canvas.get_width_height()[::-1] + (3,))
        plot_writer.append_data(image)
        plt.close(figure)
    plot_writer.close()

def savitzky_golay(y, window_size, order, deriv=0, rate=1):

    import numpy as np
    from math import factorial

    window_size = np.abs(np.int(window_size))
    order = np.abs(np.int(order))
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')




labels1 = np.load('/media/pankaj/D04BA26478DAA96B/PANKAJ/testpresent/104207 Time Lapse Demo/predicted_labels.npy', mmap_mode='r')
rois1 = np.load('/media/pankaj/D04BA26478DAA96B/PANKAJ/testpresent/104207 Time Lapse Demo/rois.npy', mmap_mode='r')
length1 = labels1.shape[0]
roi_labels1 = labels1 * rois1
counts1 = roi_labels1.sum(axis=(1, 2))
counts1 = savitzky_golay(counts1, 7, 3)
print('finished 1')



labels2 = np.load('/media/pankaj/D04BA26478DAA96B/PANKAJ/testpresent/200608 Time Lapse Demo/predicted_labels.npy', mmap_mode='r')
rois2 =np.load('/media/pankaj/D04BA26478DAA96B/PANKAJ/testpresent/200608 Time Lapse Demo/rois.npy', mmap_mode='r')
length2 = labels2.shape[0]
roi_labels2 = labels2 * rois2
counts2 = roi_labels2.sum(axis=(1, 2))
counts2 = savitzky_golay(counts2, 7, 3)
print('finished 2')


labels3 = np.load('/media/pankaj/D04BA26478DAA96B/PANKAJ/testpresent/200702 Time Lapse Demo/predicted_labels.npy', mmap_mode='r')
rois3 =np.load('/media/pankaj/D04BA26478DAA96B/PANKAJ/testpresent/200702 Time Lapse Demo/rois.npy', mmap_mode='r')
length3 = labels3.shape[0]
roi_labels3 = labels3 * rois3
counts3 = roi_labels3.sum(axis=(1, 2))
counts3 = savitzky_golay(counts3, 7, 3)
print('finished 3')


labels4 = np.load('/media/pankaj/D04BA26478DAA96B/PANKAJ/testpresent/500717 Time Lapse Demo/predicted_labels.npy', mmap_mode='r')
rois4 =np.load('/media/pankaj/D04BA26478DAA96B/PANKAJ/testpresent/500717 Time Lapse Demo/rois.npy', mmap_mode='r')
length4 = labels4.shape[0]
roi_labels4 = labels4 * rois4
counts4 = roi_labels4.sum(axis=(1, 2))
counts4 = savitzky_golay(counts4, 7, 3)
print('finished 4')



indexer = np.arange(3000)



np.save('data1.npy', counts1)

np.save('data2.npy', counts2)

np.save('data3.npy', counts3)

np.save('data4.npy', counts4)

#






































def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    window_size = np.abs(np.int(window_size))
    order = np.abs(np.int(order))
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')