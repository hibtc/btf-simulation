# -*- coding: utf-8 -*-

"""
Tools for the analysis of the BTF simulation
"""

import numpy as np
from scipy import interpolate

def magnitude_phase_difference(t, s1, s2, f):
    """
    Implementation of the In-phase and quadrature components to extract the 
    amplitude and phase of two signals from a certain frequency component
    Author: P. Niedermayer

    :param t: array of sampling times
    :param s1: array of signal 1 real values
    :param s2: array of signal 2 real values
    :param f: frequency of interest
    :return: relative magnitude and phase of signal 2 compared to signal 1 at given frequency
             returned phase is in radians and in range [-pi, +pi]
    """
    # IQ demodulation
    import matplotlib.pyplot as plt
    sin = np.sin(2*np.pi*f*t)
    cos = np.cos(2*np.pi*f*t)
    C1 = np.trapz(s1*cos) + 1j*np.trapz(s1*sin)
    C2 = np.trapz(s2*cos) + 1j*np.trapz(s2*sin)    
    return np.abs(C2)/np.abs(C1), (np.angle(C1)-np.angle(C2)+np.pi)%(2*np.pi)-np.pi
    #(np.angle(C2)-np.angle(C1)+np.pi)%(2*np.pi)-np.pi
    

def fft_mag_phase(signal, f):
    """
    Implementation of the FFT analysis of the BTF signal to
    retrieve the magnitude and the phase

    :param signal: array of the signal
    :param f: frequency of interest
    :return: relative magnitude and phase of signal 2 
             returned phase is in radians and in range [-pi, +pi]
    """
    freq = np.fft.rfftfreq(len(signal)) 
    fft  = np.fft.rfft(signal)

    fft_inter_mag = interpolate.interp1d(freq, abs(fft), kind='linear')
    fft_inter_phi = interpolate.interp1d(freq, np.angle(fft), kind='linear')

    return fft_inter_mag(f), fft_inter_phi(f) 

