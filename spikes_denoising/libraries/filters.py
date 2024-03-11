#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: %(Val-Calvo, Mikel and Alegre-Cortés, Javier)
@emails: %(mikel1982mail@gmail.com, jalegre@umh.es)
@institutions: %(Dpto. de Inteligencia Artificial, Universidad Nacional de Educación a Distancia (UNED), Postdoctoral Researcher Instituto de Neurociencias UMH-CSIC)
"""
from scipy.signal import butter, sosfilt, iirnotch, filtfilt

def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfilt(sos, data)
    return y
    
def butter_lowpass(data, cut, fs, order=5):
    sos = butter(order, cut, 'low', fs=fs, output='sos')
    y = sosfilt(sos, data)
    return y
    
def butter_highpass(data, cut, fs, order=3):
    sos = butter(order, cut, 'hp', fs=fs, output='sos')
    filtered = sosfilt(sos, data)
    return filtered

def notch_50Hz(sig, fs, w0=50.0,Q=30.0):
    b, a = iirnotch(2 * w0/fs, Q)
    return filtfilt(b, a, sig) 
