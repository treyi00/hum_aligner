# DSP_Tools.py -- a set of functions required for basic DSP
# 
# normaliseRMS: adjust root-mean-squre (RMS) of the signal to the target RMS.
# snr: compute speech-to-noise ratio of given speech and noise signals.
# rms: compute RMS of the signal
# 
# Python (c) 2020 Yan Tang University of Illinois at Urbana-Champaign (UIUC)
#
# Created: March 14, 2019
# Updated: November 11, 2019
# Updated: Feburary 11, 2020 - Updated mkDataStructFMT to support 32-bit
# Updated: April 23, 2020 - Added LPC implementation
# Updated: Junuary 13, 2023 - Added mean HNR implementation


import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt


def __sign(x):
    if x >= 0:
        return 1
    else:
        return -1

def corrcoef(x, y):
    d_x = x - np.mean(x)
    d_y = y - np.mean(y)
    return  (np.dot(d_x, d_y)) / (np.sqrt( np.sum(d_x ** 2) ) * np.sqrt( np.sum(d_y ** 2) ))

def normaliseRMS(x, tarRMS):
    # detech the number of channels in the signal
    # enable the function to deal with signals that contain more than one channel
    dim = np.shape(x) 
    nb_sample = np.max(dim) # number of samples
    nb_chan = np.min(dim)   # number of channels

    k = tarRMS * np.sqrt(nb_sample * nb_chan / (np.sum(x**2)))
    return k * x, k

def snr(s, n):
    return 10 * np.log10(np.sum(s**2) / np.sum(n**2))

def setSNR(s, n, SNR_tar):
    SNR_c = snr(s, n)
    k = pow(10, (SNR_c - SNR_tar)/20)

    return n * k, k


def rms(x):
    # detech the number of channels in the signal
    # enable the function to deal with signals that contain more than one channel
    dim = np.shape(x)
    nb_sample = np.max(dim) # number of samples
    nb_chan = np.min(dim)   # number of channels

    return np.sqrt(np.sum(x**2)/(nb_sample * nb_chan))

def scalesig(x, scalar=1.1):
    return x / (scalar * np.max(np.abs(x)))

# Make packing format for number-to-byte conversion and vise versa
def mkDataStructFMT(bits, total_samples):

    if bits == 8: 
        tp = "b" # 1-byte signed char
    elif bits == 16:
        tp = "h" # 2-byte signed shorts
    elif bits == 32:
        tp = "i" # 4-byte signed integer
    else:
        raise ValueError("Only supports 8-, 16-bit and 32-bit audio formats.")

    return "{}{}".format(total_samples, tp)


# Implementation of Convolution of two waveforms
# take the input waveform x and impluse response h
def convolve(x, h, durationMatch=False):
    len_x = len(x)
    len_h = len(h)
    len_y = len_x + len_h - 1

    # initialise an empty waveform for convoluation  
    y = np.zeros((len_y, 1), dtype=np.float_)

    # match the lengh of the output vector
    data_x = np.concatenate((x, np.zeros((len_y - len_x, 1), dtype = np.float)), axis=0)
    data_h = np.concatenate((h, np.zeros((len_y - len_h, 1), dtype = np.float)), axis=0)

    # for each output sample in y
    idx_x = [i for i in range(0, len_y)]
    idx_h = np.flip(idx_x)
    for n in range(0, len_y):
        # tmp = []
        # for k in range(0, len_y):
        #     tmp.append(data_x[k] * data_h[n - k])
        # y.data[n] = np.sum(tmp)

        ri = np.roll(idx_h, n+1) # shift for time delay
        y[n] = np.sum(data_x[idx_x] * data_h[ri]) 
        
    # return the convolved sequence
    return y

def energy(x):
    return np.sum(x**2)

def power(x):
    return energy(x)/len(x)


def ZCR(x):
    nb_sample = len(x)

    tmp = 0
    for idx in range(1, nb_sample):
        tmp += np.abs(__sign(x[idx]) - __sign(x[idx-1]))

    return tmp / (2 * nb_sample)

def findEndpoint(sig, fs, win_size=0.02, threshold_en=50, threshold_zcr=0.05):
    nb_sample = len(sig)
    nb_spfrme = round(fs * win_size)
    nb_frame = int(np.ceil(nb_sample / nb_spfrme))

    nb_sample2 = nb_spfrme * nb_frame
    nb_pad = nb_sample2 - nb_sample
    if nb_sample2 > 0:
        sig = np.pad(sig, ((0, nb_pad), (0, 0)))

    wins = np.reshape(sig, (nb_frame, nb_spfrme)).T

    win_fn = np.hanning(nb_spfrme)
    en = np.zeros((1, nb_frame))
    zcr = np.zeros((1, nb_frame))
    for idx in range(nb_frame):
        seg = wins[:, idx] * win_fn + np.finfo(float).eps
        en[0, idx] = energy(seg)
        zcr[0, idx] = ZCR(seg)

    en_db = 20 * np.log10(en / np.sqrt(nb_spfrme))   
    lc_en = np.max(en_db) - threshold_en
    lc_zcr = threshold_zcr

    # b_en = (en_db < lc_en)
    # b_zcr = (zcr < lc_zcr)
    isSil = ((en_db < lc_en) & (zcr < lc_zcr))

    return isSil

def instPeriodicity(sig, lagstep=1, win_type="rectangular", fn="ac", normalisation=True):
    nb_sample = len(sig)
    if nb_sample % 2:
        sig = np.pad(sig, ((0, 1), (0, 0)))
        nb_sample += 1

    nb_win = int(nb_sample/2)

    if win_type == "hamming":
        win = np.hamming(nb_win).reshape(-1, 1)
    elif win_type == "rectangular":
        win = np.ones((nb_win, 1))
    else:
        raise ValueError("Only support hamming ('hamming') and rectangular ('rect') window!")

    base = sig[0:nb_win] * win
    
    steps = range(0, nb_win, lagstep)
    nb_step = len(steps)
    pdt = np.zeros((nb_step,))
    for step in range(nb_step):
        frame_move = sig[step:step+nb_win,:] * win
        if fn.lower() == "ac":
            pdt[step] = np.dot(base.T, frame_move)
        elif fn.lower() == "amdf":
            pdt[step] = np.sum(np.abs(base - frame_move))
        else:
            raise ValueError("Only support autocorrelatoin ('ac') and average magnitude difference function ('amdf')1")

        # if normalisation is done
        if normalisation:
            pdt[step] = pdt[step] / np.sqrt(np.sum(base ** 2) * np.sum(frame_move ** 2))

    return pdt, base

def __centreClip(sig, threshold=0.3, mode="normal"):
    ## Find the greatest absolute value in sig as the peak value
    val_max = np.max(np.abs(sig))

    ## Determine the threshold
    CL = threshold * val_max

    clipped = sig.copy()
    if mode.lower() == "normal":
        # For samples above CL, the output is equal to the input sample minus the clipping level.
        # For samples below CL, the output is zero.
        # clipped[np.where(np.abs(sig)>CL)[0]] = clipped[np.where(np.abs(sig)>CL)[0]]
        clipped = (np.abs(sig) > CL) * (sig - np.sign(sig) * CL)

    elif mode.lower() == "3level":
        # SIGNAL(n)>CL  +1;
        # SIGNAL(n)<-CL -1;     
        # Otherwise 0
        clipped = (np.abs(sig) > CL) * np.sign(sig)

    else:
        raise ValueError("Incorrect clipping mode! Use 'normal' or '3level'.")

    return clipped


def __findFirstPeak(s, thrld=0.4):
    ## Remove the first element where signal sigal is compared without delay
    s = np.delete(s, 0)
    ## A quick min-max normalisation
    s  = (s - np.min(s)) / (np.max(s)+np.finfo('float').eps - np.min(s))
    s[s< np.max(s)*thrld] = 0

    for i in range(1, len(s)-1):
        if s[i-1] < s[i] > s[i+1]:
            break
    
    if i == (len(s) - 2):
        i = 0
    else:
        i += 1  ## compensate the sample removed in the beignning

    return i


def meanHNR(sig, fs, size_win=0.015, overlap=0.5):
    ## determine the analytic window size
    nb_sample = len(sig) 
    nb_spframe = round(fs * size_win)
    nb_shift = round( nb_spframe * (1 - overlap)) # number of samples to shift
    ## Figure out the starting index of each window with overlap
    idx_frame = np.array([i for i in range(0, nb_sample, nb_shift)])

    """
    Avoide artefacts due to zero padding
    Discard a number of windows at the end of the signal where padding-zeros is required to make enough sampels for given window size
    This is done by checking the ending index of a window, given the actual size is size_win * 2 for autocorrelation
    """
    idx_frame = idx_frame[np.where(idx_frame + nb_spframe*2 < nb_sample)]
    nb_frame = len(idx_frame)

    ## Traverse all the windows
    HNRs = np.zeros((nb_frame,1))
    ACs = np.zeros((nb_frame,1))
    for f in range(0, nb_frame):
        data_win = sig[idx_frame[f]:idx_frame[f] + nb_spframe*2] # Feed as twice many samples as the in an effecive analytice window

        ## Compuate the AC for the curent window
        ac = instPeriodicity(data_win, win_type="rectangular")[0]
        ## perform centre clipping to enhance the target peak 
        clipped = __centreClip(ac, 0.7, 'normal')
        ## locate target peak loction
        m = __findFirstPeak(clipped)
        # print(m)
        # plt.plot(clipped)
        # plt.show()
        if m == 0:
            # ACs[f] = np.finfo('float').eps
            ACs[f] = np.nan
        else:
            ACs[f] = np.abs(ac[m+1])

        # HNRs[f] = 10*np.log10(ACs[f]/(1-ACs[f]))

    ## Compute mean HNR:np.mean(HNRs)
    AC_avg = np.nanmean(ACs)
    return 10*np.log10(AC_avg/(1-AC_avg)), AC_avg


def whiteNoise(nb_sample, amp=0.9):
    return np.random.uniform(-amp, amp, (nb_sample, 1))

