"""hum_aligner.py: Resize a hum audio to match its corresponding speech audio.

The non-silent parts of the hum audio is extracted as intervals.
The word intervals in the speech audio, taken using the speech textgrid, are
matched to a hum segment using the length-based aligner Gale and Church algorithm;
it is possible for many words to be mapped to a single hum segment.
The total length of the words mapped to a hum segment is used to resize the hum segment
using psola. Half a hanning window is applied to 10 msec of each ends of each resized hum
segments to make transitions smooth, and the segments are overlapped by 10 msec.

Errors in alignment can happen when the length of the word segments and hum segments
are drastically different in cases where there is a long continuous hum made up of multiple
hummed words. These errors can be detected by checking if the size of the speech audio
is drastically different from the resized hum audio.

Author: Tredayne B. Cabanlit
"""

import librosa
from lib.Audio import WAVReader, WAVWriter
from librosa.effects import time_stretch
import numpy as np
import sys
from nltk.translate.gale_church import align_blocks
import textgrids

class HumAligner:
    """
    Given a speech textgrid, the hum audio will try to match the word onset and offset.

    Parameters
    ----------
    speech_textgrid : str
        A praat textgrid that has an interval tier called 'word' with silence at the
        beginning and end.
    hum_wav : str
        A file path to the hum wav files.
    """
    def __init__(self, speech_textgird, hum_wav):
        hum_wav = WAVReader(hum_wav)
        self.hum_data = hum_wav.getData()
        self.hum_data = np.reshape(self.hum_data, (len(self.hum_data),))
        self.hum_fs = hum_wav.getSamplingRate()
        self.hum_period = 1/self.hum_fs
        self.speech_textgrid = textgrids.TextGrid(speech_textgrid)
        self.norm_factor_speech = 0.05 #estimated duration in sec of a single syllable word
        self.norm_factor_hum = 0.15 #estimated duration in sec of a single syllable hum

    def split_hum(self):
        """
        Extract the non-silent parts of the hum wav.

        Returns
        -------
        intervals : ndarray; array of arrays
            An array of intervals. Each interval consists of start and end index.
        """
        n = len(self.hum_data)
        data = np.reshape(self.hum_data, (n,))

        intervals = librosa.effects.split(data, top_db=20)

        ##discard interval if less than norm_factor or 0.05 seconds
        #discard_idxs = []
        #intervals = list(intervals) #convert to list so we can pop
        #for i in range(len(intervals)):
            #if (intervals[i][1] - intervals[i][0]) < (0.05 / self.hum_period):
                #discard_idxs.append(i)

        #discard_idxs.sort(reverse=True) #so indeces don't get messed up by pop

        #for idx in discard_idxs:
            #intervals.pop(idx)

        #intervals = np.array(intervals)

        return intervals
    
    def intervals2block(self, intervals):
        """
        Creates an array of interval length. Interval index length is
        taken and converted to seconds and divided by norm_factor.
        
        Parameters
        ----------
        intervals : ndarray; array of arrays
            Array containing arrays of start and end of hum interval.

        Returns
        -------
        block : ndarray
            Array of interval length values normalized to be a small int.
        """
        block = []
        n = len(intervals)

        for i in range(n):
            block.append(intervals[i][1] - intervals[i][0])
        
        #convert length to seconds and normalize
        block = np.rint(np.array(block)*self.hum_period/self.norm_factor_hum)

        return block

    def textgrid2block(self):
        """
        Parse the textgrid and extract the word interval lengths. Create
        an array of the normalized lengths.

        Returns
        -------
        block : ndarray
            Array of word interval length from textgrid and normalized.
        """
        block = []
        tier = 'word'

        #ignore beginning and ending silence in speech audio in range()
        for i in range(1,len(self.speech_textgrid[tier])-1):
            block.append(self.get_duration(tier,i))
        
        block = np.rint(np.array(block)/self.norm_factor_speech)

        return block

    def match_blocks(self, speech_block, hum_block):
        """
        Use Gale and Chuch algorithm to match speech block to hum block.

        Parameters
        ----------
        speech_block : ndarray
            array of word lengths as integers.
        hum_block : ndarray
            array of hum interval lengths as integers.

        Returns
        -------
        alignments : list; list of tuples
            A list of tuples where each tuple contains index from speech block
            and the index in the hum block that it's mapped to.
        """
        alignment = align_blocks(speech_block, hum_block)

        return alignment

    def get_duration(self, tier, index):
        """
        Get the duration in seconds of an interval in a textgrid tier.

        Parameters
        ----------
        tier : str
            The name of the tier.
        index : int
            The index of the interval in the tier whose duration we want.

        Returns
        -------
        duration : int
            The length of the interval.
        """
        interval = self.speech_textgrid[tier][index]
        duration = interval.xmax - interval.xmin
        
        return duration

    def equalize_hum(self):
        """
        Get the hum and speech word intervals and map them to each other. Then, resize
        the hum intervals based on the length of the word intervals mapped to them.
        Apply a hanning window to the resized hum intervals and concatenate them.

        Returns
        -------
        s_synt : tuple; tuple(ndarray, float)
            The new resized hum signal and its sample rate.
        """
        hum_intervals = self.split_hum()
        hum_intervals_sec = hum_intervals * self.hum_period

        hum_block = self.intervals2block(hum_intervals)
        speech_block = self.textgrid2block()
        alignment = self.match_blocks(speech_block, hum_block)

        speech_durs = np.zeros(len(hum_intervals))

        #get total duration for speech segment(s) matched to a hum segment
        for speech_idx, hum_idx in alignment:
            speech_durs[hum_idx] += self.get_duration('word', speech_idx+1) #ignore silence interval

        hum_durs = [x[1]-x[0] for x in hum_intervals_sec]
        gammas = hum_durs / speech_durs

        len_10msec = int(0.01 / self.hum_period) #10 msec as index length
        rising_filter = np.hanning(len_10msec*2)[0:len_10msec]
        falling_filter = np.hanning(len_10msec*2)[len_10msec:len_10msec*2]

        #add beginning silence
        start_silence = self.get_duration('word', 0)
        #need extra 10msec because will be overlapped with next interval
        s_synt = np.zeros(round(start_silence/self.hum_period)+len_10msec)

        for i in range(len(gammas)):
            interval = self.hum_data[hum_intervals[i][0] : hum_intervals[i][1]]

            #length after time stretching + 0.01 sec on each end
            interval_len_after = (len(interval)/gammas[i] * self.hum_period + 0.02)
            new_gamma = len(interval)*self.hum_period / interval_len_after
            new_hum = time_stretch(y=interval, rate=new_gamma)

            #apply waning filter on 0.01 sec on each edge
            filtr = np.concatenate([rising_filter, np.ones(len(new_hum)-(len_10msec*2)), falling_filter])
            new_hum = new_hum*filtr
            
            #create 10 msec overlap using zero padding
            padded_s_synt = np.pad(s_synt, (0, len(new_hum)-len_10msec),'constant',constant_values=0)
            padded_new_hum = np.pad(new_hum, (len(s_synt)-len_10msec, 0),'constant',constant_values=0)
            s_synt = padded_s_synt + padded_new_hum #element-wise addition

        #add ending silence
        end_silence = round(self.get_duration('word', len(self.speech_textgrid['word'])-1),2)
        n = round(end_silence/self.hum_period)
        s_synt = np.pad(s_synt,(0, n-len_10msec),'constant',constant_values=0)
        
        #make into a column vector
        s_synt = np.reshape(s_synt, (len(s_synt),1))

        return (s_synt, self.hum_fs)

if __name__ == '__main__':
    number = sys.argv[1]
    path = '/home/trey/Projects/hum_aligner/'
    speech_textgrid = path + 'data/speech_' + str(number).zfill(3) + '.TextGrid'
    hum_wav = path + 'data/hum_' + str(number).zfill(3) + '.wav'
    new_hum_wav = path + 'data/matched_hum_' + str(number).zfill(3) + '.wav'

    hum_align = HumAligner(speech_textgrid, hum_wav)
    s_synt,fs = hum_align.equalize_hum()

    writer = WAVWriter(new_hum_wav, s_synt, fs=fs)
    writer.write()

