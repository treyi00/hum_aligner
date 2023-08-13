# Audio.py -- a wrapper of Python Wave_read and Wav_write object, providing simple interface for students to read and write WAV file
#   WAVReader converts the raw data in bytes to amplitude values
#   WAVWriter converts amplitude values into raw data in bytes
# 
# Python (c) 2020 Yan Tang University of Illinois at Urbana-Champaign (UIUC)
#
# Created: August 18, 2019
# Modified: November 09, 2019
# Modified: April 28, 2021: merged WAVReader and WAVWriter into Audio; added support for 24-bit quantaisation level
# Modified: Feburary 15, 2023 - Fixed 8-bit quantisation resolution where unsigned int should be used
#                               Fixed a potential integer overflow

import wave
import struct, sys
import numpy as np
import lib.DSP_Tools as dsp

class WAVReader():
    def __init__(self, wavfile):
        self.filename = wavfile
        self.__wav_obj = wave.open(self.filename, "rb")

        self.__fs = self.__wav_obj.getframerate()
        self.__bits = self.__wav_obj.getsampwidth() * 8
        self.__nb_chan = self.__wav_obj.getnchannels()
        self.__nb_sample = self.__wav_obj.getnframes()
        self.__data = self.__decode()
        
        self.__wav_obj.close()

    # Get the array of amplitude values
    def getData(self):
        return self.__data

    # Get the quantisation resolution in bits 
    def getBitsPerSample(self):
        return self.__bits

    # Get the sampling frequency/rate
    def getSamplingRate(self):
        return self.__fs

    # Get the number of channels in the file
    def getChannelNO(self):
        return self.__nb_chan

    # Get the number of sample points
    def getSampleNO(self):
        return self.__nb_sample

    # Get the duration of the signal in second
    def getDuration(self):
        return self.__nb_sample / self.__fs

    # Core function for coverting bytes to amplitude values
    def __decode(self):
        raw_bytes = self.__wav_obj.readframes(self.__nb_sample)
        total_samples = self.__nb_sample * self.__nb_chan


        # Convert bytes to integers
        if self.__bits == 24:
            print(sys.byteorder)
            
            fmt = dsp.mkDataStructFMT(self.__bits, 1)
            byte_shift = int(self.__bits / 8)

            raw_int = []
            for i in range(0, len(raw_bytes), byte_shift):
                byte_tmp = raw_bytes[i:i+byte_shift]
                raw_int.append(struct.unpack(fmt, b"\x00" + byte_tmp)[0])

            bits_dequant = self.__bits + 8

            # raise ValueError("Reading 24-bit is not supported!")
        else:
            fmt = dsp.mkDataStructFMT(self.__bits, total_samples)
            raw_int = struct.unpack(fmt, raw_bytes)
            bits_dequant = self.__bits
           

        # Dequantisation with given resolution
        if self.__bits > 8:
            shift = 0
        else:
            shift = int(2 ** self.bits / 2)
        data = np.array([float(int_quant - shift) / pow(2, bits_dequant - 1) for int_quant in raw_int])
        data.shape = (self.__nb_sample, self.__nb_chan)

        return data


"""
# WAVWriter.py -- a wrapper of Python Wave_write object, providing simple interface for students to write data to a WAV file
#   It converts amplitude values into raw data in bytes
# 
# Created: August 18, 2019
# Modified: Januray 25, 2021
"""
class WAVWriter():
    def __init__(self, wavfile, data, fs=44100, bits=16, cmode="scale"):
        # Path of output wav file
        self.filename = wavfile
        # Samples to write out
        self.data = data
        # sampling frequency/rate
        self.fs = fs
        # Quatisation resolution
        self.bits = bits
        # how to handle clipping
        self.cmode = cmode
        if self.cmode != "scale" and self.cmode != "clip":
            raise(ValueError("Unsupported scaling method: '{}'. Use either '{}' or '{}'.".format(self.cmode, "scale", "clip")))


        # Check the dimmension of the input matrix, in order to determine how data is organised.
        # Conventionally, row for sample points and column for channels 
        dim = self.data.shape
        if dim[0] < dim[1]:
            self.__SampleNo =  dim[1]
            self.__ChannelNO = dim[0]
        else:
            self.__SampleNo =  dim[0]
            self.__ChannelNO = dim[1]

    # Write encoded data (in bytes) to wav file
    def write(self):
        # Initialised a Wave_write object
        wav_obj = wave.open(self.filename, "wb")
        wav_obj.setnchannels(self.__ChannelNO)
        wav_obj.setsampwidth(int(self.bits / 8)) # convert bits to bytes
        wav_obj.setframerate(self.fs)
        wav_obj.setnframes(self.__SampleNo)

        wav_obj.writeframes(self.__encode())
        wav_obj.close()

    # Core function for coverting amplitude values to bytes
    def __encode(self):
        total_samples = self.__SampleNo * self.__ChannelNO
        array_data = self.data.copy() # avoid change the dimension of the original data structure - Jan 25, 2021
        if np.max(np.abs(array_data)) >= 1:
            if self.cmode == "scale":
                array_data = dsp.scalesig(array_data)
            elif self.cmode == "clip":
                array_data[array_data >= 1] = 0.99
                array_data[array_data <= -1] = -0.99

            print("WARNING: The peak amplitude exceeds 1. The data has been scaled/clipped before writing to WAV file.")

        array_data.shape = (1, total_samples)

        # Quantisation with given resolution
        ## For 8-bit, unsigned integers are used
        if self.bits > 8:
            shift = 0
        else:
            shift = int(2 ** self.bits / 2)
        raw_int = [int(sp * (pow(2, self.bits-1))-1) + shift for sp in np.nditer(array_data)]

        # Convert integers to bytes
        if self.bits == 24:
            fmt = dsp.mkDataStructFMT(self.bits, 1)
            raw_bytes = struct.pack(fmt, raw_int.pop(0))[:3]
            for n in raw_int:
                raw_bytes += struct.pack("i", n)[:3]
        else:
            fmt = dsp.mkDataStructFMT(self.bits, total_samples)
            raw_bytes = struct.pack(fmt, *raw_int)


        return raw_bytes
