"""

ADC's Project"   _______________     "2021 winter"

"""      
#__________________________________________________________________#

""" modules  """

import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import fft
import sys
    
"""__________________________________________________________________"""

""" Initialization  """

bpsk_6 = {
        "rate": 6,
        "coding rate": 1/2,
        "Nbpsc": 1,
        "Ncbps": 48,
        "Ndbps": 24,
        "RATE": [1, 1, 0, 1],
    }
bpsk_9 = {
        "rate": 9,
        "coding rate": 3/4,
        "Nbpsc": 1,
        "Ncbps": 48,
        "Ndbps": 36,
        "RATE": [1, 1, 1, 1],
    }
qpsk_12 = {
        "rate": 12,
        "coding rate": 1/2,
        "Nbpsc": 2,
        "Ncbps": 96,
        "Ndbps": 48,
        "RATE": [0, 1, 0, 1],
    }
qpsk_18 = {
        "rate": 18,
        "coding rate": 3/4,
        "Nbpsc": 2,
        "Ncbps": 96,
        "Ndbps": 72,
        "RATE": [0, 1, 1, 1],
    }
qam16_24 = {
        "rate": 24,
        "coding rate": 1/2,
        "Nbpsc": 4,
        "Ncbps": 192,
        "Ndbps": 96,
        "RATE": [1, 0, 0, 1],
    }
qam16_36 = {
        "rate": 36,
        "coding rate": 3/4,
        "Nbpsc": 4,
        "Ncbps": 192,
        "Ndbps": 144,
        "RATE": [1, 0, 1, 1],
    }
qam64_48 = {
        "rate": 48,
        "coding rate": 2/3,
        "Nbpsc": 6,
        "Ncbps": 288,
        "Ndbps": 192,
        "RATE": [0, 0, 0, 1],
    }
qam64_54 = {
        "rate": 54,
        "coding rate": 3/4,
        "Nbpsc": 6,
        "Ncbps": 288,
        "Ndbps": 216,
        "RATE": [0, 0, 1, 1],
    }

mod = qam16_24
snr = 60
channel = 'rayleigh'
plot_constellation = 'yes'
fade_vec = [1,0.2,0.2,0.2,0.2]
synch_type = 'defined'
plot_corr_curve = 'on'
use_eq = 'on'
long_th =0.7
short_th = 0.7

tail = np.zeros(6, dtype=int)

service = np.zeros(16, dtype=int)

s_initialization = np.array([1,1,1,1,1,1,1], dtype=int)          ##  Scrambler Initialization

service[0:7] = [1,1,1,1,1,1,1]

nfft = 64          # fft size

cplen = 16

ndcps = 48          # number of data carriers per OFDM symbol

ndspp = 4095          # number of data symbols per packet

ndsppps = 4096          # number of data symbols per packet plus SIGNAL

pl = 4100          # packet length

st = 500

ntspdp = (nfft+cplen)*ndspp          # number of time sample per data packet

ntspdpps = (nfft+cplen)*ndsppps         # number of time sample per data packet plus SIGNAL  

ntspp = (nfft+cplen)*pl          # number of time sample per packet

ndbpp = (mod['Ndbps']*ndspp)         # number of data bit per packet

ndbps = mod['Ndbps']          # number of data bit per symbol

ndbppts = ndbpp - len(tail) - len(service)  ##ndbpp -len(tail) -len(service)



"_____________________________________________________START_________________________________________________ "

"Generating data"
source = np.random.randint(0, high=2, size=15000, dtype=int)         
# source = in_bits

"Calculating number of packet"
nop = math.ceil(len(source)/(ndbppts)) 

"Detected data"
sourcehat = []

"""
Loop :  
choose packet: calculate number of data bits for each packet
make packet: make a time domain packet to send through channel
extract packet : take each packet and try to detect the transmitted bits
"""
for i in range(nop):
    print(f'progress bar: {(i/nop)*100}%')              #showing progress bar 
    
    ch_data = choose_packet(source,nop,i,ndbppts)       
    
    t_signal = make_packet(ch_data, mod, tail, service, sfft=64)
    
    a_ch_signal = channel_type(t_signal,snr,channel)
    
    ex_data = extract_packet(a_ch_signal, mod, sfft=64)
    
    sourcehat = np.append(sourcehat,ex_data) 
   
    


if len(source) != len(sourcehat):
    print("Reciever can't detect transmitted data")
if len(source) == len(sourcehat):
    num_of_error = 0
    for i in range(len(source)):
        if sourcehat[i] != source[i]:
            num_of_error +=1
    print(f'number of error is: {num_of_error}')
    BER = num_of_error/len(sourcehat)
    print(f'the BER is: {BER}')
