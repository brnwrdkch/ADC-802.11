

## Modules


import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import fft

    


## Initialization

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

mod = qpsk_18

tail = np.zeros(6, dtype=int)

service = np.zeros(16, dtype=int)

s_initialization = np.array([0, 0, 0, 0, 0, 0, 0], dtype=int)          ##  Scrambler Initialization

nfft = 64          # fft size

cplen = 16

ndcps = 48          # number of data carriers per OFDM symbol

ndspp = 4095          # number of data symbols per packet

ndsppps = 4096          # number of data symbols per packet plus SIGNAL

pl = 4100          # packet length

st = 400         # silence time

ntspdp = (nfft+cplen)*ndspp          # number of time sample per data packet

ntspdpps = (nfft+cplen)*ndsppps         # number of time sample per data packet plus SIGNAL  

ntspp = (nfft+cplen)*pl          # number of time sample per packet

ndbpp = (mod['Ndbps']*ndspp)         # number of data bit per packet

ndbps = mod['Ndbps']          # number of data bit per symbol

ndbppts = ndbpp - len(tail) - len(service)  ##ndbpp -len(tail) -len(service)



def test_encoder(inputd, rate):     
    L = len(inputd)
    if rate == 1/2:
        codbit = np.zeros(L)
    elif rate == 2/3:
        codbit = np.zeros(int(L/2))
    elif rate == 3/4:
        codbit = np.zeros(int(L/3))
    outputd = np.hstack((inputd,codbit))
    return outputd

def test_decoder(inputd,rate):
    L = len(inputd)
    if rate == 1/2:
        codbit = np.delete(inputd,range(int(L/2),L))
    elif rate == 2/3:
        codbit = np.delete(inputd,range(int((2*L)/3),L))
    elif rate == 3/4:
        codbit = np.delete(inputd,range(int((3*L)/4),L))
    
    return codbit

## interleaver

def interleaver_Core (inp_vec, N_CBPS, N_BPSC):
    s = max(N_BPSC/2,1)

    inner_vec = np.ones(N_CBPS,dtype=np.int8)
    for cnt in range(N_CBPS):
        i = (N_CBPS/16) * (cnt % 16) + math.floor(cnt/16)
        inner_vec[int(i)] = inp_vec[cnt]

    outp_vec = np.ones(N_CBPS,dtype=np.int8)
    for cnt in range(N_CBPS):
        j = (s * math.floor(cnt/s)) + ((cnt + N_CBPS - math.floor(16*cnt/N_CBPS)) % s)
        outp_vec[int(j)] = inner_vec[cnt]

    return outp_vec

def interleaver(inp_vec, N_CBPS, N_BPSC):
    inp_len = len(inp_vec)
    iteration = inp_len/N_CBPS

    outp_vec = np.ones(inp_len,dtype=np.int8)
    for cnt in range(int(iteration)):
        vec = inp_vec[N_CBPS*cnt:N_CBPS*(cnt+1)]
        outp_vec[N_CBPS*cnt:N_CBPS*(cnt+1)] = interleaver_Core(vec, N_CBPS, N_BPSC)

    return outp_vec


# __________________Deinterleaver unit __________________
def deinterleaver_Core (inp_vec, N_CBPS, N_BPSC):
    s = max(N_BPSC/2,1)

    inner_vec = np.zeros(N_CBPS,dtype=np.int8)
    for cnt in range(N_CBPS):
        i = (s * math.floor(cnt/s)) + ((cnt + math.floor(16*cnt/N_CBPS)) % s)
        inner_vec[int(i)] = inp_vec[cnt]

    outp_vec = np.zeros(N_CBPS,dtype=np.int8)
    for cnt in range(N_CBPS):
        k = 16 * cnt - (N_CBPS-1) * math.floor(16*cnt/N_CBPS)
    outp_vec[int(k)] = inner_vec[cnt]

    return outp_vec

def deinterleaver(inp_vec, N_CBPS, N_BPSC):

    inp_len = len(inp_vec)
    iteration = inp_len/N_CBPS

    outp_vec = np.ones(inp_len,dtype=np.int8)
    for cnt in range(int(iteration)):
        vec = inp_vec[N_CBPS*cnt:N_CBPS*(cnt+1)]
        outp_vec[N_CBPS*cnt:N_CBPS*(cnt+1)] = deinterleaver_Core(vec, N_CBPS, N_BPSC)

    return outp_vec

###  interleaver ###

def interleaver(data,Ncbps,Nbpsc):
    data = np.reshape(data,(int(len(data)/Ncbps),Ncbps))
    output = np.array([])
    for j in range(len(data)):
        i_data = np.array([])       # interleaved data
        arranged = np.reshape(data[j],(Nbpsc,48))      # arranged to get interleaved
        for i in range(48): 
            i_data = np.append(i_data,arranged[:,i])
        output = np.append(output,i_data)
    return output


### deinterleaver ###


def deinterleaver(data,Ncbps,Nbpsc):
    data = np.reshape(data,(int(len(data)/Ncbps),Ncbps))
    output = np.array([])
    for j in range(len(data)):
        de_data = np.array([])
        arranged = np.reshape(data[j], (48,Nbpsc))
        for i in range(Nbpsc):
            de_data = np.append(de_data,arranged[:,i])
        output = np.append(output,de_data)
    return output

    



## modulator

def bpsk(Data):
    import numpy as np
    import matplotlib.pyplot as plt
    import math
    y=list(Data)
    modulation=1
    g=len(y)/modulation
    h=list(np.zeros(int(g)))
    for k in  range(int(g)):
        if y[k]==1:
            h[k]=1
        elif y[k]==0:
            h[k]=-1
     
    
    return h


def QPSK(Data):
    import numpy as np
    import matplotlib.pyplot as plt
    import math
    mapping = {
              (0,0) : -1-1j,
              (0,1) : -1+1j,
              (1,0) :  1-1j,
              (1,1) :  1+1j,
              }
    y=list(Data)
    modulation=2
    g=len(y)/modulation
    h=list(np.zeros(int(g)))
    t=0
    for k in  range(int(g)):
        s =y[int(t):int(t+2)]
        tpl = tuple(s)#  Original Data to tuple
        h[k]=mapping.get(tpl)
        t=t+2
     
    
    return h

        


def QAM16(Data):
      
    import numpy as np
    import matplotlib.pyplot as plt
    import math
    mapping = {
           (0,0,0,0) : -3-3j,
           (0,0,0,1) : -3-1j,
           (0,0,1,0) : -3+3j,
           (0,0,1,1) : -3+1j,
           (0,1,0,0) : -1-3j,
           (0,1,0,1) : -1-1j,
           (0,1,1,0) : -1+3j,
           (0,1,1,1) : -1+1j,
           (1,0,0,0) :  3-3j,
           (1,0,0,1) :  3-1j,
           (1,0,1,0) :  3+3j,
           (1,0,1,1) :  3+1j,
           (1,1,0,0) :  1-3j,
           (1,1,0,1) :  1-1j,
           (1,1,1,0) :  1+3j,
           (1,1,1,1) :  1+1j
           }
    y=list(Data)
    modulation=4
    g=len(y)/modulation
    h=list(np.zeros(int(g)))
    t=0
    for k in  range(int(g)):
        s =y[int(t):int(t+4)]
        tpl = tuple(s)#  Original Data to tuple
        h[k]=mapping.get(tpl)
        t=t+4

    return h


def QAM64(Data):
    import numpy as np
    import matplotlib.pyplot as plt
    import math
    mapping = {
             (1,0,0,1,0,0) :  7+7j,
             (1,0,0,1,0,1) :  7+5j,
             (1,0,0,1,1,1) :  7+3j,
             (1,0,0,1,1,0) :  7+1j,
             (1,0,0,0,1,0) :  7-1j,
             (1,0,0,0,1,1) :  7-3j,
             (1,0,0,0,0,1) :  7-5j,
             (1,0,0,0,0,0) :  7-7j,
             (1,0,1,1,0,0) :  5+7j,
             (1,0,1,1,0,1) :  5+5j,
             (1,0,1,1,1,1) :  5+3j,
             (1,0,1,1,1,0) :  5+1j,
             (1,0,1,0,1,0) :  5-1j,
             (1,0,1,0,1,1) :  5-3j,
             (1,0,1,0,0,1) :  5-5j,
             (1,0,1,0,0,0) :  5-7j,
             (1,1,1,1,0,0) :  3+7j,
             (1,1,1,1,0,1) :  3+5j,
             (1,1,1,1,1,1) :  3+3j,
             (1,1,1,1,1,0) :  3+1j,
             (1,1,1,0,1,0) :  3-1j,
             (1,1,1,0,1,1) :  3-3j,
             (1,1,1,0,0,1) :  3-5j,
             (1,1,1,0,0,0) :  3-7j,
             (1,1,0,1,0,0) :  1+7j,
             (1,1,0.1,0,1) :  1+5j,
             (1,1,0,1,1,1) :  1+3j,
             (1,1,0,1,1,0) :  1+1j,
             (1,1,0,0,1,0) :  1-1j,
             (1,1,0,0,1,1) :  1-3j,
             (1,1,0,0,0,1) :  1-5j,
             (1,1,0,0,0,0) :  1-7j,
             (0,1,0,1,0,0) :  -1+7j,
             (0,1,0,1,0,1) :  -1+5j,
             (0,1,0,1,1,1) :  -1+3j,
             (0,1,0,1,1,0) :  -1+1j,
             (0,1,0,0,1,0) :  -1-1j,
             (0,1,0,0,1,1) :  -1-3j,
             (0,1,0,0,0,1) :  -1-5j,
             (0,1,0,0,0,0) :  -1-7j,
             (0,1,1,1,0,0) :  -3+7j,
             (0,1,1,1,0,1) :  -3+5j,
             (0,1,1,1,1,1) :  -3+3j,
             (0,1,1,1,1,0) :  -3+1j,
             (0,1,1,0,1,0) :  -3-1j,
             (0,1,1,0,1,1) :  -3-3j,
             (0,1,1,0,0,1) :  -3-5j,
             (0,1,1,0,0,0) :  -3-7j,
             (0,0,1,1,0,0) :  -5+7j,
             (0,0,1,1,0,1) :  -5+5j,
             (0,0,1,1,1,1) :  -5+3j,
             (0,0,1,1,1,0) :  -5+1j,
             (0,0,1,0,1,0) :  -5-1j,
             (0,0,1,0,1,1) :  -5-3j,
             (0,0,1,0,0,1) :  -5-5j,
             (0,0,1,0,0,0) :  -5-7j,
             (0,0,0,1,0,0) :  -7+7j,
             (0,0,0,1,0,1) :  -7+5j,
             (0,0,0,1,1,1) :  -7+3j,
             (0,0,0,1,1,0) :  -7+1j,
             (0,0,0,0,1,0) :  -7-1j,
             (0,0,0,0,1,1) :  -7-3j,
             (0,0,0,0,0,1) :  -7-5j,
             (0,0,0,0,0,0) :  -7-7j,
             }
    y=list(Data)
    modulation=6
    g=len(y)/modulation
    h=list(np.zeros(int(g)))
    t=0
    for k in  range(int(g)):
        s =y[int(t):int(t+6)]
        tpl = tuple(s)#  Original Data to tuple
        h[k]=mapping.get(tpl)
        t=t+6

    return h


def dembpsk(Data):
    import numpy as np
    import matplotlib.pyplot as plt
    import math
    y=list(Data)
    modulation=1
    g=len(y)/modulation
    h=list(np.zeros(int(g)))
    for k in  range(int(g)):
        if y[k]>=0:
            h[k]=1
        elif y[k]< 0:
            h[k]=0
        
    return h


def demQPSK(Data):
    import numpy as np
    import matplotlib.pyplot as plt
    import math
    b=[value.real for value in Data]
    
    c=[value.imag for value in Data]
    
    modulation=2
    g=len(b)*modulation
    h=list(np.zeros(int(g)))
    
    t=0
    for k in  range(len(b)):
      
      
      if b[k]>=0 and c[k]>=0:  # Decision region 1
       h[t]=1
       h[t+1]=1
      elif b[k]<=0 and c[k] > 0:# Decision region 2
       h[t]=0 
       h[t+1]=1
      elif b[k]<0 and c[k]<=0: # Decision region 3
       h[t]=0
       h[t+1]=0
      elif b[k]>=0 and c[k]<0: # Decision region 4
       h[t]=1
       h[t+1]=0
      t=t+2 
    return (h)



def demQAM16(Data):
    import numpy as np
    import matplotlib.pyplot as plt
    import math
    mapping = {
              (0,0,0,0) : -3-3j,
              (0,0,0,1) : -3-1j,
              (0,0,1,0) : -3+3j,
              (0,0,1,1) : -3+1j,
              (0,1,0,0) : -1-3j,
              (0,1,0,1) : -1-1j,
              (0,1,1,0) : -1+3j,
              (0,1,1,1) : -1+1j,
              (1,0,0,0) :  3-3j,
              (1,0,0,1) :  3-1j,
              (1,0,1,0) :  3+3j,
              (1,0,1,1) :  3+1j,
              (1,1,0,0) :  1-3j,
              (1,1,0,1) :  1-1j,
              (1,1,1,0) :  1+3j,
              (1,1,1,1) :  1+1j
              }
    ###############################
    ss=len(Data)
   ## h=[]# Zero vector
    ##for t in range(4*ss):
        ## h.append(0)
    #################################     
    modulation=4
    values = mapping.values()# extract values
    values_list = list(values)# tuple to list
    #################################
    x = mapping.keys()# extract key
    y1=list(x)
    
    #################################
    h=[]
    po=list(np.zeros(16))
    for k in range(ss):
        
        for i in range(16):
              po[i]=abs(values_list[i] - Data[k])
              
     

        min_value = min(po)
        min_index = po.index(min_value)
        
        h.extend(y1[min_index])
    return h



def demQAM64(Data):
    import numpy as np
    import matplotlib.pyplot as plt
    import math
    mapping = {
          (1,0,0,1,0,0) :  7+7j,
          (1,0,0,1,0,1) :  7+5j,
          (1,0,0,1,1,1) :  7+3j,
          (1,0,0,1,1,0) :  7+1j,
          (1,0,0,0,1,0) :  7-1j,
          (1,0,0,0,1,1) :  7-3j,  
          (1,0,0,0,0,1) :  7-5j,
          (1,0,0,0,0,0) :  7-7j,
          (1,0,1,1,0,0) :  5+7j,
          (1,0,1,1,0,1) :  5+5j,
          (1,0,1,1,1,1) :  5+3j,    
          (1,0,1,1,1,0) :  5+1j,
          (1,0,1,0,1,0) :  5-1j,
          (1,0,1,0,1,1) :  5-3j,
          (1,0,1,0,0,1) :  5-5j,
          (1,0,1,0,0,0) :  5-7j,
          (1,1,1,1,0,0) :  3+7j,
          (1,1,1,1,0,1) :  3+5j,
          (1,1,1,1,1,1) :  3+3j,
          (1,1,1,1,1,0) :  3+1j,
          (1,1,1,0,1,0) :  3-1j,
          (1,1,1,0,1,1) :  3-3j,
          (1,1,1,0,0,1) :  3-5j,
          (1,1,1,0,0,0) :  3-7j,
          (1,1,0,1,0,0) :  1+7j,
          (1,1,0.1,0,1) :  1+5j,
          (1,1,0,1,1,1) :  1+3j,
          (1,1,0,1,1,0) :  1+1j,
          (1,1,0,0,1,0) :  1-1j,
          (1,1,0,0,1,1) :  1-3j,
          (1,1,0,0,0,1) :  1-5j,
          (1,1,0,0,0,0) :  1-7j,
          (0,1,0,1,0,0) :  -1+7j,
          (0,1,0,1,0,1) :  -1+5j,
          (0,1,0,1,1,1) :  -1+3j,
          (0,1,0,1,1,0) :  -1+1j,
          (0,1,0,0,1,0) :  -1-1j,
          (0,1,0,0,1,1) :  -1-3j,
          (0,1,0,0,0,1) :  -1-5j,
          (0,1,0,0,0,0) :  -1-7j,
          (0,1,1,1,0,0) :  -3+7j,
          (0,1,1,1,0,1) :  -3+5j,
          (0,1,1,1,1,1) :  -3+3j,
          (0,1,1,1,1,0) :  -3+1j,
          (0,1,1,0,1,0) :  -3-1j,
          (0,1,1,0,1,1) :  -3-3j,
          (0,1,1,0,0,1) :  -3-5j,
          (0,1,1,0,0,0) :  -3-7j,
          (0,0,1,1,0,0) :  -5+7j,
          (0,0,1,1,0,1) :  -5+5j,
          (0,0,1,1,1,1) :  -5+3j,
          (0,0,1,1,1,0) :  -5+1j,
          (0,0,1,0,1,0) :  -5-1j,
          (0,0,1,0,1,1) :  -5-3j,
          (0,0,1,0,0,1) :  -5-5j,
          (0,0,1,0,0,0) :  -5-7j,
          (0,0,0,1,0,0) :  -7+7j,
          (0,0,0,1,0,1) :  -7+5j,
          (0,0,0,1,1,1) :  -7+3j,
          (0,0,0,1,1,0) :  -7+1j,
          (0,0,0,0,1,0) :  -7-1j,
          (0,0,0,0,1,1) :  -7-3j,
          (0,0,0,0,0,1) :  -7-5j,
          (0,0,0,0,0,0) :  -7-7j,
          }
    ###############################
    ss=len(Data)
   ### h=[]# Zero vector
   ### for t in range(4*ss):
     ###   h.append(0)
    #################################
    modulation=6
    values = mapping.values()# extract values
    values_list = list(values)# tuple to list
    #
    x = mapping.keys()# extract key
    y1=list(x)
    #
    h=[]
    po=list(np.zeros(64))
    for k in range(ss):
        for i in range(64):
            po[i]=abs(values_list[i] - Data[k])
            
        min_value = min(po)
        min_index = po.index(min_value)
          
        h.extend(y1[min_index])
    return h


def modulation(in_data, modulationtype):         #interleaved data
    if modulationtype == bpsk_6 or modulationtype == bpsk_9:
        modulated_data = bpsk(in_data)
    elif modulationtype == qpsk_12 or modulationtype == qpsk_18:
        modulated_data = QPSK(in_data)
    elif modulationtype == qam16_24 or modulationtype == qam16_36:
        modulated_data = QAM16(in_data)
    elif modulationtype == qam64_48 or modulationtype == qam64_54:
        modulated_data = QAM64(in_data)
    return modulated_data


def demodulation(mo_data, modulationtype):         #modulated data
    if modulationtype == bpsk_6 or modulationtype == bpsk_9:
        demodulated_data = dembpsk(mo_data)
    elif modulationtype == qpsk_12 or modulationtype == qpsk_18:
        demodulated_data = demQPSK(mo_data)
    elif modulationtype == qam16_24 or modulationtype == qam16_36:
        demodulated_data = demQAM16(mo_data)
    elif modulationtype == qam64_48 or modulationtype == qam64_54:
        demodulated_data = demQAM64(mo_data)
    return demodulated_data

def choose_packet(data,nop,iteration,ndbpp):
    complet_packet = data[i*ndbpp:(i+1)*ndbpp]
    last_paacket = data[(nop-1)*ndbpp:]
    if iteration<nop-1:
        output = complet_packet
    elif iteration==nop-1:
        output = last_paacket
    return output



def add_pad(data, mod):   ##packet binary data with service and tail
    p_data = data
    pad_len = 0      #pad length
    if len(data)%mod["Ndbps"] != 0:
        pad = np.zeros(mod["Ndbps"]-(len(data)%mod["Ndbps"]))
        p_data = np.hstack((data,pad))   # data with pad
        pad_len = len(pad)
    return p_data,pad_len


def delet_pad(p_data, pad_len):   ## data with pad
    data = np.delete(p_data,range(len(p_data)-pad_len,len(p_data)))
    return data

def make_packet(chosen_packet, mod_type, tail, service):
    sdatat = np.hstack((service, chosen_packet, tail))     ## [service chosen packet tail]
    p_data,plen = add_pad(sdatat,mod_type)
    en_data = test_encoder(p_data, mod_type["coding rate"])   ##encoded data
    mo_data = modulation(en_data,mod_type)      ##modulated data
    return mo_data,plen

def extract_packet(time_signal, mod_type, len_pad):
    demo_data = demodulation(time_signal, mod_type)
    dec_data = test_decoder(demo_data, mod_type["coding rate"]) 
    sdatat = delet_pad(dec_data, len_pad)
    data = sdatat[len(service):len(sdatat)-len(tail)]

    
    
    return data


source = np.random.randint(0, high=2, size=1000000)
# source = np.hstack((source,tail))   #added tail source
# pad = np.zeros((len(source)-((nop-1)*ndbppts))%ndbps, dtype=int)
# source = np.hstack((tsource,pad))
# print(len(psource))

# psourcehat = []
nop = math.ceil(len(source)/(ndbppts))          # number of packets


sourcehat = []
for i in range(nop):
    ch_data = choose_packet(source,nop,i,ndbppts)       ## chosen packet   (bit)
    t_signal,plen = make_packet(ch_data, mod, tail, service)    ## plen :pad length
    ex_data = extract_packet(t_signal, mod, plen)
    sourcehat = np.append(sourcehat,ex_data) 
    print(len(ch_data))
    print(len(t_signal))
    print(len(ex_data))
    print(len(sourcehat))
    print(plen)


for i in range(len(sourcehat)):
    if sourcehat[i] != source[i]:
        print('false')
print(len(sourcehat))
source[0:20]

sourcehat[0:20]

t_signal[0:20]

plt.plot(t_signal[0:100])

