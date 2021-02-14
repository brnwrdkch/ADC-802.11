"""

ADC's Project"   _______________     "2021 winter"

"""   

"_____________________________________________________START_________________________________________________ "

"Generating data"
source = np.random.randint(0, high=2, size=10000, dtype=int)         

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
    ch_data = choose_packet(source,nop,i,ndbppts)       
    t_signal,plen = make_packet(ch_data, mod, tail, service, sfft=64)   
    a_ch_signal = awgn(t_signal,snr)
    ex_data = extract_packet(a_ch_signal, mod, plen, sfft=64)
    sourcehat = np.append(sourcehat,ex_data) 


"""test"""
for i in range(5):
    print(i)
    
    
    
    
    
    
"""

ADC's Project"   _______________     "2021 winter"

"""      
#__________________________________________________________________#

""" modules  """

import numpy as np
import math
import matplotlib.pyplot as plt
from numpy import fft

    
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

mod = bpsk_6
snr = 22
channel = 'awgn'
plot_constellation = 'yes'
fade_vec = [1,0.7,0.5,0.2,0.1,0.1,0.2]
synch_type = 'corr'

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



ntspdp = (nfft+cplen)*ndspp          # number of time sample per data packet

ntspdpps = (nfft+cplen)*ndsppps         # number of time sample per data packet plus SIGNAL  

ntspp = (nfft+cplen)*pl          # number of time sample per packet

ndbpp = (mod['Ndbps']*ndspp)         # number of data bit per packet

ndbps = mod['Ndbps']          # number of data bit per symbol

ndbppts = ndbpp - len(tail) - len(service)  ##ndbpp -len(tail) -len(service)


###scrambler

def scrambler_core(inp_vec, Initial_Vec):
  vec_len = len(inp_vec)
  out_vec = np.zeros(vec_len,dtype=np.int8)
 
  scrambler_vec = Initial_Vec; # initializing the scrambler
 
  for i in range(vec_len):
    temp = np.bitwise_xor(int(scrambler_vec[3]) , int(scrambler_vec[6]))
    scrambler_vec[1:] = scrambler_vec[0:6]
    scrambler_vec[0] = temp
    out_vec[i] = np.bitwise_xor(int(scrambler_vec[0]) , int(inp_vec[i]))
 
  return out_vec
 
# __________________Scrambler unit __________________
 
def scrambler (inp_vec, service_len, tail_len):
  vec_len = len(inp_vec)
  out_vec = np.zeros(vec_len,dtype=np.int8)
  
  
  Initial_Vec = inp_vec[0:7]
  out_vec[0:service_len] = inp_vec[0:service_len]
  out_vec[vec_len-tail_len:] = inp_vec[vec_len-tail_len:]

  out_vec[service_len:vec_len-tail_len] = scrambler_core(inp_vec[service_len:vec_len-tail_len],Initial_Vec)
 
  return out_vec
 
# __________________Decrambler unit __________________
 
def descrambler (inp_vec, service_len):
  descrambler_init = inp_vec[0:7]
 
  vec_len = len(inp_vec)
  out_vec = np.zeros(vec_len,dtype=np.int8)
  out_vec[0:service_len] = inp_vec[0:service_len]
  out_vec[service_len:] = scrambler_core(inp_vec[service_len:],descrambler_init)
  return out_vec
 
 


## encoder for test
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


## interleaver abl
def interleaver_Core(inp_vec, N_CBPS, N_BPSC):
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

def interleaver_a(inp_vec, N_CBPS, N_BPSC):
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

def deinterleaver_a(inp_vec, N_CBPS, N_BPSC):

  inp_len = len(inp_vec)
  iteration = inp_len/N_CBPS

  outp_vec = np.ones(inp_len,dtype=np.int8)
  for cnt in range(int(iteration)):
    vec = inp_vec[N_CBPS*cnt:N_CBPS*(cnt+1)]
    outp_vec[N_CBPS*cnt:N_CBPS*(cnt+1)] = deinterleaver_Core(vec, N_CBPS, N_BPSC)

  return outp_vec



"""
###  interleaver ###

def interleaver_a(data,Ncbps,Nbpsc):
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


def deinterleaver_a(data,Ncbps,Nbpsc):
    data = np.reshape(data,(int(len(data)/Ncbps),Ncbps))
    output = np.array([])
    for j in range(len(data)):
        de_data = np.array([])
        arranged = np.reshape(data[j], (48,Nbpsc))
        for i in range(Nbpsc):
            de_data = np.append(de_data,arranged[:,i])
        output = np.append(output,de_data)
    return output

    

"""

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
        h[k]=(mapping.get(tpl))*(1/math.sqrt(2))
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
        h[k]=mapping.get(tpl)*(1/math.sqrt(10))
        t=t+4

    return h


# 64 QAM Modulator by Hamed Ghanbari
def QAM64(Data):
       import numpy as np
       import matplotlib.pyplot as plt
       import math
       mapping = {
                 (1,0,0,1,0,0) :  +7+7j,
                 (1,0,0,1,0,1) :  +7+5j,
                 (1,0,0,1,1,1) :  +7+3j,
                 (1,0,0,1,1,0) :  +7+1j,
                 (1,0,0,0,1,0) :  +7-1j,
                 (1,0,0,0,1,1) :  +7-3j,
                 (1,0,0,0,0,1) :  +7-5j,
                 (1,0,0,0,0,0) :  +7-7j,
                 (1,0,1,1,0,0) :  +5+7j,
                 (1,0,1,1,0,1) :  +5+5j,
                 (1,0,1,1,1,1) :  +5+3j,
                 (1,0,1,1,1,0) :  +5+1j,
                 (1,0,1,0,1,0) :  +5-1j,
                 (1,0,1,0,1,1) :  +5-3j,
                 (1,0,1,0,0,1) :  +5-5j,
                 (1,0,1,0,0,0) :  +5-7j,
                 (1,1,1,1,0,0) :  +3+7j,
                 (1,1,1,1,0,1) :  +3+5j,
                 (1,1,1,1,1,1) :  +3+3j,
                 (1,1,1,1,1,0) :  +3+1j,
                 (1,1,1,0,1,0) :  +3-1j,
                 (1,1,1,0,1,1) :  +3-3j,
                 (1,1,1,0,0,1) :  +3-5j,
                 (1,1,1,0,0,0) :  +3-7j,
                 (1,1,0,1,0,0) :  +1+7j,
                 (1,1,0,1,0,1) :  +1+5j,
                 (1,1,0,1,1,1) :  +1+3j,
                 (1,1,0,1,1,0) :  +1+1j,
                 (1,1,0,0,1,0) :  +1-1j,
                 (1,1,0,0,1,1) :  +1-3j,
                 (1,1,0,0,0,1) :  +1-5j,
                 (1,1,0,0,0,0) :  +1-7j,
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
         h[k]=mapping.get(tpl)*(1/math.sqrt(42))
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
        
    return h,Data


def demQPSK(Data):
    import numpy as np
    import matplotlib.pyplot as plt
    import math
    # b=[(value.real) for value in Data]
    
    # c=[(value.imag) for value in Data]
    b=list(np.zeros(len(Data)))
    c=list(np.zeros(len(Data)))
    z=np.zeros(len(Data),dtype=complex)
    for i in range(len(Data)):
        b[i] = (Data[i].real)*math.sqrt(2)
        c[i] = (Data[i].imag)*math.sqrt(2) 
        z[i] = b[i]+c[i]*1j 
    plt.plot([x.real for x in z], [x.imag for x in z], 'b.')
    plt.axis('equal')
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
    return (h),z



# fast 16 QAM Demodulator by Hamed Ghanbari
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
    factor = np.full(shape=len(Data),fill_value=math.sqrt(10),dtype=complex)
    Data = np.prod([factor,Data],axis=0)
    # plt.plot([x.real for x in Data], [x.imag for x in Data], 'b.')
    # plt.axis('equal')
    Data1 =np.array(Data)
    
   ## h=[]# Zero vector
    ##for t in range(4*ss):
        ## h.append(0)
    #################################     
    modulation=4
    values = mapping.values()# extract values
    values_list =np.array( list(values))# tuple to list
    #################################
    x = mapping.keys()# extract key
    y1=list(x)
    
    #################################
    h=[]
    po=np.zeros(16)
    for k in range(ss):
        po=np.absolute(values_list - Data1[k])
        min_value = np.argmin(po)
        h.extend(y1[min_value])
   
    return h,Data



#64 QAM Demodulator by Hamed Ghanbari
def demQAM64(Data):
    import numpy as np
    import matplotlib.pyplot as plt
    import math
    mapping = {
              (1,0,0,1,0,0) :  +7+7j,
              (1,0,0,1,0,1) :  +7+5j,
              (1,0,0,1,1,1) :  +7+3j,
              (1,0,0,1,1,0) :  +7+1j,
              (1,0,0,0,1,0) :  +7-1j,
              (1,0,0,0,1,1) :  +7-3j,
              (1,0,0,0,0,1) :  +7-5j,
              (1,0,0,0,0,0) :  +7-7j,
              (1,0,1,1,0,0) :  +5+7j,
              (1,0,1,1,0,1) :  +5+5j,
              (1,0,1,1,1,1) :  +5+3j,
              (1,0,1,1,1,0) :  +5+1j,
              (1,0,1,0,1,0) :  +5-1j,
              (1,0,1,0,1,1) :  +5-3j,
              (1,0,1,0,0,1) :  +5-5j,
              (1,0,1,0,0,0) :  +5-7j,
              (1,1,1,1,0,0) :  +3+7j,
              (1,1,1,1,0,1) :  +3+5j,
              (1,1,1,1,1,1) :  +3+3j,
              (1,1,1,1,1,0) :  +3+1j,
              (1,1,1,0,1,0) :  +3-1j,
              (1,1,1,0,1,1) :  +3-3j,
              (1,1,1,0,0,1) :  +3-5j,
              (1,1,1,0,0,0) :  +3-7j,
              (1,1,0,1,0,0) :  +1+7j,
              (1,1,0,1,0,1) :  +1+5j,
              (1,1,0,1,1,1) :  +1+3j,
              (1,1,0,1,1,0) :  +1+1j,
              (1,1,0,0,1,0) :  +1-1j,
              (1,1,0,0,1,1) :  +1-3j,
              (1,1,0,0,0,1) :  +1-5j,
              (1,1,0,0,0,0) :  +1-7j,
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
    factor = np.full(shape=len(Data),fill_value=math.sqrt(42),dtype=complex)
    Data = np.prod([factor,Data],axis=0)
    # plt.plot([x.real for x in Data], [x.imag for x in Data], 'b.')
    # plt.axis('equal')
    Data1 =np.array(Data)
   ### h=[]# Zero vector
   ### for t in range(4*ss):
     ###   h.append(0)
    #################################
    modulation=6
    values = (mapping.values())# extract values
    values_list =np.array( list(values))# tuple to list
    #
    x = mapping.keys()# extract key
    y1=list(x)
    #
    h=[]
    po=np.zeros(64)
    for k in range(ss):
          po= np.absolute(values_list-Data1[k]) 
          min_value = np.argmin(po)
          h.extend((y1[min_value]))
    return h,Data



####"subcarrier_allocation extract_data"

def subcarrier_allocation(mdata):         # mdata : modulated data 
    d_index = np.array([1, 2, 3, 4, 5, 6, 8, 9, 10,           # data index
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            22, 23, 24, 25, 26, 38, 39, 40,
            41, 42, 44, 45, 46, 47, 48, 49, 50,
            51, 52, 53, 54, 55, 56, 58, 59, 60,
            61, 62, 63], dtype=int)
    allocated_subs = np.zeros(64, dtype=complex)
    for i in range(len(d_index)):
        allocated_subs[d_index[i]] = mdata[i]
    return allocated_subs

#### "extract_data"
def extract_data(mdata):         # mdata : modulated data
    d_index = np.array([1, 2, 3, 4, 5, 6, 8, 9, 10,           # data index
             11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
             22, 23, 24, 25, 26, 38, 39, 40,
             41, 42, 44, 45, 46, 47, 48, 49, 50,
             51, 52, 53, 54, 55, 56, 58, 59, 60,
             61, 62, 63], dtype=int)
    extracted_data = np.zeros(48, dtype=complex)
    for i in range(len(d_index)):
        extracted_data[i] = mdata[d_index[i]]
    return extracted_data

    

###### "ifft_pilot"

def ifft_pilot(i_data, nfft):          ## i_data = input data (freq. domain signal)
    p_index = np.array([7, 21, 43, 57], dtype=int)          ## pilot index
    p_value = np.array([1, 1, 1, -1], dtype=complex)          ## pilot value
    for i in range(len(p_index)):
        i_data[p_index[i]] = p_value[i]
    o_data = np.fft.ifft(i_data, n=nfft)          ## output data
    
    return o_data          ## returns time domain signal 
    
    

###### "fft block"


def fft_pilot(i_data,nfft):          ## i_data = input data (time domain signal)
    fd_signal = np.fft.fft(i_data, nfft)          #freq. domain data
    p_index = np.array([7, 21, 43, 57], dtype=int)          ## pilot index
    p_value = np.zeros(4, dtype=complex)
    for i in range(len(p_index)):
        p_value[i] = fd_signal[p_index[i]]
    for i in range(len(p_index)):
        fd_signal[p_index[i]] = 0
    return fd_signal, p_value          ##  freq. domain signal , p_value for eq.

########################################### Add Cyclic Perfix ###############################################

def add_CP(OFDM_time):
    CP = 16                                                            #size of cyclic perfix
    cp = OFDM_time[-CP:]
    return np.hstack([cp,OFDM_time])

######################################### Remove Cyclic Perfix ###############################################

def remove_CP(OFDM_RX):
    CP = 16                                                           #size of cyclic perfix
    k = 64
    return OFDM_RX[CP:(CP+k)]

def IFFT(signal):
    return np.fft.ifft(signal)



############################################# Long_preamble ##################################################

long = np.array([0,1,-1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1])
long_out = IFFT(long)
long_withCP = add_CP(long_out)
preamble_long = np.hstack([long_withCP,long_withCP])

############################################ Short preamble ###################################################

x = math.sqrt(13.6)
short = np.array([0,0,0,0,-1-1j,0,0,0,-1-1j,0,0,0,1+1j,0,0,0,1+1j,0,0,0,1+1j,0,0,0,1+1j,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1+1j,0,0,0,-1-1j,0,0,0,1+1j,0,0,0,-1-1j,0,0,0,-1-1j,0,0,0,1+1j,0,0,0])
short_in = short*x
short_out = IFFT(short_in)
short_withCP = add_CP(short_out)
preamble_short = np.hstack([short_withCP,short_withCP])

################################################################################################################

preamble = np.hstack([preamble_short,preamble_long])


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


def demodulation(mo_data, modulationtype, datatype):         #modulated data
    if modulationtype == bpsk_6 or modulationtype == bpsk_9:
        demodulated_data,datafp = dembpsk(mo_data)
    elif modulationtype == qpsk_12 or modulationtype == qpsk_18:
        demodulated_data,datafp = demQPSK(mo_data)
    elif modulationtype == qam16_24 or modulationtype == qam16_36:
        demodulated_data,datafp = demQAM16(mo_data)
    elif modulationtype == qam64_48 or modulationtype == qam64_54:
        demodulated_data,datafp = demQAM64(mo_data)
    if plot_constellation == 'yes' and datatype == 'data':
        plt.plot([x.real for x in datafp], [x.imag for x in datafp], 'b.')
        plt.axis('equal')
    
    elif plot_constellation == 'no':
        pass
    return demodulated_data

# SIGNAL generation by H.Ghanbari
def SIGNAL(o,b):
     import numpy as np
     import matplotlib.pyplot as plt
     import math
     num=24# Number of bits
    
     t=0
     d=0
     q=[]# Zero vector
     for t in range(2*num):
         q.append(0)
         t += 1
     x=[]# Zero vector
     for d in range(num):
       x.append(0)
       d += 1
     f=x
     y=p=q # Dummy variables 
     # Data rate
    
     x[0]=o[0]# First bit of Data rate 
     x[1]=o[1]# Secound bit of Data rate 
     x[2]=o[2]# Thired bit of Data rate 
     x[3]=o[3]# Fourth bit of Data rate
     for i in range(len(b)):
       x[16-i] = b[-i+len(b)-1]
     for i in range(12-len(b)):
       x[i+5] = 0
    #  x[5]=b[0]
    #  x[6]=b[1]
    #  x[7]=b[2]
    #  x[8]=b[3]
    #  x[9]=b[4]
    #  x[10]=b[5]
    #  x[11]=b[6]
    #  x[12]=b[7]
    #  x[13]=b[8]
    #  x[14]=b[9]
    #  x[15]=b[10]
    #  x[16]=b[11]
    #  # caution
     fh1=[0,0,0,0,0,0]
     fh1.extend(x)
    # print(len(fh1))
     i=0
     # Convolutional encoder 1/2 
    #  for a in range(num):
    #     y[i]=f[a]^f[a-2]^f[a-3]^f[a-5]^f[a-6]
    #     y[i+1]=f[a]^f[a-1]^f[a-2]^f[a-3]^f[a-6]
    #     i=i+2
    # # print(y)
     return f

## this function makes signal
def make_signal(mod_rate, in_data, Ncbps):     ## in_data:interleaved data
    noos = int(len(in_data)/Ncbps)      # number if ofdm symbol per packet
    
    bi_noos = bin(noos)      #binary noos
    
    bi_noos = bi_noos.split('b')
    bi_noos = [int(x) for x in bi_noos[1]]
    bi_noos = np.array(bi_noos)
    
    signal = SIGNAL(mod_rate, bi_noos)     # raw siganl
    
    en_signal = test_encoder(signal,1/2)     # encoded signal
    
    int_signal = interleaver_a(en_signal, 48, 1)    # interleaved signal mod:bpsk_6
    
    mo_signal = modulation(int_signal, bpsk_6)
    
    return mo_signal

inpic = 'a.png'
outpic = 'square_out.png'
in_bytes = np.fromfile(inpic, dtype = "uint8")
in_bits = np.unpackbits(in_bytes)
len(in_bits)

# by H.Ghanbari
def awgn(Data,SNR):
   SNR = 10**(SNR/10)
   len_data = len(Data)
   real_noise = np.random.normal(0,math.sqrt(0.5),len(Data))
   imag_noise = np.random.normal(0,math.sqrt(0.5),len(Data))
   factor = np.full(shape=len(Data),fill_value=1j,dtype=complex)
   imag_noise = np.prod([imag_noise,factor],axis=0)
   noise = np.sum([real_noise,imag_noise],axis=0)
   factor = np.full(shape=len(Data),fill_value=1/math.sqrt(SNR),dtype=complex)
   noise = np.prod([noise,factor],axis=0)
   noisy_data = np.sum([Data,noise],axis=0)
   return noisy_data

def simple_fading(fading,Data):
    # Data is the output of IFFT in TX
    # fading=[1 ,.025, .075]
    fading=np.array(fading)# Impulse response of input fading
    h1=np.convolve(fading, Data)# Convolutional method
    n=len(Data)
    h=h1[0:n]
    return h

def channel_type(Data,snr,chtype):
    if chtype == 'awgn':
        outdata = awgn(Data, snr)
    if chtype == 'rayleigh':
        outdata = simple_fading(fade_vec, Data)
        outdata = awgn(outdata,snr)
    return outdata



## extract data




def extract_signal(data):
    signal = data[0:80]
    rcp_signal = remove_CP(signal)
    fft_signal,h = fft_pilot(rcp_signal,64)
    deallo_signal = extract_data(fft_signal)
    de_signal = demodulation(deallo_signal, bpsk_6, 'signal')
    deint_signal = deinterleaver_a(de_signal, 48, 1)
    dec_signal = test_decoder(deint_signal, 1/2)
    p_len = dec_signal[5:17]
    mod = dec_signal[0:4]
    packet_len_time = int("".join(str(i) for i in p_len),2)
    return packet_len_time,mod

## find modoulation type from RATE bit


def find_mod(mod_rate):
    mod_rate =np.split(mod_rate, 4)
    if mod_rate == bpsk_6["RATE"]:
        mod_type = bpsk_6
    elif mod_rate == bpsk_9["RATE"]:
        mod_type = bpsk_9
    elif mod_rate == qpsk_12["RATE"]:
        mod_type = qpsk_12
    elif mod_rate == qpsk_18["RATE"]:
        mod_type = qpsk_18
    elif mod_rate == qam16_24["RATE"]:
        mod_type = qam16_24
    elif mod_rate == qam16_36["RATE"]:
        mod_type = qam16_36
    elif mod_rate == qam64_48["RATE"]:
        mod_type = qam64_48
    elif mod_rate == qam64_54["RATE"]:
        mod_type = qam64_54
    else:
        mod_type = None
        print("Due to extrem noise ,the reciver couldn't find out what is the coding rate")
        exit()
    return mod_type

def choose_packet(data,nop,iteration,ndbpp):
    
    "packet with 4095 OFDM symbol"
    complet_packet = data[i*ndbpp:(i+1)*ndbpp]

    "packet with less than 4095 OFDM symbol"
    last_paacket = data[(nop-1)*ndbpp:]
    
    if iteration<nop-1:
        output = complet_packet
    elif iteration==nop-1:
        output = last_paacket
    
    return output


def choose_symbol(data, iteration):
    symbol = data[48*iteration:48*(1+iteration)]
    return symbol


def choose_recieved_symbol(dat, it):
    symbol = dat[80*it:80*(it+1)]
    return symbol

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

def Correlation(Packet_RX):

    

############################################# Long_preamble ##################################################

    long = np.array([0,1,-1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1])
    long_out = IFFT(long)
    C_long = add_CP(long_out)
    
############################################ Short preamble ###################################################

    x = math.sqrt(13.6)
    short = np.array([0,0,0,0,-1-1j,0,0,0,-1-1j,0,0,0,1+1j,0,0,0,1+1j,0,0,0,1+1j,0,0,0,1+1j,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1+1j,0,0,0,-1-1j,0,0,0,1+1j,0,0,0,-1-1j,0,0,0,-1-1j,0,0,0,1+1j,0,0,0])
    short_in = short*x
    short_out = IFFT(short_in)
    C_short = short_out[-16:]
    i = 0
    j = 0
    m = 0
    n = 0
    
    while i<10:
        c1 = np.corrcoef(C_short, Packet_RX[0+j:16+j]) 
        j = j+1
        
        if c1[0,1]>0.7:
            i = i+1        
    
    
    while m<2:
        
        c2 = np.corrcoef(C_long, Packet_RX[15+j+n:95+j+n])
        start = 95+j+n
        n = n+1

        
        if c2[0,1]>0.7:
            m = m+1
    
    return start-320



def add_silence(data,st):
    if synch_type == 'defined':
        silence = np.zeros(st)
    elif synch_type == 'corr':
        st = np.random.randint(100,high=600,size=1)
        silence = np.zeros(st)
    return np.hstack((silence,data))

def synch(data):
    if synch_type == 'defined':
        time_signal = np.delete(data, range(st))
    elif synch_type == 'corr':
        start_time = Correlation(data)
        time_signal = np.delete(data,range(start_time))
    return time_signal

def make_packet(chosen_packet, mod_type, tail, service,sfft):
    
    " add service and tail "
    sdatat = np.hstack((service, chosen_packet, tail))    
    
    " add pad "
    p_data,plen = add_pad(sdatat,mod_type)
    
    "scrambling data"
    s_data = scrambler(p_data,16,6)
    
    "encoding data"
    en_data = test_encoder(s_data, mod_type["coding rate"])   ##encoded data
    
    " interleaving "
    in_data = interleaver_a(en_data, mod_type["Ncbps"], mod_type["Nbpsc"])     ##interleaved data
    
    " modulation "
    mo_data = modulation(in_data,mod_type)      ##modulated data
    
    
    
    
    
    " make SIGNAL symbol "
    signal = make_signal(mod_type["RATE"], in_data, mod_type["Ncbps"])
    
    " add SIGNAL to the begining of the packet "
    adde_signal = np.hstack((signal,mo_data))
    
    " Calcualte number of OFDM symbol per packet "
    nos = int(len(adde_signal)/48) 
    
    """ 
    loop:
    choose_symbol: choose modulated symbols for each OFDM symbol

    """
    time_signal = np.array([])
    for i in range(nos):
        ch_symbol = choose_symbol(adde_signal, i)
        
        
        all_symbol = subcarrier_allocation(ch_symbol)     
        ifft_symbol = ifft_pilot(all_symbol,sfft)
        
        add_cp_symbol = add_CP(ifft_symbol)
        time_signal = np.append(time_signal,add_cp_symbol)
    
    " add preamble to the begining of the packet "
    time_signal = np.hstack((preamble, time_signal))
    
    "add silence time befor each packet"
    time_signal = add_silence(time_signal,400)
   
    return time_signal,plen

def extract_packet(t_signal, mod_type, len_pad, sfft):     
    
    " estimate begining of the packet"
    t_signal = synch(t_signal)
    
    " delet preamble "
    t_signal = np.delete(t_signal,range(320))
    

    " extract packet len "
    pack_len,mod_rate = extract_signal(t_signal)    
    
    " determine packet length and rate"
    """
     loop:
     choose recieved symbol from each packet to get detected
    """
    adde_signal = []
    for i in range(pack_len+1):
        ch_r_symbol = choose_recieved_symbol(t_signal, i)
        rem_cp_symbol = remove_CP(ch_r_symbol)
        fft_symbol,pilot_value = fft_pilot(rem_cp_symbol, sfft) 
        exall_symbol = extract_data(fft_symbol)
        
        
        adde_signal = np.append(adde_signal,exall_symbol)
    
    """ detect rate ,
     return modulated data in each packet,
     calculate length of each packet"""
    

    " detect modulation type from detected rate "
    mod_type = find_mod(mod_rate)
    
    " demodulation of packet symbols"
    demo_data = demodulation(adde_signal[48:], mod_type, 'data')
    
    " deinterleaving "
    de_data = deinterleaver_a(demo_data, mod_type["Ncbps"], mod_type["Nbpsc"])
    
    " decoding "
    dec_data = test_decoder(de_data, mod_type["coding rate"]) 
    
    " descrambling "
    des_data = descrambler(dec_data,16)
    
    " remove pad"
    sdatat = delet_pad(des_data, len_pad)
    
    " remove tail and service and return transmited data bits of each packet "
    data = sdatat[len(service):len(sdatat)-len(tail)]
    
    return data

"_____________________________________________________START_________________________________________________ "

"Generating data"
source = np.random.randint(0, high=2, size=100000, dtype=int)         
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
    print(f'progress bar: {(i/nop)*100}%')
    ch_data = choose_packet(source,nop,i,ndbppts)       
    
    t_signal,plen = make_packet(ch_data, mod, tail, service, sfft=64)
    
    a_ch_signal = channel_type(t_signal,snr,channel)
    
    ex_data = extract_packet(a_ch_signal, mod, plen, sfft=64)
    
    sourcehat = np.append(sourcehat,ex_data) 
   
    

print(f'len sourcehat: {len(sourcehat)}')
print(f'len source: {len(source)}')
num_of_error = 0
for i in range(len(source)):
    if sourcehat[i] != source[i]:
        num_of_error +=1

print(f'len of sourcehat: {len(sourcehat)}')
print(f'number of error is: {num_of_error}')
BER = num_of_error/len(sourcehat)
print(f'the BER is: {BER}')

out_bit = np.array([int(x) for x in sourcehat])

outbyt=np.packbits(out_bit)
outbyt.tofile('outbit.png')
