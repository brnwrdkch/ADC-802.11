" Hamed Qanbari " 

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
    return h



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
          h.extend(y1[min_value])
    return h
