" Hamed Qanbari "


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
