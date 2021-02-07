# convolutional coder with Rate 1/2 
import numpy as np
import sk_dsp_comm.fec_conv as fec
def Convolutional_coder1(data):
    coded_data = np.array([])
    cc1 = fec.fec_conv(('1111001','1011011'),Depth=30)    # use pakage sk_dsp_comm.fec_conv for coded and decoced
    coded_data= cc1.conv_encoder(data,'000000')
    return coded_data
#Data =np.array([0,1,1,0,1,0,0,0,0,1,1,0])
#ja=Convolutional_coder1(Data)
#print(ja)

# convolutional coder with Rate 3/4 and Rate 2/3
def Convolutional_coder2(data,Rate):
  coded=Convolutional_coder1(data) # rate=1/2
  C= list(coded[0])
  print('c=',C)
  print(type(C))
  Coder2=[]
  Coder3=[]
  List = []
  if Rate == 3/4:
    i=-1
    for n in range(0,len(C)):
       List.append(C[n])
       i=i+1
       if i == 5 :
         List[3:5] = []
         Coder2.extend(List)
         i=-1
         List[ : ] = []
    return Coder2
   
  elif Rate == 2/3:
    i=-1
    for n in range(0,len(C)):
      List.append(C[n])
      i=i+1
      if i == 3 :
         del(List[3])   #inja jay 3 , 2 bod ke eshtebah shode bod!!!!
         Coder3.extend(List)
         i=-1
         List[ : ] = []
    return Coder3 
