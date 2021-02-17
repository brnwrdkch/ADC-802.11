###______Omid Abasi___9912344________###
def awgn(Data,SNR):           # awgn channel, SNR(db)
   SNR = 10**(SNR/10)          
   len_data = len(Data)
   real_noise = np.random.normal(0,math.sqrt(0.5),len(Data))         # generate noise with 'average power=1/2' 
   imag_noise = np.random.normal(0,math.sqrt(0.5),len(Data))
   factor = np.full(shape=len(Data),fill_value=1j,dtype=complex)
   imag_noise = np.prod([imag_noise,factor],axis=0)                 
   noise = np.sum([real_noise,imag_noise],axis=0)      #add imag and real noise and generate noise with 
   factor = np.full(shape=len(Data),fill_value=1/math.sqrt(SNR),dtype=complex)      # 'average power=1'
   noise = np.prod([noise,factor],axis=0)               # change noise power to 1/SNR"
   noisy_data = np.sum([Data,noise],axis=0)              # add noise to data "
   return noisy_data
   
   
   ### hamed ghanbari
   def simple_fading(fading,Data):
    # Data is the output of IFFT in TX
    # fading=[1 ,.025, .075]
    fading=np.array(fading)# Impulse response of input fading
    h1=np.convolve(fading, Data)# Convolutional method
    n=len(Data)
    h=h1[0:n]
    return h
