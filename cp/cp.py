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
