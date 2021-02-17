###### "ifft_pilot"

def ifft_pilot(i_data, nfft):          ## i_data = input data (freq. domain signal)
    p_index = np.array([7, 21, 43, 57], dtype=int)          ## pilot index
    p_value = np.array([1, 1, 1, -1], dtype=complex)          ## pilot value
    for i in range(len(p_index)):
        i_data[p_index[i]] = p_value[i]
    o_data = np.fft.ifft(i_data, n=nfft)          ## output data
    
    return o_data          ## returns time domain signal 
    
    

###### "fft block"


def fft_pilot(i_data,nfft,factor):          ## i_data = input data (time domain signal)
    fd_signal = np.fft.fft(i_data, nfft)          #freq. domain data
    fd_signal = fd_signal/factor                   # equalize OFDM symbol
    p_index = np.array([7, 21, 43, 57], dtype=int)          ## pilot index
    p_value = np.zeros(4, dtype=complex)
    for i in range(len(p_index)):
        p_value[i] = fd_signal[p_index[i]]
    for i in range(len(p_index)):
        fd_signal[p_index[i]] = 0
    return fd_signal, p_value          ##  freq. domain signal , p_value for eq.
