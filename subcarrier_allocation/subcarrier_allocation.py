" Omid Abasi"

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
