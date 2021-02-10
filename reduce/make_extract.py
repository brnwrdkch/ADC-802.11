" Omid Abasi "

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


##awgn channel
def awgn(Data,snr):
    import numpy as np
    import matplotlib.pyplot as plt
    import math
    squared_numbers =  [abs(x)**2 for x in Data]
    n=len(squared_numbers)
    f= sum(squared_numbers) / n# averaging of signal power
    z = np.random.normal(loc=0, scale=np.sqrt(2)/2, size=(n,2)).view(np.complex128) # random noise generating 
    y1 =list(z) 
    
    a=math.sqrt((10**( snr/10 ))/f)
    multiplied_list = [element * a for element in Data]
    h=list(np.zeros(n))
    for i in range(n):
              h[i]=multiplied_list[i] + y1[i]
    return h




## extract data

def extract_signal(data):
    signal = data[0:48]
    de_signal = demodulation(signal, bpsk_6)
    deint_signal = deinterleaver_a(de_signal, 48, 1)
    dec_signal = test_decoder(deint_signal, 1/2)
    p_len = dec_signal[5:17]
    mod = dec_signal[0:4]
    p_data = data[48:]
    return mod, p_len, p_data

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
    return time_signal,plen

def extract_packet(t_signal, mod_type, len_pad, sfft):     
    
    " delet preamble "
    t_signal = np.delete(t_signal,range(320))
    
    """
     loop:
     choose recieved symbol from each packet to get detected
    """
    adde_signal = []
    for i in range(int(len(t_signal)/80)):
        ch_r_symbol = choose_recieved_symbol(t_signal, i)
        rem_cp_symbol = remove_CP(ch_r_symbol)
        fft_symbol,pilot_value = fft_pilot(rem_cp_symbol, sfft) 
        exall_symbol = extract_data(fft_symbol)
        adde_signal = np.append(adde_signal,exall_symbol)
    
    """ detect rate ,
     return modulated data in each packet,
     calculate length of each packet"""
    mod_rate,packet_len,mo_data = extract_signal(adde_signal)
    
    " detect modulation type from detected rate "
    mod_type = find_mod(mod_rate)
    
    " demodulation of packet symbols"
    demo_data = demodulation(mo_data, mod_type)
    
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
