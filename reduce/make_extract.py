def modulation(in_data, modulationtype):            
    if modulationtype == bpsk_6 or modulationtype == bpsk_9:
        modulated_data = bpsk(in_data)
    elif modulationtype == qpsk_12 or modulationtype == qpsk_18:
        modulated_data = QPSK(in_data)
    elif modulationtype == qam16_24 or modulationtype == qam16_36:
        modulated_data = QAM16(in_data)
    elif modulationtype == qam64_48 or modulationtype == qam64_54:
        modulated_data = QAM64(in_data)
    return modulated_data


def demodulation(mo_data, modulationtype, datatype):         #demodulation based on modulated type 
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

def make_signal(mod_rate, in_data, Ncbps):     # get interleaved data and generate signal's modulated symbol 
    noos = int(len(in_data)/Ncbps)             # number if ofdm symbol per packet "
    bi_noos = bin(noos)      
    bi_noos = bi_noos.split('b')
    bi_noos = [int(x) for x in bi_noos[1]]
    bi_noos = np.array(bi_noos)
    signal = SIGNAL(mod_rate, bi_noos)     # generate signal 
    en_signal = test_encoder(signal,1/2)    # encoding signal 
    int_signal = interleaver_a(en_signal, 48, 1)    # interleaving signal 
    mo_signal = modulation(int_signal, bpsk_6)       # modulate signal's bits
    
    return mo_signal

def channel_type(Data,snr,chtype):     # if 'chtype : awgn' use awgn channel
    if chtype == 'awgn':               # if 'chtype : rayleigh' use rayleigh channel
        outdata = awgn(Data, snr)
    if chtype == 'rayleigh':
        outdata = simple_fading(fade_vec, Data)
        outdata = awgn(outdata,snr)
    return outdata

def extract_signal(data,factor):     # gets signal and return legth of packet(nomber of ofdm symbols per packet)
    signal = data[0:80]              # and RATE
    rcp_signal = remove_CP(signal)
    fft_signal,h = fft_pilot(rcp_signal,64,factor)
    deallo_signal = extract_data(fft_signal)
    de_signal = demodulation(deallo_signal, bpsk_6, 'signal')
    deint_signal = deinterleaver_a(de_signal, 48, 1)
    dec_signal = test_decoder(deint_signal, 1/2)
    p_len = dec_signal[5:17]
    mod = dec_signal[0:4]
    packet_len_time = int("".join(str(i) for i in p_len),2)
    return packet_len_time,mod


def find_mod(mod_rate):               #find modulation type from transmitted RATE(bit) 
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


def choose_symbol(data, iteration):          # choose modulated symbol for each OFDM symbol
    symbol = data[48*iteration:48*(1+iteration)]
    return symbol


def choose_recieved_symbol(dat, it):        # choose OFDM symbol to take fft 
    symbol = dat[80*it:80*(it+1)]
    return symbol

def add_pad(data, mod):                     # add pad to each packet to complet last OFDM symbol in the packet "
    p_data = data
    pad_len = 0                             #pad length
    if len(data)%mod["Ndbps"] != 0:
        pad = np.zeros(mod["Ndbps"]-(len(data)%mod["Ndbps"]))
        p_data = np.hstack((data,pad))      # data with pad
        pad_len = len(pad)
    return p_data,pad_len

def pad2service(pad_length):                    # add legth(binary) of pad to service  
    bi_pad = bin(pad_length) 
    bi_pad = bi_pad.split('b')
    bi_pad = [int(x) for x in bi_pad[1]]
    bi_pad = np.array(bi_pad)
    for i in range(len(bi_pad)):
       service[15-i] = bi_pad[-i+len(bi_pad)-1]
    for i in range(8-len(bi_pad)):
       service[i+5] = 0
    return service



def delet_pad(p_data, pad_len):                   # p_data :data with pad 
    data = np.delete(p_data,range(len(p_data)-pad_len,len(p_data)))
    return data


def add_silence(data,st):                #add silence time before the transmitted packet 
    if synch_type == 'defined':          # if 'synch_type : corr' silence time is random 
        silence = np.zeros(st)           # if 'synch_type : defined' silence time is defined 
    elif synch_type == 'corr':
        st = np.random.randint(100,high=st,size=1)
        silence = np.zeros(st)
    return np.hstack((silence,data))




def synch(data):                          # if 'synch_type : corr' then uses correlation for synchronization
    if synch_type == 'defined':
        time_signal = np.delete(data, range(st))
    elif synch_type == 'corr':
        start_time = Correlation(data)
        time_signal = np.delete(data,range(start_time))
    return time_signal


def rec_pad_len(data):                           #detect number of bits added as pad
    pad_len = data[7:16]
    pad_len = int("".join(str(i) for i in pad_len),2)
    return pad_len


def equa_fa(long_preamble_freq,data):            # return a equalization  factor 
    if use_eq == 'on':                           #if 'use_eq : on' then uses equalizer 
        l_preamble = data[256:320]
        l_preamble = np.fft.fft(l_preamble)
        factor = l_preamble/long_preamble_freq
    else:
        factor = np.full(64,1)
    return factor

def make_packet(chosen_packet, mod_type, tail, service,sfft):
    
    " add service and tail "
    sdatat = np.hstack((service, chosen_packet, tail))    
    
    " add pad "
    p_data,plen = add_pad(sdatat,mod_type)
    
    " add length of pad to service"
    service = pad2service(plen)
    p_data[0:16] = service
    
    "scrambling data"
    s_data = scrambler(p_data,16,6)
    
    "encoding data"
    en_data = test_encoder(s_data, mod_type["coding rate"])  
    
    " interleaving "
    in_data = interleaver_a(en_data, mod_type["Ncbps"], mod_type["Nbpsc"])     
    
    " modulation "
    mo_data = modulation(in_data,mod_type)     
    
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
    time_signal = add_silence(time_signal,st)
   
    return time_signal

def extract_packet(t_signal, mod_type, sfft):     
    
    " estimate begining of the packet "
    t_signal = synch(t_signal)
    
    " return a equalization  factor "
    factor = equa_fa(long,t_signal)
    
    " delet preamble "
    t_signal = np.delete(t_signal,range(320))
    
    " extract packet length and transmitted rate "
    pack_len,mod_rate = extract_signal(t_signal,factor)    
    
    
    """
     loop:
     choose recieved symbol from each packet to get detected
    """
    adde_signal = []
    for i in range(pack_len+1):
        
        ch_r_symbol = choose_recieved_symbol(t_signal, i)
        
        rem_cp_symbol = remove_CP(ch_r_symbol)
        
        fft_symbol,pilot_value = fft_pilot(rem_cp_symbol, sfft, factor) 
        
        exall_symbol = extract_data(fft_symbol)
        
        adde_signal = np.append(adde_signal,exall_symbol)
        

    " detect modulation type from detected rate "
    mod_type = find_mod(mod_rate)
    
    " demodulation of symbols per packet"
    demo_data = demodulation(adde_signal[48:], mod_type, 'data')
    
    " deinterleaving "
    de_data = deinterleaver_a(demo_data, mod_type["Ncbps"], mod_type["Nbpsc"])
    
    " decoding "
    dec_data = test_decoder(de_data, mod_type["coding rate"]) 
    
    " descrambling "
    des_data = descrambler(dec_data,16)
    
    " detect length of added pad "
    pad_len = rec_pad_len(des_data)
    
    " remove pad"
    sdatat = delet_pad(des_data, pad_len)
    
    " remove tail and service and return transmited data bits of each packet "
    data = sdatat[len(service):len(sdatat)-len(tail)]
    
    return data
