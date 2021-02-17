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

