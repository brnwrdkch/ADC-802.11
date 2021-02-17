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

