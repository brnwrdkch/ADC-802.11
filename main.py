"""

ADC's Project"   _______________     "2021 winter"

"""   

"_____________________________________________________START_________________________________________________ "

"Generating data"
source = np.random.randint(0, high=2, size=10000, dtype=int)         

"Calculating number of packet"
nop = math.ceil(len(source)/(ndbppts)) 

"Detected data"
sourcehat = []

"""
Loop :  
choose packet: calculate number of data bits for each packet
make packet: make a time domain packet to send through channel
extract packet : take each packet and try to detect the transmitted bits
"""
for i in range(nop):
    ch_data = choose_packet(source,nop,i,ndbppts)       
    t_signal,plen = make_packet(ch_data, mod, tail, service, sfft=64)   
    a_ch_signal = awgn(t_signal,snr)
    ex_data = extract_packet(a_ch_signal, mod, plen, sfft=64)
    sourcehat = np.append(sourcehat,ex_data) 


"""test"""
for i in range(5):
    print(i)