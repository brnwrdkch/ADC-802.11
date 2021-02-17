def preamble():
    ############################################# Long_preamble ##################################################

    
    long_out = IFFT(long)
    long_withCP = add_CP(long_out)
    preamble_long = np.hstack([long_withCP,long_withCP])

    ############################################ Short preamble ###################################################

    x = math.sqrt(13.6)
    short = np.array([0,0,0,0,-1-1j,0,0,0,-1-1j,0,0,0,1+1j,0,0,0,1+1j,0,0,0,1+1j,0,0,0,1+1j,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1+1j,0,0,0,-1-1j,0,0,0,1+1j,0,0,0,-1-1j,0,0,0,-1-1j,0,0,0,1+1j,0,0,0])
    short_in = short*x
    short_out = IFFT(short_in)
    short_withCP = add_CP(short_out)
    preamble_short = np.hstack([short_withCP,short_withCP])

    ################################################################################################################

    preamble = np.hstack([preamble_short,preamble_long])
    return preamble
