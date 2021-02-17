def Correlation(Packet_RX):

    

############################################# Long_preamble ##################################################

    long = np.array([0,1,-1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1,1,1,-1,-1,1,1,-1,1,-1,1,1,1,1])
    long_out = IFFT(long)
    C_long = add_CP(long_out)
    
############################################ Short preamble ###################################################

    x = math.sqrt(13.6)
    short = np.array([0,0,0,0,-1-1j,0,0,0,-1-1j,0,0,0,1+1j,0,0,0,1+1j,0,0,0,1+1j,0,0,0,1+1j,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1+1j,0,0,0,-1-1j,0,0,0,1+1j,0,0,0,-1-1j,0,0,0,-1-1j,0,0,0,1+1j,0,0,0])
    short_in = short*x
    short_out = IFFT(short_in)
    C_short = short_out[-16:]
    i = 0
    j = 0
    m = 0
    n = 0
    X = np.array([])
    Y = np.array([])
    while i<10:
        c1 = np.corrcoef(C_short, Packet_RX[0+j:16+j]) 
        X = np.append(X,j)
        Y = np.append(Y,c1[0,1])
        j = j+1
        if c1[0,1]>short_th:
            i = i+1 
        if j>5*st:
            print("can't find short preambles")
            sys.exit()
    if plot_corr_curve == 'on':
        plt.plot(X,Y)
        plt.show()
        X = np.array([])
        Y = np.array([])
    while m<2:
        
        c2 = np.corrcoef(C_long, Packet_RX[15+j+n:95+j+n])
        start = 95+j+n
        n = n+1
        X = np.append(X,n+j+15)
        Y = np.append(Y,c2[0,1])
        if c2[0,1]>long_th:
            m = m+1
        if start>10*st:
            print("can't find long preambles")
            sys.exit()
    if plot_corr_curve == 'on':
        plt.plot(X,Y)
        plt.show()
    return start-320
