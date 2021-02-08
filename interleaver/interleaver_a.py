def interleaver(data,Ncbps,Nbpsc):
    data = np.reshape(data,(int(len(data)/Ncbps),Ncbps))
    output = np.array([])
    for j in range(len(data)):
        i_data = np.array([])       # interleaved data
        arranged = np.reshape(data[j],(Nbpsc,48))      # arranged to get interleaved
        for i in range(48): 
            i_data = np.append(i_data,arranged[:,i])
        output = np.append(output,i_data)
    return output
