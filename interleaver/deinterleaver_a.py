def deinterleaver(data,Ncbps,Nbpsc):
    data = np.reshape(data,(int(len(data)/Ncbps),Ncbps))
    output = np.array([])
    for j in range(len(data)):
        de_data = np.array([])
        arranged = np.reshape(data[j], (48,Nbpsc))
        for i in range(Nbpsc):
            de_data = np.append(de_data,arranged[:,i])
        output = np.append(output,de_data)
    return output
