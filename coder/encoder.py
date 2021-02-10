## encoder for test
def test_encoder(inputd, rate):     
    L = len(inputd)
    if rate == 1/2:
        codbit = np.zeros(L)
    elif rate == 2/3:
        codbit = np.zeros(int(L/2))
    elif rate == 3/4:
        codbit = np.zeros(int(L/3))
    outputd = np.hstack((inputd,codbit))
    return outputd

def test_decoder(inputd,rate):
    L = len(inputd)
    if rate == 1/2:
        codbit = np.delete(inputd,range(int(L/2),L))
    elif rate == 2/3:
        codbit = np.delete(inputd,range(int((2*L)/3),L))
    elif rate == 3/4:
        codbit = np.delete(inputd,range(int((3*L)/4),L))
    
    return codbit
