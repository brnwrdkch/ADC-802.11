def scrambler_core(inp_vec, Initial_Vec):
  vec_len = len(inp_vec)
  out_vec = np.zeros(vec_len,dtype=np.int8)
 
  scrambler_vec = Initial_Vec; # initializing the scrambler
 
  for i in range(vec_len):
    temp = np.bitwise_xor(int(scrambler_vec[3]) , int(scrambler_vec[6]))
    scrambler_vec[1:] = scrambler_vec[0:6]
    scrambler_vec[0] = temp
    out_vec[i] = np.bitwise_xor(int(scrambler_vec[0]) , int(inp_vec[i]))
 
  return out_vec
 
# __________________Scrambler unit __________________
 
def scrambler (inp_vec, service_len, tail_len):
  vec_len = len(inp_vec)
  out_vec = np.zeros(vec_len,dtype=np.int8)
  
  
  Initial_Vec = inp_vec[0:7]
  out_vec[0:service_len] = inp_vec[0:service_len]
  out_vec[vec_len-tail_len:] = inp_vec[vec_len-tail_len:]

  out_vec[service_len:vec_len-tail_len] = scrambler_core(inp_vec[service_len:vec_len-tail_len],Initial_Vec)
 
  return out_vec
 
# __________________Decrambler unit __________________
 
def descrambler (inp_vec, service_len):
  descrambler_init = inp_vec[0:7]
 
  vec_len = len(inp_vec)
  out_vec = np.zeros(vec_len,dtype=np.int8)
  out_vec[0:service_len] = inp_vec[0:service_len]
  out_vec[service_len:] = scrambler_core(inp_vec[service_len:],descrambler_init)
  return out_vec
