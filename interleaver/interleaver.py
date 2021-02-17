def interleaver_Core(inp_vec, N_CBPS, N_BPSC):
  s = max(N_BPSC/2,1)

  inner_vec = np.ones(N_CBPS,dtype=np.int8)
  for cnt in range(N_CBPS):
    i = (N_CBPS/16) * (cnt % 16) + math.floor(cnt/16)
    inner_vec[int(i)] = inp_vec[cnt]
  
  outp_vec = np.ones(N_CBPS,dtype=np.int8)
  for cnt in range(N_CBPS):
    j = (s * math.floor(cnt/s)) + ((cnt + N_CBPS - math.floor(16*cnt/N_CBPS)) % s)
    outp_vec[int(j)] = inner_vec[cnt]

  return outp_vec

def interleaver_a(inp_vec, N_CBPS, N_BPSC):
  inp_len = len(inp_vec)
  iteration = inp_len/N_CBPS

  outp_vec = np.ones(inp_len,dtype=np.int8)
  for cnt in range(int(iteration)):
    vec = inp_vec[N_CBPS*cnt:N_CBPS*(cnt+1)]
    outp_vec[N_CBPS*cnt:N_CBPS*(cnt+1)] = interleaver_Core(vec, N_CBPS, N_BPSC)
  
  return outp_vec


# __________________Deinterleaver unit __________________
def deinterleaver_Core (inp_vec, N_CBPS, N_BPSC):
  s = max(N_BPSC/2,1)

  inner_vec = np.zeros(N_CBPS,dtype=np.int8)
  for cnt in range(N_CBPS):
    i = (s * math.floor(cnt/s)) + ((cnt + math.floor(16*cnt/N_CBPS)) % s)
    inner_vec[int(i)] = inp_vec[cnt]
  
  outp_vec = np.zeros(N_CBPS,dtype=np.int8)
  for cnt in range(N_CBPS):
    k = 16 * cnt - (N_CBPS-1) * math.floor(16*cnt/N_CBPS)
    outp_vec[int(k)] = inner_vec[cnt]

  return outp_vec

def deinterleaver_a(inp_vec, N_CBPS, N_BPSC):

  inp_len = len(inp_vec)
  iteration = inp_len/N_CBPS

  outp_vec = np.ones(inp_len,dtype=np.int8)
  for cnt in range(int(iteration)):
    vec = inp_vec[N_CBPS*cnt:N_CBPS*(cnt+1)]
    outp_vec[N_CBPS*cnt:N_CBPS*(cnt+1)] = deinterleaver_Core(vec, N_CBPS, N_BPSC)

  return outp_vec
