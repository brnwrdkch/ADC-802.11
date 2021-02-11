#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import division

import functools
import math
from warnings import warn
import commpy.channelcoding.convcode as cc

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection

from commpy.utilities import dec2bitarray, bitarray2dec, hamming_dist, euclid_dist



def conv_encode(message_bits, trellis, termination = 'cont', puncture_matrix=None):

    k = trellis.k
    n = trellis.n
    total_memory = trellis.total_memory
    rate = float(k)/n
    
    code_type = trellis.code_type

    puncture_matrix = np.ones((trellis.k, trellis.n))

    number_message_bits = np.size(message_bits)
    

    inbits = message_bits
    number_inbits = number_message_bits
    number_outbits = int(number_inbits/rate)
    outbits = np.zeros(number_outbits, 'int')
    p_outbits = np.zeros(int(number_outbits*
        puncture_matrix[0:].sum()/np.size(puncture_matrix, 1)), 'int')
    next_state_table = trellis.next_state_table
    output_table = trellis.output_table

    # Encoding process - Each iteration of the loop represents one clock cycle
    current_state = 0
    j = 0

    for i in range(int(number_inbits/k)): # Loop through all input bits
        current_input = bitarray2dec(inbits[i*k:(i+1)*k])
        current_output = output_table[current_state][current_input]
        outbits[j*n:(j+1)*n] = dec2bitarray(current_output, n)
        current_state = next_state_table[current_state][current_input]
        j += 1

    j = 0
    for i in range(number_outbits):
        if puncture_matrix[0][i % np.size(puncture_matrix, 1)] == 1:
            p_outbits[j] = outbits[i]
            j = j + 1

    return p_outbits


def _where_c(inarray, rows, cols, search_value, index_array):

    number_found = 0
    res = np.where(inarray == search_value)
    i_s, j_s = res
    for i, j in zip(i_s, j_s):
        if inarray[i, j] == search_value:
            index_array[number_found, 0] = i
            index_array[number_found, 1] = j
            number_found += 1

    return number_found


@functools.lru_cache(maxsize=128, typed=False)
def _compute_branch_metrics(decoding_type, _r_codeword: tuple, _i_codeword_array: tuple):
    r_codeword = np.array(_r_codeword)
    i_codeword_array = np.array(_i_codeword_array)
    if decoding_type == 'hard':
        return hamming_dist(r_codeword.astype(int), i_codeword_array.astype(int))
    elif decoding_type == 'soft':
        neg_LL_0 = np.log(np.exp(r_codeword) + 1)  # negative log-likelihood to have received a 0
        neg_LL_1 = neg_LL_0 - r_codeword  # negative log-likelihood to have received a 1
        return np.where(i_codeword_array, neg_LL_1, neg_LL_0).sum()
    elif decoding_type == 'unquantized':
        i_codeword_array = 2 * i_codeword_array - 1
        return euclid_dist(r_codeword, i_codeword_array)


def _acs_traceback(r_codeword, trellis, decoding_type,
                   path_metrics, paths, decoded_symbols,
                   decoded_bits, tb_count, t, count,
                   tb_depth, current_number_states):

    k = trellis.k
    n = trellis.n
    number_states = trellis.number_states
    number_inputs = trellis.number_inputs

    branch_metric = 0.0

    next_state_table = trellis.next_state_table
    output_table = trellis.output_table
    pmetrics = np.empty(number_inputs)
    index_array = np.empty([number_states, 2], 'int')

    # Loop over all the current states (Time instant: t)
    for state_num in range(current_number_states):

        # Using the next state table find the previous states and inputs
        # leading into the current state (Trellis)
        number_found = _where_c(next_state_table, number_states, number_inputs, state_num, index_array)

        # Loop over all the previous states (Time instant: t-1)
        for i in range(number_found):

            previous_state = index_array[i, 0]
            previous_input = index_array[i, 1]

            # Using the output table, find the ideal codeword
            i_codeword = output_table[previous_state, previous_input]
            i_codeword_array = dec2bitarray(i_codeword, n)

            # Compute Branch Metrics
            branch_metric = _compute_branch_metrics(decoding_type, tuple(r_codeword), tuple(i_codeword_array))

            # ADD operation: Add the branch metric to the
            # accumulated path metric and store it in the temporary array
            pmetrics[i] = path_metrics[previous_state, 0] + branch_metric

        # COMPARE and SELECT operations
        # Compare and Select the minimum accumulated path metric
        path_metrics[state_num, 1] = pmetrics.min()

        # Store the previous state corresponding to the minimum
        # accumulated path metric
        min_idx = pmetrics.argmin()
        paths[state_num, tb_count] = index_array[min_idx, 0]

        # Store the previous input corresponding to the minimum
        # accumulated path metric
        decoded_symbols[state_num, tb_count] = index_array[min_idx, 1]

    if t >= tb_depth - 1:
        current_state = path_metrics[:,1].argmin()

        # Traceback Loop
        for j in reversed(range(1, tb_depth)):

            dec_symbol = decoded_symbols[current_state, j]
            previous_state = paths[current_state, j]
            decoded_bitarray = dec2bitarray(dec_symbol, k)
            decoded_bits[t - tb_depth + 1 + (j - 1) * k + count:t - tb_depth + 1 + j * k + count] = decoded_bitarray
            current_state = previous_state

        paths[:,0:tb_depth-1] = paths[:,1:]
        decoded_symbols[:,0:tb_depth-1] = decoded_symbols[:,1:]



def viterbi_decode(coded_bits, trellis, tb_depth=None, decoding_type='unquantized'):

    # k = Rows in G(D), n = columns in G(D)
    k = trellis.k
    n = trellis.n
    rate = k/n
    total_memory = trellis.total_memory

    # Number of message bits after decoding
    L = int(len(coded_bits)*rate)

    if tb_depth is None:
        tb_depth = min(5 * total_memory, L)


    path_metrics = np.full((trellis.number_states, 2), np.inf)
    path_metrics[0][0] = 0
    paths = np.empty((trellis.number_states, tb_depth), 'int')
    paths[0][0] = 0

    decoded_symbols = np.zeros([trellis.number_states, tb_depth], 'int')
    decoded_bits = np.empty(int(math.ceil((L + tb_depth) / k) * k), 'int')
    r_codeword = np.zeros(n, 'int')

    tb_count = 1
    count = 0
    current_number_states = trellis.number_states

    for t in range(1, int((L+total_memory)/k)):
        # Get the received codeword corresponding to t
        if t <= L // k:
            r_codeword = coded_bits[(t-1)*n:t*n]
        # Pad with '0'
        else:
            r_codeword[:] = -1


        _acs_traceback(r_codeword, trellis, decoding_type, path_metrics, paths,
                decoded_symbols, decoded_bits, tb_count, t, count, tb_depth,
                current_number_states)

        if t >= tb_depth - 1:
            tb_count = tb_depth - 1
            count = count + k - 1
        else:
            tb_count = tb_count + 1

        # Path metrics (at t-1) = Path metrics (at t)
        path_metrics[:, 0] = path_metrics[:, 1]

    return decoded_bits[:len(message_bits)]


