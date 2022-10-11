# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 15:25:36 2021

@author: Hoang
"""

import numpy as np


# This function was referenced from https://github.com/zhaonat/Rigorous-Coupled-Wave-Analysis
def convmat2D(A, P, Q):    
    N = A.shape;

    NH = (2*P+1) * (2*Q+1) ;
    p = list(range(-P, P + 1)); 
    q = list(range(-Q, Q + 1));
  
    Af = (1 / np.prod(N)) * np.fft.fftshift(np.fft.fft2(A));    
    
    p0 = int((N[1] / 2));     q0 = int((N[0] / 2)); 

    ret = np.zeros((NH, NH),dtype=complex)
    
    for qrow in range(2*Q+1): 
        for prow in range(2*P+1):             
            row = (qrow) * (2*P+1) + prow; 
            for qcol in range(2*Q+1):
                for pcol in range(2*P+1):
                    col = (qcol) * (2*P+1) + pcol; 
                    pfft = p[prow] - p[pcol]; 
                    qfft = q[qrow] - q[qcol];
                    ret[row, col] = Af[q0 + pfft, p0 + qfft]; 

    return ret

