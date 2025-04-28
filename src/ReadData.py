# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 20:36:26 2022

@author: Hoang
"""


import numpy as np
from scipy.interpolate import  interp1d
import re
import os


def read_nk(path):
    f=open(path)
    f1=f.readlines()
    f_str=np.array([i.split() for i in f1 if len(i)!=1])
    f_str=f_str[1:]
    if len(f_str[0])==3:
        wl=f_str[:,0].astype('float')
        n=f_str[:,1].astype('float')
        k=f_str[:,2].astype('float')        
        return wl, [n,k]  
    else:
        wl=f_str[:,0].astype('float')
        n=f_str[:,1].astype('float')       
        return wl, [n]    

def interpolate_nk(wl_target,wl_data, nk_data):    # N=n+ik  
    
    n_data=nk_data[0]    
    n_interpolate = interp1d(wl_data,n_data)
    
    if len(nk_data)>1:
        k_data=nk_data[-1]   
        k_interpolate = interp1d(wl_data,k_data)
        
    if len(nk_data)>1: 
        nk_index=np.array([n_interpolate(wl_i)+1j*k_interpolate(wl_i) for wl_i in wl_target])   
    else:        
        nk_index=np.array([n_interpolate(wl_i) for wl_i in wl_target]) 
    
    return nk_index

def interpolate_eps(wl_target,wl_data, nk_data):      
    nk_index=interpolate_nk(wl_target,wl_data, nk_data)  
    return np.conj(nk_index**2)  #eps=e1-je2

def interpolate(wl,*arg):    
    if len(arg)==1:
        return [interpolate_eps(wl,a[0], a[1]) for a  in arg][0]
    else: 
        return [interpolate_eps(wl,a[0], a[1]) for a  in arg]

