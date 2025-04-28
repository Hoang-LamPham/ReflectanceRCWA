#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 12:07:24 2023

@author: pham
"""
import numpy as np
     

def matrix(J11,J12,J21,J22): #rpp,rps,rsp,rss
    
    m11=0.5*(np.abs(J11)**2 + np.abs(J22)**2 + np.abs(J12)**2 + np.abs(J21)**2 )
    m12=0.5*(np.abs(J11)**2 - np.abs(J22)**2 - np.abs(J12)**2 + np.abs(J21)**2 )
    m13= np.real(np.conj(J11)*J12 + np.conj(J21)*J22)
    m14=-np.imag(np.conj(J11)*J12 + np.conj(J21)*J22)  
   
    m22=0.5*(np.abs(J11)**2 + np.abs(J22)**2 - np.abs(J12)**2 - np.abs(J21)**2 )
    m23=np.real(np.conj(J11)*J12 - np.conj(J21)*J22)
    m24=np.imag(-np.conj(J11)*J12 + np.conj(J21)*J22)
    
    m33=np.real(np.conj(J11)*J22 + np.conj(J12)*J21) 
    m34=np.imag(-np.conj(J11)*J22 + np.conj(J12)*J21)
    m44=np.real(np.conj(J11)*J22 - np.conj(J12)*J21)     
   
    return m11,m12,m13,m14,m22,m23,m24,m33,m34,m44

def matrix_4(J11,J12,J21,J22): #rpp,rps,rsp,rss
    
    m11=0.5*(np.abs(J11)**2 + np.abs(J22)**2 + np.abs(J12)**2 + np.abs(J21)**2 )
    m12=0.5*(np.abs(J11)**2 - np.abs(J22)**2 - np.abs(J12)**2 + np.abs(J21)**2 )
    
    m33=np.real(np.conj(J11)*J22 + np.conj(J12)*J21) 
    m34=np.imag(-np.conj(J11)*J22 + np.conj(J12)*J21)     
   
    return m11,m12,m33,m34


def MM9_to_MM16(MM9):  #MM9:  9 x  wavelength
    MM16=np.ones((16,MM9.shape[1])) 

    MM16[1]=MM9[0]
    MM16[2]=MM9[1]
    MM16[3]=MM9[2]

    MM16[4]=MM9[0]
    MM16[5]=MM9[3]
    MM16[6]=MM9[4]
    MM16[7]=MM9[5]

    MM16[8]=-MM9[1]
    MM16[9]=-MM9[4]
    MM16[10]=MM9[6]
    MM16[11]=MM9[7]

    MM16[12]=MM9[2]
    MM16[13]=MM9[5]
    MM16[14]=-MM9[7]
    MM16[15]=MM9[8]  
    return MM16 #MM16:  16 x wavelength

