# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 15:10:35 2021

@author: Hoang
"""
import numpy as np
from numpy.linalg import solve  # 

#########compute layer scattering matrix######################################
def homogeneous(Kx, Ky, e_r, m_r = 1):            
    
    arg = (np.conj(m_r)*np.conj(e_r)-Kx**2-Ky**2);    
    arg = arg.astype('complex');
    Kz = np.conj(np.sqrt(arg)); 
           
    eigen_v=1j*Kz     
    
    #V=Q/eigen_v    
    V=np.array([Kx*Ky/eigen_v,(e_r-Kx**2)/eigen_v,(Ky**2-e_r)/eigen_v,-Kx*Ky/eigen_v])
    
    return V,Kz# array of only diagonal


def PQ_matrix(Kx, Ky, e_conv, mu_conv):
    
    P11=Kx @ solve(e_conv,Ky)
    P12=mu_conv- Kx @ solve(e_conv,Kx)
    P21=Ky @ solve(e_conv,Ky) - mu_conv
    P22=-Ky@ solve(e_conv,Kx)
    
    P=np.block([[P11,P12],[P21,P22]])
    
    Q11=Kx*Ky
    Q12=e_conv - Kx*Kx
    Q21=Ky*Ky - e_conv
    Q22=-Ky*Kx
    
    Q=np.block([[Q11,Q12],[Q21,Q22]])  
    return P,Q 

def S_matrix_layer(V_i, W_i,Vg,X):
     A = np.linalg.inv(W_i) + np.linalg.inv(V_i)@ Vg
     B = np.linalg.inv(W_i) - np.linalg.inv(V_i)@ Vg                 
            
     term=X@B@solve(A,X)
     S11=solve(A-term@B, term@A -B)
     S12=solve(A-term@B, X)@(A-B@solve(A,B))
     return S11,S12
 ##############################################################################
 

########### global scattering matrix###########  

def redheffer_global(sL, sG,Nharm,Sim='quarter'):# sG: global; sL:layer 
    # using for bottom-up: sG=sL ⨂ sG
    # Sim='quarter': S11
    # Sim=half: S12, S21
    # Sim=full: S11,S12,S21,S22 
    unit_mat = np.identity(2*Nharm);
   
    sL11 = sL[0]    ;sL12 = sL[1];    sL21 = sL[2]    ;sL22 = sL[3];
    
    sG11=sG[0]
    
    d_mat = np.linalg.inv(unit_mat - sG11 @ sL22)
    
    s11 = sL11 + sL12 @ d_mat@ sG11@ sL21  
   
   
    if Sim=='quarter':
        return s11 
    
    elif Sim=='half':
        sG21=sG[1]
        f_mat=np.linalg.inv(unit_mat - sL22 @ sG11)      
        s21=sG21@f_mat@sL21
       
        return s11,s21
    
    elif Sim=='full':   
        
        sG12=sG[1];      sG21=sG[2] ;    sG22=sG[3]    
      
        d_mat = np.linalg.inv(unit_mat - sG11 @ sL22)
        f_mat = np.linalg.inv(unit_mat - sL22 @ sG11)    
   
        s11 = sL11 + sL12@ d_mat@ sG11@ sL21  
        s12 = sL12 @ d_mat @ sG12
        s21 = sG21 @ f_mat @ sL21              
        s22 = sG22 + sG21 @ f_mat @ sL22 @ sG12    
        return s11,s12,s21,s22  