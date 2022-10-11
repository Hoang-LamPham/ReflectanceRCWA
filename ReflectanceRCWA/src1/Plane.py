# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 23:01:04 2021

@author: Hoang
"""
import numpy as np

def diag4_inv(A):
    # To inverse matrix composed of 4 diagonal matrices 
    return np.array([A[3], -A[1], -A[2],A[0]] )/(A[0]*A[3] -A[1]*A[2])

def diag44_AB(A,B):
    #To multiply 2 matrices composed of 4 diagonal matrices
    return np.array([A[0]*B[0] +A[1]*B[2], A[0]*B[1] + A[1]*B[3],A[2]*B[0] +A[3]*B[2], A[2]*B[1] +A[3]*B[3] ]) 

def trn_matrix(Vt,Vg,Nharm,Sim='quarter'):
    # Sim='quarter': S11
    # Sim=half: S12, S21
    # Sim=full: S11,S12,S21,S22    
    lam1=np.array([np.ones(Nharm),np.zeros(Nharm),np.zeros(Nharm),np.ones(Nharm)])
    lam2=diag44_AB(diag4_inv(Vg),Vt) #lam2=solve(Vg,Vt)
        
    At = lam1 +lam2     ;Bt = lam1 - lam2
    term=diag4_inv(At)
        
    S_trn_11 = diag44_AB(Bt,term)              #S_trn_11 = Bt@ np.linalg.inv(At)
    if Sim=='quarter':
        return S_trn_11                        
    elif Sim=='half':
        S_trn_21=2*term                        #S_trn_21 = 2*np.linalg.inv(At)
        return S_trn_11, S_trn_21
    elif Sim=='full':
        S_trn_12=0.5*(At-diag44_AB(Bt,diag44_AB(term,Bt))      )
        S_trn_21=2*term
        S_trn_22=-diag44_AB(term,Bt)           #S_trn_22 = -np.linalg.inv(At)@ Bt
        return S_trn_11,S_trn_12,S_trn_21,S_trn_22
         

def ref_matrix(Vr,Vg,Nharm, Sim='quarter'):
    # Sim='quarter': S11
    # Sim=half: S12, S21
    # Sim=full: S11,S12,S21,S22  
    lam1=np.array([np.ones(Nharm),np.zeros(Nharm),np.zeros(Nharm),np.ones(Nharm)])
    lam2=diag44_AB(diag4_inv(Vg),Vr) #lam2=solve(Vg,Vr)
	
    At = lam1 +lam2     ;Bt = lam1 - lam2
    term=diag4_inv(At)
    S_ref_11 = -diag44_AB(term,Bt)              #S_trn_11 = np.linalg.inv(At)@Bt
    if Sim=='quarter':
        return S_ref_11
    elif Sim=='half':    
        S_ref_21=0.5*(At+diag44_AB(Bt,S_ref_11))
        return S_ref_11, S_ref_21
    elif Sim=='full':
        S_ref_12=2*term        
        S_ref_21=0.5*(At+diag44_AB(Bt,S_ref_11))
        S_ref_22=diag44_AB(Bt,term)
        return S_ref_11, S_ref_12, S_ref_21,S_ref_22
        

def S_matrix_layer_plane(V_i, Vg,x,Nharm):
    
    I=np.array([np.ones(Nharm),np.zeros(Nharm),np.zeros(Nharm),np.ones(Nharm)])    
    
    A = I + diag44_AB(diag4_inv(V_i), Vg)
    B = 2*I - A  
            
    term=x*diag44_AB(B,diag4_inv(A))
            
    S11=diag44_AB(diag4_inv(A -diag44_AB(term,x*B)),diag44_AB(term,x*A)-B)
    S12=diag44_AB(diag4_inv(A -diag44_AB(term,x*B)),x*A-diag44_AB(term,B))     
    
    return S11,S12

def redheffer_global_plane(sL, sG,Nharm,Sim='quarter'):#both Sl and sG are homogeneous
    # using for bottom-up: sG=sL ⨂ sG
    unit_mat=np.array([np.ones(Nharm),np.zeros(Nharm),np.zeros(Nharm),np.ones(Nharm)])
    
    sL_11 = sL[0]    ;sL_12 = sL[1]; 
    sG_11=sG[0]        
    
    d_mat = diag4_inv(unit_mat - diag44_AB(sG_11 , sL_11))    
    
    s_11 = sL_11 + diag44_AB(diag44_AB(sL_12, d_mat),diag44_AB(sG_11, sL_12) )  
    if Sim=='quarter':
        return s_11 
    elif Sim=='half':
        sG_21=sG[1]
        f_mat=diag4_inv(unit_mat - diag44_AB(sL_11 , sG_11)) 
        s_21 = diag44_AB(diag44_AB(sG_21, f_mat),sL_12) 
        return s_11,s_21