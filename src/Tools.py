        #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 12:57:41 2023

@author: pham
"""
import numpy as np

def Fill_material(mask,e_base,e_grt):         
    return e_grt*mask   + e_base*(1-mask)   

################# Functions for RCWA ############################################################
def Polarization_vector(AOI_r,Azimuth_r,pol):   
    # from the angle of incidence, azimuth, polarization 
    Px=np.cos(pol)*np.cos(AOI_r)*np.cos(Azimuth_r) - np.sin(pol)*np.sin(Azimuth_r)
    Py=np.cos(pol)*np.cos(AOI_r)*np.sin(Azimuth_r) + np.sin(pol)*np.cos(Azimuth_r)    
    Pz=-np.cos(pol)*np.sin(AOI_r)    
    return np.stack([Px,Py,Pz])


############## Functions for RCWA: homegeneous layers in multilayers ################
def inv_diag4(A):
    '''
    Input
        A: 2D or 3D matrix, which is composed of 4 diagonal matrices
    Return: fast inverse of A
    '''        
    if A.ndim==2:  #4 x Nharm  
        term=A[0]*A[3] -A[1]*A[2]
        
        idx=np.where(term == 0)[0]    
        term[idx]=1e-10  # avoid division zeros
        
        res=np.stack([A[3], -A[1], -A[2],A[0]] )/term
        
        
    elif A.ndim==3: #wl x 4 x Nharm
        term=A[:,0]*A[:,3] -A[:,1]*A[:,2]
        
        res=np.stack([A[:,3], -A[:,1], -A[:,2],A[:,0]] )/term
        res=res.swapaxes(0,1)
    return res


def mul_diag44(A,B):
    '''
    Input
        A,B: 2D or 3D matrix, which is composed of 4 diagonal matrices
    Return: fast matmul(A,B)
    '''      
    
    if A.ndim==2 and B.ndim==2: #4 x Nharm
        res=np.stack([A[0]*B[0] +A[1]*B[2], A[0]*B[1] + A[1]*B[3],A[2]*B[0] +A[3]*B[2], A[2]*B[1] +A[3]*B[3] ]) 
        
    elif A.ndim==3 and B.ndim==3: # wl x 4 x Nharm
        res=np.stack([A[:,0]*B[:,0] +A[:,1]*B[:,2], A[:,0]*B[:,1] + A[:,1]*B[:,3],A[:,2]*B[:,0] +A[:,3]*B[:,2], A[:,2]*B[:,1] +A[:,3]*B[:,3] ])
        res=res.swapaxes(0,1) # wl x 4 x Nharm  
    elif A.ndim==3 and B.ndim==2:    
        res=np.stack([A[:,0]*B[0] +A[:,1]*B[2], A[:,0]*B[1] + A[:,1]*B[3],A[:,2]*B[0] +A[:,3]*B[2], A[:,2]*B[1] +A[:,3]*B[3] ])
        res=res.swapaxes(0,1) # wl x 4 x Nharm 
    elif A.ndim==2 and B.ndim==3: # wl x 4 x Nharm
        res=np.stack([A[0]*B[:,0] +A[1]*B[:,2], A[0]*B[:,1] + A[1]*B[:,3],A[2]*B[:,0] +A[3]*B[:,2], A[2]*B[:,1] +A[3]*B[:,3] ])
        res=res.swapaxes(0,1) # wl x 4 x Nharm  
    return res    
    

def diag2m(a):
    if  a.ndim==2:##(m,n)->(m,n,n) return 3D array with diagonal matrix
        a_m=np.einsum('ij,jk->ijk', a, np.eye(a.shape[1], dtype=a.dtype,device=a.device))
    else: a_m=np.diag(a)    #n ->(n,n)
    return a_m
   

def vec2m(A):
    # (4,n)->(2n,2n) or (m,4,n)->(m,2n,2n)      
   
    if A.ndim==2:# 4 x Nharm
        A02=np.vstack([np.diag(A[0]),np.diag(A[2])])
        A13=np.vstack([np.diag(A[1]),np.diag(A[3])])
        A=np.hstack([A02,A13])
   
    elif A.ndim==3:   
        A02=np.cat([diag2m(A[:,0]),diag2m(A[:,2])],axis=-2)
        A13=np.cat([diag2m(A[:,1]),diag2m(A[:,3])],axis=-2)
        A=np.cat([A02,A13],axis=-1)           
    return A  
   
