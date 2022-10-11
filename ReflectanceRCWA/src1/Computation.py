# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 15:20:51 2021

@author: Hoang
"""
import numpy as np

 
def Optical_Response(Polarization,S_global_sub,Nharm):   
    #optical responses: reflectance or transmittance, 
    #depending on S_global_sub: S_global_11 for Reflectance and  S_global_21 for Transmittance   
    delta_vec = np.zeros(Nharm)
    delta_vec[int(np.floor(Nharm/2))] = 1       
           
    #Compute Source Field
    E_inc = np.zeros(2*Nharm, dtype=complex)    
    E_inc[0:Nharm] = Polarization[0]*delta_vec
    E_inc[Nharm:2*Nharm] = Polarization[1]*delta_vec  
    
    #Step 12: Compute Reflected  Fields 
    #Compute mode coefficients of the source    
    c_inc=E_inc.reshape(2*Nharm,1)    
       
    #Compute Compute reflected and transmitted fields
    E_ref = S_global_sub@ c_inc # x, y directions: E_ref_xy
    rx=E_ref[0:Nharm,:]
    ry=E_ref[Nharm:2*Nharm,:]  
    
    return rx,ry 

