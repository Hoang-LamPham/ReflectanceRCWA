# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 15:31:47 2021

@author: Hoang
"""
import numpy as np


def Circle(Nx,Ny, layer): # layer= [[x_center, y_center,radius,'material','shape'],h]   
    y = np.arange(0, Ny)
    x = np.arange(0, Nx)    
    
    x_center=layer[0][0];    y_center=layer[0][1]        
    radius=int(layer[0][2])                  
        
    mask = (y[np.newaxis,:]-x_center)**2 + (x[:,np.newaxis]-y_center)**2 < radius**2  
    return mask 

def Rectangle(Nx,Ny,layer): # layer= [[x_center, y_center, width_x, width_y,'material','shape'],h] 
    
    x_center=layer[0][0];    y_center=layer[0][1]
    widthx=layer[0][2];       widthy=layer[0][3]        
       
    x_start=int(x_center-widthx/2);    x_end=int(x_start+widthx)
   
    y_start=int(y_center-widthy/2);    y_end=int(y_start+widthy)  
    
   
    mask=np.zeros((Nx,Ny),dtype=bool)
    mask[x_start:x_end,y_start:y_end ]=True    
    return mask 

def Split_layer(TCD,BCD,N_split=10): # used for lamellar layers
    # TCD,BCD: top and bottom critical dimension
    d=(BCD-TCD)/N_split
    CD_range=[BCD-d*(N_split-i) for i in range(N_split)]
    return CD_range