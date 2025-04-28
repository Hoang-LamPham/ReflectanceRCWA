#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 16 11:31:16 2023

@author: pham
"""

import numpy as np


def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Geometry:    
       
    cm='brg' # cmap 
     
    def nm_cell(self,*arg): # convert values in nm to cell    
        if len(arg)==1: return [a_ /self.reso for a_ in arg][0]      
        else: return [a_ /self.reso for a_ in arg]                
        
    def Grid(self):           
        self.reso=self.Lx/self.Nx    
        
        x = np.arange(self.Nx)+0.5 
        y = np.arange(self.Ny)+0.5
        self.x_grid, self.y_grid = np.meshgrid(x,y,indexing='ij')       
      
        
    def layer(self):    
        
        _layer={}
        _layer['medium']={'eps':False,'h':False,'eps_name':False}
        _layer['pattern']={'eps':False,'mask':self.mask_medium,'eps_name':False}
        return _layer
    
    def Rectangle(self,Wx=0,Wy=0,Cx='center',Cy='center',theta=0,box=False,binary=False):
        # Wx,Wy: x-width, y-width; Cx,Cy: x-center, y-center in nm
        # theta: angle at center    
        # box:   list: [x_st,y_st,x_end,y_end]
        if Cx=='center':
            Wx,Wy=self.nm_cell(Wx,Wy)
            if Wy==0: Wy=1 # 2D structures  
            Cx,Cy=self.Nx/2,self.Ny/2
        else:
            Wx,Wy,Cx,Cy=self.nm_cell(Wx,Wy,Cx,Cy)  
                
        if box: 
            x_st,y_st,x_end,y_end=box          
            
            x_grid=self.x_grid[x_st:x_end,y_st:y_end]
            y_grid=self.y_grid[x_st:x_end,y_st:y_end]
        else:
            x_grid=self.x_grid
            y_grid=self.y_grid
                
        level = 1. - (np.maximum(np.abs(((x_grid-Cx)*np.cos(theta)+(y_grid-Cy)*np.sin(theta))/(Wx/2.)),
                                 np.abs((-(x_grid-Cx)*np.sin(theta)+(y_grid-Cy)*np.cos(theta))/(Wy/2.))))
        
        mask=Sigmoid(self.edge_sharpness*level)
        if binary: mask=np.round(mask,decimals=0)        
        return mask   

    def Circle(self,D=0,Cx='center',Cy='center',box=False,binary=False):
        # D:diameter; Cx,Cy: x-center, y-center in nm     
        # box:   list: [x_st,y_st,x_end,y_end]
        if Cx=='center':
            R=self.nm_cell(D/2)
            Cx,Cy=self.Nx/2,self.Ny/2
        else:
            R,Cx,Cy=self.nm_cell(D/2,Cx,Cy)   
        if box: 
            x_st,y_st,x_end,y_end=box          
            
            x_grid=self.x_grid[x_st:x_end,y_st:y_end]
            y_grid=self.y_grid[x_st:x_end,y_st:y_end]
        else:
            x_grid=self.x_grid
            y_grid=self.y_grid
                
        level = 1. - np.sqrt(((x_grid-Cx)/R)**2 + ((y_grid-Cy)/R)**2)
        
        mask=Sigmoid(self.edge_sharpness*level)
        if binary: mask=np.round(mask,decimals=0)        
        return mask      
    
   
     
    