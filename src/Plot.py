#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 19:54:13 2024

@author: pham
"""

import numpy as np
from .Tools import Fill_material
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import matplotlib.colors

color=['blue','gray','red','green','purple','pink','brown','gold','orange']
cmap_uf = matplotlib.colors.ListedColormap( color)
cmap_uf='brg'

class Plot:     
    
######################## Plot ######################################################    
    def mask_from_layer(self,layer):    
        mask=layer['pattern']['mask']    
        mask=np.round(mask,decimals=0) # binary mask for plot           
        return mask 
    
    def mask_for_plot(self,Structure):
        
        # step 1: retrieve binary mask from all layers of Structure
        mask_binary=np.stack([self.mask_from_layer(layer) for layer in Structure])
        
        #step 2: retrieve all eps_name , set unique to remove duplicates 
        #and set each  unique eps_name a value for plot        
        
        eps_name_all=[]# retrieve all eps_name from layers
        for layer in Structure: 
            if layer['pattern']['eps_name']==False:# no pattern, or homogeneous layer
                eps_name_all.append([layer['medium']['eps_name']])  
            else:
                eps_name_all.append([layer['medium']['eps_name'],layer['pattern']['eps_name']])            
        self.eps_name_all=eps_name_all
        
        eps_name_flatten=[i for sub_name in eps_name_all for i in sub_name]               
        self.eps_name=list(set(eps_name_flatten)) # remove duplicates
        
        self.values=(np.arange(len(self.eps_name))+1)*2  # set a value of each eps_name_unique
        
        # step 3: set value_of_eps for plot
        value_of_eps=[]
        for sub_name in eps_name_all:    
            temp=[]
            for name in sub_name:
                idx=self.eps_name.index(name)
                temp.append(self.values[idx])
            value_of_eps.append(temp)
                
        # step 4: Fill binary mask with value_of_eps for plot
        Mask_plot=[]
        for lth in range(len(mask_binary)):
            eps_medium=value_of_eps[lth][0]
            if len(value_of_eps[lth])==2:   # pattern layer     
                eps_pattern=value_of_eps[lth][1]    
                mask_lth=Fill_material(mask_binary[lth],eps_medium,eps_pattern)
            else:                
                mask_lth=eps_medium*mask_binary[lth]
            Mask_plot.append(mask_lth)
        
        #self.Mask_plot=np.array(Mask_plot)
        return np.array(Mask_plot)
        
    
    def XY(self,Structure,layer_position=[0],origin='lower',figure_size=(8,4),color=[]):  
        
        # custom cmap from color
        if color:cmap = ListedColormap(color)
        else: cmap=cmap_uf
                       
            
        Mask_plot= self.mask_for_plot(Structure)  
        
        v_min=np.min(self.values)
        v_max=np.max(self.values)
                
        if Mask_plot.shape[-1]==1: # 2D structure
            print("XY view is not available for 2D structures")   
        
        else:
            if len(layer_position) >1:
                fig, ax = plt.subplots(nrows=1, ncols=len(layer_position),figsize=figure_size)
                for i in range(len(layer_position)):
                    lth=layer_position[i]
                    im=ax[i].imshow(Mask_plot[lth].T,vmin=v_min,vmax=v_max,origin=origin,cmap=cmap)
                    ax[i].title.set_text('Layer '+str(layer_position[i]))
                    if i==0:
                        ax[i].set_xlabel('x (pixel)')
                        ax[i].set_ylabel('y (pixel)')
                    else:
                        ax[i].set_xticks([]);       ax[i].set_yticks([]) 
            else:
                fig, ax = plt.subplots(1, 1,figsize=figure_size)
        
                lth=layer_position[0]
                im=ax.imshow(Mask_plot[lth].T,vmin=v_min, vmax=v_max,origin=origin,cmap=cmap)
                ax.title.set_text('Layer '+str(lth))
            
                ax.set_xlabel('x (pixel)')
                ax.set_ylabel('y (pixel)')  
                
            
            fig.suptitle('Structure in XY view (1 pixel = {} nm)'.format(self.reso),fontsize=12,y=-0.02) 
   
            colors = [ im.cmap(im.norm(value)) for value in self.values]
            patches = [ mpatches.Patch(color=colors[i], label= self.eps_name[i]) for i in range(len(self.values)) ]
            plt.legend(handles=patches, bbox_to_anchor=(1.1, 0.95), loc=2,  borderaxespad=0.5)
            

    def XZ(self,Structure,figure_size=(8,4),color=[]):
        
        # custom cmap from color
        if color:cmap = ListedColormap(color)
        else: cmap=cmap_uf
                
        Mask_plot= self.mask_for_plot(Structure)  
        
        v_min=np.min(self.values)
        v_max=np.max(self.values)
        
        Thickness=[]
        for layer in Structure:
            Thickness.append(layer['medium']['h'])
        
        if Mask_plot.shape[-1]==1: #2D
            Mask_xz=Mask_plot[:,:,0]
            Mask_xz_plot=broad_height(Mask_xz,Thickness,self.reso)         
            
            fig, ax = plt.subplots(nrows=1, ncols=1,figsize=figure_size)
           
            im=ax.imshow(Mask_xz_plot.real,vmin=v_min, vmax=v_max,cmap=cmap)   
            
            ax.set_xlabel('x (pixel)')
            ax.set_ylabel('z (pixel)')
            
            colors = [ im.cmap(im.norm(value)) for value in self.values]
            patches = [ mpatches.Patch(color=colors[i], label= self.eps_name[i]) for i in range(len(self.values)) ]
            plt.legend(handles=patches, bbox_to_anchor=(1.1, 0.95), loc=2,  borderaxespad=0.5)
            
            fig.suptitle('Nanostructure in XZ view (1 pixel = {} nm)'.format(self.reso),fontsize=12,y=-0.02)

            plt.show()  
        
        else: #3D
            Nx=Mask_plot.shape[1]//2
            Ny=Mask_plot.shape[2]//2

            Mask_xz=Mask_plot[:,:,Ny]
            Mask_xz_plot=broad_height(Mask_xz,Thickness,self.reso)
            
            fig, ax = plt.subplots(nrows=1, ncols=1,figsize=figure_size)   
            im=ax.imshow(Mask_xz_plot,vmin=v_min, vmax=v_max,cmap=cmap)    
            ax.set_xlabel('x (pixel)')
            ax.set_ylabel('z (pixel)')
            fig.suptitle('Structure in XZ view (1 pixel = {} nm)'.format(self.reso),fontsize=12,y=-0.02)
      
            
            colors = [ im.cmap(im.norm(value)) for value in self.values]
            patches = [ mpatches.Patch(color=colors[i], label= self.eps_name[i]) for i in range(len(self.values)) ]
             
            plt.legend(handles=patches, bbox_to_anchor=(1.1, 0.95), loc=2,  borderaxespad=0.5)
           
            plt.show()     
    
    def YZ(self,Structure,figure_size=(8,4),color=[]):
        
        # custom cmap from color
        if color:cmap = ListedColormap(color)
        else: cmap=cmap_uf
                
        Mask_plot= self.mask_for_plot(Structure)  
        
        v_min=np.min(self.values)
        v_max=np.max(self.values)
        
        Thickness=[]
        for layer in Structure:
            Thickness.append(layer['medium']['h'])
        
        if Mask_plot.shape[-1]==1: #2D
            print("YZ view is not available for 2D structures")
                    
        else: #3D
            Nx=Mask_plot.shape[1]//2
            Ny=Mask_plot.shape[2]//2            
                
            Mask_yz=Mask_plot[:,Nx,:]
            Mask_yz_plot=broad_height(Mask_yz,Thickness,self.reso)           
            
            
            fig, ax = plt.subplots(nrows=1, ncols=1,figsize=figure_size)   
            im=ax.imshow(Mask_yz_plot,vmin=v_min, vmax=v_max,cmap=cmap )   
            ax.set_xlabel('y (pixel)')
            ax.set_ylabel('z (pixel)')
            fig.suptitle('Structure in YZ view (1 pixel = {} nm)'.format(self.reso),fontsize=12,y=-0.02)
            
            
            colors = [ im.cmap(im.norm(value)) for value in self.values]
            patches = [ mpatches.Patch(color=colors[i], label= self.eps_name[i]) for i in range(len(self.values)) ]
             
            plt.legend(handles=patches, bbox_to_anchor=(1.1, 0.95), loc=2,  borderaxespad=0.5)
           
            plt.show() 
    
        
        
def broad_height(geo_t,Thickness_Sim,reso):        
    geo_plot=np.broadcast_to(geo_t[0],(int(Thickness_Sim[0]/reso),)+geo_t[0].shape) 
    if len(geo_t)>1:
        for lth in range(len(geo_t))[1:]:    
            geo_plot_lth=np.broadcast_to(geo_t[lth],(int(Thickness_Sim[lth]/reso),)+geo_t[lth].shape) # transform 1D array to 2D array to plot
            geo_plot=np.vstack((geo_plot,geo_plot_lth)) 
    return geo_plot    

    
    
    
        
    