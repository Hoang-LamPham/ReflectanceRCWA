    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 00:30:46 2023

@author: pham
"""
import numpy as np
from .Tools import inv_diag4,mul_diag44

class Redheffer:    
        
    def __init__(self):
        pass
    
    def Global(self,sA, sB): # general case
    
         # S=sA ⨂ sB   
         # top down: sG=sG⨂ sL: sA=s_Global, sB=s_Layer        
         # s_Global: 4 component , s_Layer: 2 components, S_ref or S_trn: 4 components       
          
         sA11, sA12,sA21, sA22 = sA  
         
         if len(sB)==2:
             sB11,sB12=sB;      sB21=sB12;    sB22=sB11
         else:
             sB11,sB12,sB21, sB22=sB   
             
         d_mat = np.linalg.inv(self.eye2 - sB11 @ sA22)
         f_mat = np.linalg.inv(self.eye2 - sA22 @ sB11)     
     
         s11 = sA11 + sA12 @ d_mat@ sB11@ sA21  
         s12 = sA12 @ d_mat @ sB12
         s21 = sB21 @ f_mat @ sA21              
         s22 = sB22 + sB21 @ f_mat @ sA22 @ sB12       
     
         return s11,s12,s21,s22  
     
    def Global_bottom_up(self,sA, sB,Sim='1/4'):
        # S=sA ⨂ sB   
        # bottop up: sG=sL⨂ sG: sA=s_Layer, sB=s_Global
        # s_Layer: 2 components
        #s_ref or s_trn: 4 components
        # s_Global can be 1

        # Sim='1/4': S11 for reflectance
        # Sim=1/2: S12, S21 for reflectance and transmission 
        # Sim=1/1: S11,S12,S21,S22 # we do not need it in EM   
      
        if len(sA)==2:
            sA11,sA12 = sA;    sA21 = sA12    ;sA22 = sA11;
        else:
            sA11,sA12,sA21,sA22 = sA;   

        sB11=sB[0]     

        if Sim=='1/4':
            s11 = sA11 + sA12 @ np.linalg.solve(self.eye2 - sB11 @ sA22,sB11)@ sA21
            return [s11]   

        elif Sim=='2/4':       
            sB21=sB[1];            
            s11 = sA11 + sA12 @ np.linalg.solve(self.eye2 - sB11 @ sA22,sB11)@ sA21             
            s21 = sB21 @ np.linalg.solve(self.eye2 - sA22 @ sB11, sA21)           
            return s11,s21
        elif Sim=='4/4':   
            sB12=sB[1];    sB21=sB[2];     sB22=sB[3] 
            
            d_mat = np.linalg.inv(self.eye2 - sB11 @ sA22)
            f_mat = np.linalg.inv(self.eye2 - sA22 @ sB11)  
            
            s11 = sA11 + sA12 @ d_mat@ sB11@ sA21        
            s12 = sA12 @ d_mat @ sB12
            s21 = sB21 @ f_mat @ sA21              
            s22 = sB22 + sB21 @ f_mat @ sA22 @ sB12   
        
            return s11,s12,s21,s22
        
    def Global_bottom_up_plane(self,sA, sB,Sim='1/4'):
         # S=sA ⨂ sB   
         # bottop up: sG=sL⨂ sG: sA=s_Layer, sB=s_Global

         # Sim='quarter': S11 for reflectance
         # Sim=half: S12, S21 for reflectance and transmission 
         # Sim=full: S11,S12,S21,S22 # we do not need it in EM       

         if len(sA)==2:
             sA11,sA12 = sA;    sA21 = sA12    ;sA22 = sA11;
         else:
             sA11,sA12,sA21,sA22 = sA   
     
         sB11=sB[0]  
     
         if Sim=='1/4':                
             d_mat=inv_diag4(self.unit - mul_diag44(sB11 , sA22))
             s11 = sA11 + mul_diag44(mul_diag44(sA12, d_mat),mul_diag44(sB11, sA21) ) 
             return [s11]     
     
         elif Sim=='2/4':               
             d_mat=inv_diag4(self.unit - mul_diag44(sB11 , sA22))
             s11 = sA11 + mul_diag44(mul_diag44(sA12, d_mat),mul_diag44(sB11, sA21) ) 
             
             sB21=sB[1];            
             f_mat=inv_diag4(self.unit - mul_diag44(sA22 , sB11)) 
             s21 = mul_diag44(mul_diag44(sB21, f_mat),sA21)  
             
             return s11,s21
        
         elif Sim=='4/4':   
             sB12=sB[1];      sB21=sB[2];     sB22=sB[3] 
             
             d_mat=inv_diag4(self.unit - mul_diag44(sB11 , sA22))  
             f_mat=inv_diag4(self.unit - mul_diag44(sA22 , sB11))           
             
             s11 = sA11 + mul_diag44(mul_diag44(sA12, d_mat),mul_diag44(sB11, sA21) )
             s12= mul_diag44(mul_diag44(sA12, d_mat),sB12)   
             s21 = mul_diag44(mul_diag44(sB21, f_mat),sA21)              
             
             s22 = sB22 + mul_diag44(mul_diag44(sB21, f_mat),mul_diag44(sA22, sB12) )  
             #s22 = sB22 + sB21 @ f_mat @ sA22 @ sB12  
             return s11,s12,s21,s22  
         
         # top_down plane