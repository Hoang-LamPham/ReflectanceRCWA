# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 22:02:20 2022

@author: Hoang
"""

import numpy as np
from .Geometry import Geometry
from .Redheffer import Redheffer
from .Tools import Polarization_vector
from .Tools import inv_diag4, mul_diag44,diag2m, vec2m
from .Tools import Fill_material
from .Plot import Plot

def matrix_block(a11,a12,a21,a22): # [a11 a12
                                  #  a21 a22]     
    a=np.block([[a11,a12],[a21,a22]])
    return a  

def Homogeneous(Kx,Ky,e_r,m_r=1):    
    arg = np.conj(m_r)*np.conj(e_r)-Kx**2-Ky**2
        
    arg = arg.astype('complex');            
    Kz = np.conj(np.sqrt(arg));          
                 
    eigen_v=1j*Kz  
    V=np.stack([Kx*Ky/eigen_v,(e_r-Kx**2)/eigen_v,(Ky**2-e_r)/eigen_v,-Kx* Ky/eigen_v])
        
    return V, Kz# array of only diagonal 

class rcwa:    
        
    def __init__(self):
        self.Geometry=Geometry()     
        self.Redheffer=Redheffer()          
        self.Plot=Plot()        
           
    def force_array_wl_form(self,*arg): # force other paramters to form of wl array        
        zero=np.zeros_like(self.wavelength_range)        
        res=[zero+a for a in arg ]        
        return res   
            
    def setup(self,wavelength_range=[500],
              LxLy=[500,500],              # periodic length [Lx,Ly]
              NxNy=[500,500],              # simulation domain [Nx,Ny]
              order=[3,3],                 # diffraction order [mx,my]
              angle=[],                    # [AOI_d,Azimuth_d] in degree             
              e_ref=1,                     # reflected medium
              e_trn=1,                     # transmitted medium
              source='TM',                 # TM - x polarize, or TE- y polarized state 
              edge_sharpness=500,           # use in Geometry               
               ):

        # extract from function        
        self.mx,self.my=order
        Lx,Ly=LxLy  
        Nx,Ny=NxNy
                 
        self.Nharm=(2*self.mx+1)*(2*self.my+1)        
        self.wavelength_range=np.array(wavelength_range)+1e-5 
        
        if angle:        
            self.AOI_r,self.Azimuth_r=np.radians(angle)
                      
            if source=='TM':self.SourcePlaneWave( mode='TM')  # source depends on AOI,azimuth
            if source=='TE':self.SourcePlaneWave( mode='TE')

        # force e_ref,e_trn to have form of wl
        self.e_ref,self.e_trn= self.force_array_wl_form(e_ref,e_trn)            
        
        # Geometry          
        self.Geometry.Lx,self.Geometry.Ly=Lx,Ly
        self.Geometry.Nx,self.Geometry.Ny=Nx,Ny  
        self.Geometry.edge_sharpness=edge_sharpness 
        self.Geometry.mask_medium=np.ones((self.Geometry.Nx, self.Geometry.Ny))   
        self.Geometry.Grid()  # meshgrid,reso  
        
        #Plot
        self.Plot.reso=self.Geometry.reso
        self.Plot.mask_medium=self.Geometry.mask_medium
        
        #      
        self.eye=np.eye(self.Nharm)
        self.eye2=np.eye(2*self.Nharm) 
        self.zero2=np.zeros((2*self.Nharm,2*self.Nharm))
        
        self.ones=np.ones(self.Nharm)        
        zeros=np.zeros(self.Nharm)        
        self.unit=np.stack([self.ones,zeros,zeros,self.ones])       
        
        # Redheffer        
        self.Redheffer.unit=self.unit
        self.Redheffer.eye2=self.eye2         
            
       
    def wave_vector(self,idx=0,angle=[]):    # u_ref=1       
        
        if angle:  
            if idx==0:
                self.AOI_r,self.Azimuth_r=np.radians(angle)
        
    # Kx,Ky,kz_inc
        self.k0=2*np.pi/self.wavelength_range[idx]
        n_i =  np.sqrt(self.e_ref[idx])        
        kx_inc = n_i * np.sin(self.AOI_r) * np.cos(self.Azimuth_r);
        ky_inc = n_i * np.sin(self.AOI_r) * np.sin(self.Azimuth_r);      
        self.kz_inc = np.sqrt(n_i**2 - kx_inc ** 2 - ky_inc ** 2);
     
        k_x = kx_inc - 2*np.pi*np.arange(-self.mx, self.mx+1)/(self.k0*self.Geometry.Lx);
        k_y = ky_inc - 2*np.pi*np.arange(-self.my, self.my+1)/(self.k0*self.Geometry.Ly);                 
        Kx, Ky = np.meshgrid(k_x, k_y,indexing='ij');   
             
        self.Kx= Kx.flatten()
        self.Ky=Ky.flatten()
         
        self.Vr,Kzr = Homogeneous(self.Kx,self.Ky,self.e_ref[idx]) #reflection Medium         
        self.Kzr=-Kzr               
         
        self.Vt,Kzt=Homogeneous(self.Kx,self.Ky,self.e_trn[idx]) #transmission Medium          
        self.Kzt=Kzt
        self.Vg,_ = Homogeneous(self.Kx,self.Ky,1);        #gap Medium             
    
        
    def PQ_matrix(self,e_conv):        
        
        e_inv=np.linalg.inv(e_conv)
        term_y=e_inv@diag2m(self.Ky);                term_x=e_inv@diag2m(self.Kx)           
             
        #P matrix        
        P11=diag2m(self.Kx)@term_y
        P12= self.eye -diag2m(self.Kx)@term_x  # mu_conv=eye
        P21=diag2m(self.Ky)@term_y - self.eye
        P22=-diag2m(self.Ky)@term_x    
        
        P=matrix_block(P11,P12,P21,P22)
        
        #Q matrix
        Q11=diag2m(self.Kx*self.Ky)
        Q12=e_conv - diag2m(self.Kx*self.Kx)
        Q21=diag2m(self.Ky*self.Ky) - e_conv
        
        Q=matrix_block(Q11,Q12,Q21,-Q11)
        return P,Q
        
    def S_eigen(self,e_conv):   
        P,Q=self.PQ_matrix(e_conv)   
        Vg_grt=vec2m(self.Vg)
        
        # eigen problem        
        Gamma_squared = P@Q;   
        Lambda,self.W_i = np.linalg.eig(Gamma_squared);            
        self.lambda_matrix = np.sqrt(Lambda);   
        
        self.V_i=Q @ self.W_i @ np.linalg.solve(diag2m(self.lambda_matrix),self.eye2) 
            
        term1=np.linalg.solve(self.W_i,self.eye2)
        term2=np.linalg.solve(self.V_i, Vg_grt)
            
        self.A = term1 + term2
        self.B = term1 - term2   
        
        self.term_AB=self.A-self.B@np.linalg.solve(self.A,self.B)
        self.X= - self.lambda_matrix*self.k0 
        
    def S_eigen_h(self,Li):  
                   
        X=np.exp(self.X*Li)           
        X=diag2m(X) 
        
        term=X@self.B@np.linalg.solve(self.A,X)            
        term_s=np.linalg.solve(self.A-term@self.B,self.eye2)    
        
        S11=term_s@ (term@self.A -self.B)
        S12=term_s@ X@self.term_AB      
        return S11,S12  

    def S_layer_eigen(self,Li,e_conv,need_eigen=True,form='matrix'):     
        if e_conv.ndim==0:            
            S11,S12= self.S_Homogeneous(Li,e_conv,form=form)
        else:
            if need_eigen:
                self.S_eigen(e_conv)            
                S11,S12  =self.S_eigen_h(Li)  
            else:
                S11,S12  =self.S_eigen_h(Li)
        return [S11,S12]   
    
    def S_layer(self,structure,idx=0,angle=[],need_eigen=True,form='matrix'):          
        if idx==0:
            self.ERC_CONV=[self.eps_Fourier(layer) for layer in structure] # expand materials in Fourier space
            self.NL=len(structure)
            self.Thickness=[layer['medium']['h'] for layer in structure]
            
        # Compute scattering matrix
        self.wave_vector(idx=idx,angle=angle) 
        S_layer=[self.S_layer_eigen(self.Thickness[lth],self.ERC_CONV[lth][idx],need_eigen=need_eigen,form=form)for lth in range(self.NL)]  
        return S_layer 
    
         
    def S_System(self,S_layer,S_Ref=True,S_Trn=True):    
        if S_Ref:         
            S_global=self.S_Ref()          
            for lth in range(len(S_layer)):
                S_global=self.Redheffer.Global(S_global, S_layer[lth])    
        else:            
            S_global=[S_layer[0][0], S_layer[0][1],S_layer[0][1],S_layer[0][0]]
            
            for lth in range(1,len(S_layer)):
                S_global=self.Redheffer.Global(S_global, S_layer[lth])            
            
        if S_Trn:
            S_trn=self.S_Trn() 
            S_global=self.Redheffer.Global(S_global, S_trn) 
        return S_global  
    
    def S_System_bottom_up(self,S_layer,nb=0,only_R=False,S_Ref=True,S_Trn=True):    
        # only_R: simumate only reflectance
        # nb: number of homogeneous botto layers          
        if only_R:
            if nb==0:
                if S_Trn: 
                    S_trn=self.S_Trn() 
                    S_global=[S_trn[0]]
                    for lth in range(len(S_layer))[::-1]:
                        S_global=self.Redheffer.Global_bottom_up(S_layer[lth],S_global,Sim='1/4')
                if S_Trn==False:
                    S_global=S_layer[-1]
                    for lth in range(len(S_layer)-1)[::-1]:
                        S_global=self.Redheffer.Global_bottom_up(S_layer[lth],S_global,Sim='1/4')   
            else:
                if S_Trn:         
                    S_trn=self.S_Trn(form='vector')                 
                    S_global=[S_trn[0]] # bottom-up: need only 1 element for R             
                    for lth in range(len(S_layer))[::-1][:nb]:
                        S_global=self.Redheffer.Global_bottom_up_plane(S_layer[lth],S_global,Sim='1/4')  
                           
                if S_Trn==False:    
                    S_global=S_layer[-1] 
                    for lth in range(len(S_layer)-1)[::-1][:nb-1]:                
                        S_global=self.Redheffer.Global_bottom_up_plane(S_layer[lth],S_global,Sim='1/4')   
                
            
                S_global=[vec2m(S_global[0])] # covert vector into matrix to connect to grating layers 
                for lth in range(len(S_layer))[::-1][nb:]:
                    S_global=self.Redheffer.Global_bottom_up(S_layer[lth],S_global,Sim='1/4')               
            
            
            if S_Ref:
                S_ref=self.S_Ref() 
                S_global=self.Redheffer.Global_bottom_up(S_ref, S_global,Sim='1/4')
        
        else:        
            if nb==0:
                if S_Trn: 
                    S_trn=self.S_Trn() 
                    S_global=[S_trn[0],S_trn[2]]
                    for lth in range(len(S_layer))[::-1]:
                        S_global=self.Redheffer.Global_bottom_up(S_layer[lth],S_global,Sim='2/4')
                if S_Trn==False:
                    S_global=S_layer[-1] 
                    for lth in range(len(S_layer)-1)[::-1]:
                        S_global=self.Redheffer.Global_bottom_up(S_layer[lth],S_global,Sim='2/4')                
                    
            else:
                if S_Trn:         
                    S_trn=self.S_Trn(form='vector')  
                
                    S_global=[S_trn[0],S_trn[2]] # bottom-up: need only two elements for R,T             
                    for lth in range(len(S_layer))[::-1][:nb]:
                        S_global=self.Redheffer.Global_bottom_up_plane(S_layer[lth],S_global,Sim='2/4')  
                           
                if S_Trn==False:    
                    S_global=S_layer[-1] 
                    for lth in range(len(S_layer)-1)[::-1][:nb-1]:                
                        S_global=self.Redheffer.Global_bottom_up_plane(S_layer[lth],S_global,Sim='2/4')   
                
            
                S_global=[vec2m(S_global[0]),vec2m(S_global[1])] # covert vector into matrix to connect to grating layers 
                for lth in range(len(S_layer))[::-1][nb:]:
                    S_global=self.Redheffer.Global_bottom_up(S_layer[lth],S_global,Sim='2/4')
            
            
            if S_Ref:
                S_ref=self.S_Ref() 
                S_global=self.Redheffer.Global_bottom_up(S_ref, S_global,Sim='2/4') 
        return S_global    
    
    def S_Homogeneous(self,Li,e_conv,form='matrix'):  
            
        V_i,_=Homogeneous(self.Kx, self.Ky, e_conv)
        Gamma_squared=self.Kx*self.Kx + self.Ky*self.Ky -e_conv
        lambda_matrix=np.lib.scimath.sqrt(Gamma_squared.astype('complex'))
        x= np.exp(-lambda_matrix*self.k0*Li)  
       
        A = self.unit + mul_diag44(inv_diag4(V_i), self.Vg)
        B = 2*self.unit - A  
            
        term=x*mul_diag44(B,inv_diag4(A))
            
        S11=mul_diag44(inv_diag4(A -mul_diag44(term,x*B)),mul_diag44(term,x*A)-B)
        S12=mul_diag44(inv_diag4(A -mul_diag44(term,x*B)),x*A-mul_diag44(term,B))   
        if form=='matrix':
           S11,S12= vec2m(S11), vec2m(S12) 
       
        return S11,S12
        
    def S_Ref(self,form='matrix'):                  
     
        term_Vr=mul_diag44(inv_diag4(self.Vg),self.Vr)    #solve(Vg,Vr)       
        
        Ar = self.unit +term_Vr     ;Br = self.unit - term_Vr
        Ar_inv=inv_diag4(Ar)
        
        S_ref_11 = -mul_diag44(Ar_inv,Br)              #S_ref_11 = - np.linalg.inv(Ar)@Br
        S_ref_12=2*Ar_inv                              # S_ref_21 = 2*np.linalg.inv(Ar)
        S_ref_21=2*self.unit-S_ref_12
        S_ref_22=-S_ref_11 
        
        S_ref=[S_ref_11,S_ref_12,S_ref_21,S_ref_22]
        if form=='matrix':
            S_ref=[vec2m(S_ref[0]),vec2m(S_ref[1]),vec2m(S_ref[2]),vec2m(S_ref[3])]
        
        return S_ref
    
    def S_Trn(self,form='matrix'):  
        
        term_Vt=mul_diag44(inv_diag4(self.Vg),self.Vt)    #solve(Vg,Vt)
        
        At = self.unit +term_Vt     ;Bt = self.unit - term_Vt
        
        At_inv=inv_diag4(At)
        
        S_trn_11 = mul_diag44(Bt,At_inv)             #S_trn_11 = Bt@ np.linalg.inv(At)
        S_trn_21=2*At_inv                            #S_trn_21 = 2*np.linalg.inv(At)
        S_trn_12=2*self.unit-S_trn_21
        S_trn_22=-S_trn_11         
        
        S_trn=[S_trn_11,S_trn_12,S_trn_21,S_trn_22] 
        if form=='matrix':
            S_trn=[vec2m(S_trn[0]),vec2m(S_trn[1]),vec2m(S_trn[2]),vec2m(S_trn[3])]
        
        return S_trn        
############### Output ########################################################  

    def coeff(self,S_global_sub):         
        #Compute Compute reflected and transmitted fields
        E_sub = S_global_sub@ self.c_src # x, y directions: E_ref_xy # Wref or Wtrn=I
        if len(E_sub.shape) <3:
            rx=E_sub[0:self.Nharm,0]
            ry=E_sub[self.Nharm:2*self.Nharm,0] 
        else:
            rx=E_sub[:,0:self.Nharm,0]
            ry=E_sub[:,self.Nharm:2*self.Nharm,0] 
    
        return rx,ry   
    
    def Rotate_SP(self,Coeff_XY): 
        # Euler rotation   
        rxP=Coeff_XY[0];    ryP=Coeff_XY[1]
        rxS=Coeff_XY[2];    ryS=Coeff_XY[3]
   
        #TM
        rpx_new=-rxP*np.cos(self.Azimuth_r) - ryP*np.sin(self.Azimuth_r)
        rpp=rpx_new/np.cos(self.AOI_r)
        rps=-rxP*np.sin(self.Azimuth_r) + ryP*np.cos(self.Azimuth_r)

        #TE
        rsx_new=-rxS*np.cos(self.Azimuth_r) -ryS*np.sin(self.Azimuth_r)
        rsp=rsx_new/np.cos(self.AOI_r)
        rss=-rxS*np.sin(self.Azimuth_r) + ryS*np.cos(self.Azimuth_r)    

        return rpp, rps,rsp,rss
    
    def Reflectance(self,rx,ry):
        rz=-(self.Kx*rx+self.Ky*ry)/self.Kzr
        r2=np.square(np.abs(rx)) + np.square(np.abs(rz))+np.square(np.abs(ry))
        
        R=np.real(-self.Kzr)* r2/np.real(self.kz_inc)   
                   
        return R
    
    def Transmittance(self,tx,ty):
        tz=-(self.Kx*tx+self.Ky*ty)/self.Kzt
        t2=np.square(np.abs(tx)) + np.square(np.abs(ty))+ np.square(np.abs(tz))
       
        T=np.real(self.Kzt)* t2/np.real(self.kz_inc)        
        return T
    
#################################################################################################
       
    def Convmat2D(self,A): 
        N = A.shape;

        NH = (2*self.mx+1) * (2*self.my+1) # harmonic number
    
        p = list(range(-self.mx, self.mx + 1)); 
        q = list(range(-self.my, self.my + 1));  
        
        Af=np.fft.fftshift(np.fft.fft2(A))/N[-2]/N[-1]
        
        p0 = int((N[0] / 2));     q0 = int((N[1] / 2)); 
        
        ret = np.zeros((NH, NH),dtype=complex)   
        for qrow in range(2*self.my+1): 
            for prow in range(2*self.mx+1):             
                row = (prow) * (2*self.my+1) + qrow; 
                for qcol in range(2*self.my+1):
                    for pcol in range(2*self.mx+1): 
                        col = (pcol) * (2*self.my+1) + qcol; 
                        pfft = p[prow] - p[pcol]; 
                        qfft = q[qrow] - q[qcol];
                        ret[row, col] = Af[p0 + pfft, q0 + qfft]; 

        return ret    
   
    def eps_Fourier(self,layer): 
                
        mask=layer['pattern']['mask']         
        
        e_medium=layer['medium']['eps']
        e_pattern=layer['pattern']['eps']
        
        e_medium,e_pattern=self.force_array_wl_form(e_medium,e_pattern)          
          
        N_wavelength=len(self.wavelength_range)
        
        if np.all(mask==1): # homo layer
            ERC_CONV=[e_medium[wth] for wth in range(N_wavelength)]     
            return np.stack(ERC_CONV)
        
        else:               
            ERC_CONV=[];
            for wth in range(N_wavelength):  
                geo_filled=Fill_material(mask,e_medium[wth],e_pattern[wth])                   
                erc_conv = self.Convmat2D(geo_filled)                  
                ERC_CONV.append(erc_conv) 
            return np.stack(ERC_CONV)     
    
    
##################################### 2D Azimuth=0  ###########################################
    def SourcePlaneWave(self, mode='TE'):
                    
        delta_vec = np.zeros(self.Nharm)
        delta_vec[int(np.floor(self.Nharm/2))] = 1   
        
        if mode=='TE':  # 'pte': polar_angle=90// with y  
            Px= - np.sin(self.Azimuth_r)
            Py=np.cos(self.Azimuth_r)
            self.mode_2D=1
        if mode=='TM': # 'ptm': polar_angle=0
            self.mode_2D=2
            Px=np.cos(self.AOI_r)*np.cos(self.Azimuth_r) 
            Py=np.cos(self.AOI_r)*np.sin(self.Azimuth_r)             
        
        self.Px=Px ; self.Py=Py           
        #Compute Source Field
        e_src = np.zeros(2*self.Nharm)  
        e_src[0:self.Nharm] = Px*delta_vec
        e_src[self.Nharm:2*self.Nharm] =Py*delta_vec      
        self.c_src=e_src.reshape(2*self.Nharm,1)  
   
