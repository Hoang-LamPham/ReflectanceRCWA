#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 17:04:45 2023

@author: pham
"""

import numpy as np

def Tauc_Lorentz(E,parameter):# Eg,e_inf, A, E0, C
    Eg=parameter[0]
    e_inf=parameter[1]    
   
    parameter=parameter[2:]
    A=np.array([parameter[i] for i in range(len(parameter)) if i%3==0]).reshape(-1,1)
    E0=np.array([parameter[i] for i in range(len(parameter))if i%3==1]).reshape(-1,1)
    C=np.array([parameter[i] for i in range(len(parameter))if i%3==2]).reshape(-1,1)

    Eg2=Eg**2;   E2=E**2; C2=C**2; E02=E0**2

    e2= A*E0*C*(E-Eg)**2/(E*(E2 - E02)**2 + E*C2*E2)
    e2[:,E<Eg]=0# if E<=Eg:e2=0
    e2=np.sum(e2,axis=0)

    alpha=np.sqrt(4*E02 - C2); alpha2=alpha**2
    gamma=np.sqrt(E02 -C2/2)  ; gamma2=gamma**2
   
    Psi4=(E2 - gamma2)**2 +alpha2*C2/4    
   
    aln=(Eg2 -E02)*E2 +Eg2*C2 -E02*(E02+3*Eg2)
    atan=(E2-E02)*(E02+Eg2) +Eg2*C2

    t1=A*C*aln/(2*np.pi*Psi4*alpha*E0)*np.log((E02+Eg2+alpha*Eg)/(E02+Eg2 -alpha*Eg))
   
    t2= - A*atan/(np.pi*Psi4*E0)*(np.pi - np.arctan(2*Eg/C+alpha/C)+ np.arctan(-2*Eg/C+alpha/C)   )
   
    t3=4*A*E0*Eg*(E2-gamma2)/(np.pi*Psi4*alpha)*(np.arctan(alpha/C +2*Eg/C)+np.arctan(alpha/C -2*Eg/C))
   
    t4=-A*E0*C*(E2+Eg2)/(np.pi*Psi4*E)*np.log(np.abs(E-Eg)/(E+Eg))
   
    t5=2*A*E0*C*Eg/(np.pi*Psi4)*np.log(np.abs(E-Eg)*(E+Eg)/np.sqrt((E02-Eg2)**2 +Eg2*C2))
   
    e1= t1+t2+t3+t4+t5
    e1=np.sum(e1,axis=0)+e_inf
   
    return e1 - 1j*e2
    