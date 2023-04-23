# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 16:55:35 2023

@author: znoll
"""

import numpy as np
import matplotlib.pyplot as plt 

### CLT - Composite Laminate Technique for shell macroscopic stiffness
## Assumptions
# 1. No porosity 
# 2. Symmetric laminate


def input_values():
    global t, ply_angle, num_ply, num_gen
    print(f"*"*50)
    t = float(input("Ply thickness in millimetres: "))
    t /= 10e3
    
    ## Number of plies
    num_ply = int(input("Enter the number of plies:  "))    
    
    ply_angle = np.array(())
    ply_angle = np.array([])
    if num_ply % 2 == 0: # Even
        odd = 0
    else:
        odd = 1
        
    num_sym = int(num_ply/2) + odd
        
    for i in range(num_sym):
        v = int(input(f"Enter ply angle {i}:  "))
        ply_angle = np.append(ply_angle, v)
    
    ply_angle = np.append(ply_angle, ply_angle[:num_sym-odd][::-1])

    print(f"Ply angles are {ply_angle}")
    print(f"*"*50)
    
    Nx = float(input("Enter force in x-direction in N: ")) 
    Ny = float(input("Enter force in y-direction in N: ")) 
    Nxy = float(input("Enter shear force in xy-plane in N: ")) 
    Mx = float(input("Enter moment about x-axis in N.m:"))
    My = float(input("Enter moment about y-axis in N.m:"))
    Mxy = float(input("Enter twisting moment in N.m:"))
    
    num_gen = 2 
    
## Ply thickness 
try:
    num_gen
    print('Num_gen exists!')
    change_values = input("Press 1 to change values and press 2 to keep values")
    if change_values == 1:
        input_values()
    elif change_values == 2:
        pass
    elif change_values != 2:
        raise RuntimeError("Please restart and enter either 1 or 2")
        
except NameError:
    input_values()
        
       

## Material properties. 1 is fibre direction, 2 is perpendicular and 3 is out-of-plane
# Note that x,y is the basis for material direction 
#! Note: assumed from lab testing of ply 
E1 = 500e9 # GPa, fibre tensile modulus
E2 = 4e9 # GPa, matrix tensile modulus
G12 = 2e9 # GPa, shear modulus 
v12 =  0.3 # Poisson's ratio 

## Compliance matrix
S11 = 1/E1
S22 = 1/E2
S12 = -v12/E1
S66 = 1/G12
S = np.zeros((3,3))
S[0,0] = S11
S[0,1] = S[1,0] = S12
S[1,1] = S22
S[2,2] = S66

## Stiffness matrix
Q11 = E1/(1-v12**2)
Q22 = E2/(1-v12**2)
Q12 = v12*E1/(1-v12**2)
Q66 = G12
Q = np.zeros((3,3))
Q[0,0] = Q11
Q[0,1] = Q[1,0] = Q12
Q[1,1] = Q22
Q[2,2] = Q66

## Rotation matrix from global stresses to material stresses
def T_stress(angle):
    T = np.zeros((3,3)) # Stress rotation matrix
    angle = np.pi*angle/180
    m = np.cos(angle)
    n = np.sin(angle)
    
    T[0,0] = T[1,1] = m**2
    T[0,1] = T[1,0] = n**2
    T[0,2] = 2*m*n
    T[1,2] = -2*m*n
    T[2,0] = -m*n
    T[2,1] = m*n
    T[2,2] = m**2 - n**2
    
    return T

def T_strain(angle):
    T = np.zeros((3,3)) # Strain rotation matrix
    angle = np.pi*angle/180
    m = np.cos(angle)
    n = np.sin(angle)
    
    T[0,0] = T[1,1] = m**2
    T[0,1] = T[1,0] = n**2
    T[0,2] = m*n
    T[1,2] = -m*n
    T[2,0] = -2*m*n
    T[2,1] = 2*m*n
    T[2,2] = m**2 - n**2
    
    return T

## Compliance and stiffness matrices of individual plies based on global uni-directional
# properties and ply angle

def S_bar(angle): # Angle-transformed compliance matrix. From global stress to global strain.
    S_bar = np.linalg.inv(T_strain(angle)) @ S @ T_stress(angle)

    return S_bar

def Q_bar(angle): # Angle-transformed stiffness matrix. From global strain to global stress
    Q_bar = np.linalg.inv(T_stress(angle)) @ S @ T_strain(angle)

    return Q_bar

## Through-plane thickness vector 
z_vec = np.linspace(-t*((num_ply - 1)/2),t*((num_ply - 1)/2),num=num_ply) 

A = np.zeros((3,3)) # Top left matrix
D = np.zeros((3,3)) # Bottom right matrix i.e. symmetric layup 
aug_mat = np.zeros((3*2,3*2))
for k in range(len(ply_angle)):
    Qk = Q_bar(ply_angle[k])
    A += Qk * t
    D += (1/3) * Qk * t**3
aug_mat[:3,:3] = A
aug_mat[-3:,-3:] = D

N_long = np.zeros((6,1)) # Force vector. First three entries are forces.
                         # Last three are curvatures 
N_long[0,0] = 2000

epsilon_long = np.linalg.solve(aug_mat,N_long)

def stress_global(k): # In global directions
    Qk = Q_bar(ply_angle[k])
    ep = epsilon_long[:3]
    Kurvature = epsilon_long[-3:]
    z = z_vec[k]
    return Qk@ep + z*Qk@Kurvature

def stress_material(k):
    sigma_global = stress_global(k)
    angle = ply_angle[k]
    rotation_mat = T_stress(angle)
    return rotation_mat @ sigma_global

stress_matrix = np.zeros((3,num_ply))
for k in range(num_ply):
    stress_matrix[:,k] = stress_material(k)[:,0]
    print(f"The stresses in ply {k+1} are {stress_matrix[:,k]*10e-6} MPa")

print(f"*"*50)

# ! Change to user input 
F1t = 1500e6 #MPa longitudinal tensile strength
F1c = 1300e6 #Mpa, longitudinal compressive strength 
F2t = 50e6 #MPa, transverse tensile strength 
F2c = 240e6 #MPa, transverse compressive strength 
F6 = 100e6 #MPa, shear strength 

# Tsai-Hill
tsai_hill_vector = np.zeros((num_ply,1))
for k in range(num_ply):
    stress_vec = stress_matrix[:,k]
    sigma1 = stress_vec[0]
    sigma2 = stress_vec[1]
    sigma12 = stress_vec[2]
    
    # Fibre stress compressive or tensile
    if sigma1 > 0:
        X1 = F1t
    else:
        X1 = F1c
    
    # Matrix stress compressive or tensile 
    if sigma2 > 0:
        X2 = F1t 
        Y = F2t
    else: 
        X2 = F1c
        Y = F2c
    
    # Shear 
    S = F6    
    
    criterion = (sigma1/X1)**2 \
                -(sigma1*sigma2)/(X2**2) \
                +(sigma2/Y)**2 \
                +(sigma12/S)**2
    
    tsai_hill_vector[k,0] = criterion 

    print(f"The Tsai-Hill failure criterion for ply {k+1} is {criterion:.2f} ")

## Graphing 
# Create some mock data
# t = np.arange(0.01, 10.0, 0.01)
# data1 = np.exp(t)
# data2 = np.sin(2 * np.pi * t)

# fig, ax1 = plt.subplots()

# color = 'tab:red'
# ax1.set_xlabel('time (s)')
# ax1.set_ylabel('exp', color=color)
# ax1.plot(t, data1, color=color)
# ax1.tick_params(axis='y', labelcolor=color)

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# color = 'tab:blue'
# ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
# ax2.plot(t, data2, color=color)
# ax2.tick_params(axis='y', labelcolor=color)

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()
    
## 
#! Use PyQt (!)

# ! Action list
# 1. Make carpet plot (?)
# 2. Make polar graph of strength 
# 3. Make ply stress graph 
# 4. Analyse ply failure DONE
# 5. Allow user input to determine which failure model 
# 6. Make .exe with PyQt





    