import numpy as np 
from numpy import linalg as LA
import scipy as sp 
from scipy import sparse
from scipy.sparse import linalg

import os 
import sys
import math as math
import matplotlib 

from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.widgets import Slider, Button
matplotlib.use('TkAgg')


from matplotlib import rcParams, cycler  #runtime mpl layout modifications
from matplotlib import cm
plt.style.use(r'C:\Users\jiaop\OneDrive\Skrivebord\Fluxonium_vrs.3\Simulations\style.mplstyle')

save_plots = True
show_plots = True
adjust = False

# Define the Hamiltonian function as before
def hamiltonian(E_J, E_L, E_C, phi_ext, N, phi):
    delta = phi[3]-phi[2]
    e = 1.60217663*10**(-19) # electron charge in coulombs
    # e = 1

    # # make our matrix phi
    Phi = np.zeros((N,N))
    for i in range(N):
        Phi[i][i]= phi[i]

    # # q^2 approximated: 
    a = np.ones((1, N-1))[0]
    b = np.ones((1,N))[0]
    q_2 = np.dot(( np.diag(-2*b,0) + np.diag(a, -1) + np.diag(a, 1)), (-(1))/(delta**2))
    # n_2 = np.square(1/(2*e))*q_2

    # Conductor term: kinetic energy
    C = np.dot(4*E_C,q_2)

    # JJ term: should be a positive diagonal matrix. 
    JJ = np.zeros((N,N))
    for i in range(N):
        JJ[i][i] = E_J*np.cos(Phi[i][i]-phi_ext)

    # # Inductor term: positiv diagonal matrix
    inductor = np.zeros((N,N))
    for i in range(N):
        inductor[i][i] = 1/2*E_L*(Phi[i][i])**2

    #Define the Hamiltonian
    Hamiltonian = C - JJ + inductor

    eig_vals, eig_vec = sp.linalg.eigh(Hamiltonian)  
    # print(f"eigenvalues",eig_vals)
    return eig_vals,eig_vec

# Initialize parameters
E_J = 4.0
E_C = 1.0
E_L = 1.0
N = 201
phi = np.linspace(-2*np.pi, 2*np.pi, N)

# Define a range of external flux values to iterate over
phi_ext_range = np.linspace(-2*np.pi, 2*np.pi, 400)
energy_levels = np.zeros((4, len(phi_ext_range)))  # Store the first 4 energy levels

# Calculate energy levels for each external flux value
for i, phi_ext in enumerate(phi_ext_range):
    eig_vals, _ = hamiltonian(E_J, E_L, E_C, phi_ext, N, phi)
    energy_levels[:, i] = eig_vals[:4]  # Store the first 4 energy levels

periodicity = 2*np.pi



# Plotting
fig,ax = plt.subplots()
for i in range(3):
    ax.plot(phi_ext_range, energy_levels[i, :], label=f'Level {i+1}', color='C'+str(i+ 1))
    # add aotation to the energy levels |n>
    label_text = r"$|{0}\rangle$"
    ax.text(phi_ext_range[-1], energy_levels[i, -1], label_text, color='C'+str(i+ 1))



# put a strippled line at the zero energy level 
ax.axvline(x=0, color='C0', linestyle='--', lw=2)
ax.axvline(x=np.pi, color='C5', linestyle='--', lw=2)
ax.axvline(x=-np.pi, color='C5', linestyle='--', lw=2)

ax.set_xlabel(r'$\phi_{ext}$')
ax.set_ylabel(r'Energy/$h$ (GHz) ') 
# ax.set_ylim([-2, 18])
ax.set_xlim([-periodicity, periodicity])
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: '{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'))
ax.xaxis.set_major_locator(plt.MultipleLocator(base=np.pi))



# plt.title('Energy Levels as a Function of External Flux for Fluxonium Qubit')
# plt.grid(True)

plt.tight_layout()
plt.show()

if save_plots:
    path = r'C:\Users\jiaop\OneDrive\Skrivebord\Fluxonium_vrs.3\Report\Images'
    name_file = 'energy_levels_fluxonium_external.pdf'
    fig.savefig(os.path.join(path, name_file), bbox_inches='tight')
