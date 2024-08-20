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

import cmcrameri.cm as cmc  #scientific colormaps, perceptually uniform for all.
from cmcrameri import show_cmaps  #demonstrating all the colormaps

from matplotlib import rcParams, cycler  #runtime mpl layout modifications
from matplotlib import cm
plt.style.use(r'C:\Users\jiaop\OneDrive\Skrivebord\Fluxonium_vrs.3\Simulations\style.mplstyle')


# Use sparce matrices

save_plots = True
show_plots = True
adjust = False
# # colors = plt.cm.Set3(np.linspace(0.3, 1.1, 8)) 
paths = r'C:\Users\jiaop\OneDrive\Skrivebord\Fluxonium_vrs.3\Report\Images'


#-------------------------------------------------------

def hamiltonian(n_g, E_C, E_J, n_states):
    # n_states defines the range of charge states considered around 0
    n = np.arange(-n_states, n_states + 4)
    N = len(n)
    H = np.zeros((N, N))
    
    # Charging energy
    for i in range(N):
        H[i, i] = E_C * (n[i] - n_g)**2
    
    # Josephson energy
    for i in range(N-1):
        H[i, i+1] = H[i+1, i] = -E_J / 100
    
    return H

def calculate_energies(n_g_values, E_C, E_J, n_states=5):
    energies = np.zeros((len(n_g_values), 3))  # Store the first three energies for each n_g
    
    for i, n_g in enumerate(n_g_values):
        H = hamiltonian(n_g, E_C, E_J, n_states)
        eigenvalues, _ = np.linalg.eigh(H)
        energies[i, :] = eigenvalues[:3]  # Keep the first three energies
    
    return energies

# Parameters
E_C = 1.0  # Charging energy
E_J = 1.0  # Josephson energy
n_g_values = np.linspace(-2, 2, 500)

# Calculate energies
energies = calculate_energies(n_g_values, E_C, E_J)

# Plotting
fig, ax = plt.subplots()
#plot the first three energy levels with colors from the colors array
for i in range(3):
    ax.plot(n_g_values, energies[:, i]*5 + 0.4,  color='C'+str(i+1))
    # add a notation for the energy levels |n> with the same colors as the lines
    # label_text = r"$|{0}\rangle$".format(i)
    # ax.text(n_g_values[-1] + 0.3, energies[-1, i], label_text, color='C'+str(i))

ax.set_xlabel('$n_g$')
ax.set_ylabel('Energy/h (GHz)')
plt.title(f'$E_J/E_C = 1$')


if show_plots:
    plt.tight_layout()
    plt.show()


if save_plots: 
    #save the plot in the image folder
    path = paths
    name_file = 'energy_levels_Transmon_EJEC=1.pdf'
    fig.savefig(os.path.join(path, name_file), bbox_inches='tight')




#-------------------------------------------------------
def hamiltonian(n_g, E_C, E_J, n_states):
    # n_states defines the range of charge states considered around 0
    n = np.arange(-n_states, n_states + 4)
    N = len(n)
    H = np.zeros((N, N))
    
    # Charging energy
    for i in range(N):
        H[i, i] = E_C * (n[i] - n_g)**2
    
    # Josephson energy
    for i in range(N-1):
        H[i, i+1] = H[i+1, i] = -E_J / 20
    
    return H

def calculate_energies(n_g_values, E_C, E_J, n_states=5):
    energies = np.zeros((len(n_g_values), 3))  # Store the first three energies for each n_g
    
    for i, n_g in enumerate(n_g_values):
        H = hamiltonian(n_g, E_C, E_J, n_states)
        eigenvalues, _ = np.linalg.eigh(H)
        energies[i, :] = eigenvalues[:3]  # Keep the first three energies
    
    return energies

# Parameters
E_C = 1.0  # Charging energy
E_J = 5.0  # Josephson energy
n_g_values = np.linspace(-2, 2, 500)

# Calculate energies
energies = calculate_energies(n_g_values, E_C, E_J)

# Plotting
fig, ax = plt.subplots()
#plot the first three energy levels with colors from the colors array
for i in range(3):
    ax.plot(n_g_values, energies[:, i] + 0.4,  color='C'+str(i+1))
    # add a notation for the energy levels |n> with the same colors as the lines
    # label_text = r"$|{0}\rangle$".format(i)
    # ax.text(n_g_values[-1] + 0.3, energies[-1, i], label_text, color='C'+str(i))

ax.set_xlabel('$n_g$')
ax.set_ylabel('Energy/h (GHz)')
plt.title(f'$E_J/E_C = 5$')




if show_plots:
    plt.tight_layout()
    plt.show()


if save_plots == True: 
    path = r'C:\Users\jiaop\OneDrive\Skrivebord\Fluxonium_vrs.3\Report\Images'
    name_file = 'energy_levels_Transmon_EJEC=5.pdf'
    fig.savefig(os.path.join(path, name_file), bbox_inches='tight')





#-------------------------------------------------------
def hamiltonian(n_g, E_C, E_J, n_states):
    # n_states defines the range of charge states considered around 0
    n = np.arange(-n_states, n_states + 4)
    N = len(n)
    H = np.zeros((N, N))
    
    # Charging energy
    for i in range(N):
        H[i, i] = E_C * (n[i] - n_g)**2
    
    # Josephson energy
    for i in range(N-1):
        H[i, i+1] = H[i+1, i] = -E_J /15
    
    return H

def calculate_energies(n_g_values, E_C, E_J, n_states=5):
    energies = np.zeros((len(n_g_values), 3))  # Store the first three energies for each n_g
    
    for i, n_g in enumerate(n_g_values):
        H = hamiltonian(n_g, E_C, E_J, n_states)
        eigenvalues, _ = np.linalg.eigh(H)
        energies[i, :] = eigenvalues[:3]  # Keep the first three energies
    
    return energies

# Parameters
E_C = 0.2  # Charging energy
E_J = 10.0  # Josephson energy
n_g_values = np.linspace(-2, 2, 500)

# Calculate energies
energies = calculate_energies(n_g_values, E_C, E_J)

# Plotting
fig, ax = plt.subplots()
#plot the first three energy levels with colors from the colors array
for i in range(3):
    ax.plot(n_g_values, energies[:, i]*2 + 2,  color='C'+str(i+1))
    # add a notation for the energy levels |n> with the same colors as the lines
    # label_text = r"$|{0}\rangle$".format(i)
    # ax.text(n_g_values[-1] + 0.3, energies[-1, i], label_text, color='C'+str(i))

ax.set_xlabel('$n_g$')
ax.set_ylabel('Energy/h (GHz)')
plt.title(f'$E_J/E_C = 50$')




if show_plots:
    plt.tight_layout()
    plt.show()

if save_plots == True: 
    path = r'C:\Users\jiaop\OneDrive\Skrivebord\Fluxonium_vrs.3\Report\Images'
    name_file = 'energy_levels_Transmon_EJEC=50.pdf'
    fig.savefig(os.path.join(path, name_file), bbox_inches='tight')

