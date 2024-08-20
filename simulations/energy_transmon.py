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
import cycler
from matplotlib import rcParams, cycler  #runtime mpl layout modifications
from matplotlib import cm
plt.style.use(r'C:\Users\jiaop\OneDrive\Skrivebord\Fluxonium_vrs.3\Simulations\style.mplstyle')


save_plots = True
show_plots = True
adjust = False
# colors = plt.cm.Set3(np.linspace(0.3, 1.1, 8)) 

#-------------------------------------------------------------------------------------------------------------
# The energy simulation of a transmon qubitor
# Define the potential
def potential(E_J,phi):
    return -130*E_J*np.cos(phi) +24

def hamiltonian(E_J, E_C, N, phi):
    # e = 1.60217663*10**(-19) # electron charge in coulombs
    e = 1
    delta = phi[3]-phi[2]
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
        JJ[i][i] = - E_J*np.cos(Phi[i][i])

    #Define the Hamiltonian
    Hamiltonian = C + JJ

    eig_vals, eig_vec = sp.linalg.eigh(Hamiltonian)  
    # print(f"eigenvalues",eig_vals)
    return eig_vals,eig_vec


# Define the initial parameters 
init_E_J = 0.2
init_E_C = 2

N = 1000+1
periodicity = np.pi
phi = np.linspace(-periodicity, periodicity, N)

eig_vals, eig_vec = hamiltonian(init_E_J,init_E_C, N, phi)

line = {}
for x in range(1, 5):
    line['line{0}'.format(x)] = (eig_vec.T[x]+eig_vals[x])


if show_plots == True: 
    #Define the range of the variables
    # create figure 
    fig, ax = plt.subplots()
    line, = ax.plot(phi, potential(init_E_J,phi))
    # line2, = ax.plot(phi, fluxonium_potential( init_E_J,phi, init_phi_ext), lw=2)
    # line3, = ax.plot(phi,fluxonium_potential( init_E_J,phi, init_phi_ext),'ko')

    eig_vals, eig_vec = hamiltonian(init_E_J,init_E_C,N, phi)
    eig_vals = eig_vals -1.9
    for i in range(0, 4):
        print(eig_vals[i+1] - eig_vals[i])
    
    wavefunctions = {}
    lines = {}
    for x in range(0, 5):
        wavefunctions["line{0}".format(x)], = ax.plot(phi, (30*eig_vec.T[x]+eig_vals[x]), label=f"\u03A8_{x}")
        lines["line{0}".format(x)] = ax.axhline(y=eig_vals[x], color=wavefunctions["line{0}".format(x)].get_color(), linestyle='--', lw = 1)
        # add aotation to the energy levels |n>
        label_text = r"$|{0}\rangle$".format(x)
        ax.text(periodicity + 0.2, eig_vals[x], label_text, color=wavefunctions["line{0}".format(x)].get_color(), )

    ax.set_xlabel(r'$\phi$')
    ax.set_ylabel(r'Energy/$h$ (GHz) ') 
    ax.set_ylim([-2,    52])
    ax.set_xlim([-periodicity, periodicity])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: '{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'))
    ax.xaxis.set_major_locator(plt.MultipleLocator(base=np.pi))
    # ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    # fig.suptitle('Energy levels of LC harmonic oscillator', fontsize=16)
    
    if adjust == True:
        # adjust the main plot to make room for the sliders
        fig.subplots_adjust(left=0.1, bottom=0.3)

        # Make horizontal sliders to control the induction energy.
        ax_E_J = fig.add_axes([0.1, 0.1, 0.7, 0.04])
        E_J_slider = Slider(
            ax=ax_E_J,
            label='$E_J$/h (GHz)',
            valmin=0.,
            valmax=10,
            valinit=init_E_J,
        )

        # Make horizontal sliders to control the capacitor energy.
        ax_E_C = fig.add_axes([0.1, 0.05, 0.7, 0.04])
        E_C_slider = Slider(
            ax=ax_E_C,
            label="$E_C$/h (GHz)",
            valmin=0.,
            valmax=10,
            valinit=init_E_C,
        )


        # The function to be called anytime a slider's value changes
        def update(val):
            line.set_ydata(potential( E_J_slider.val,phi))
            eig_vals, eig_vec = hamiltonian(E_J_slider.val, E_C_slider.val,N, phi)
            for x in range(0, 5):
                wavefunctions["line{0}".format(x)].set_ydata(10*eig_vec.T[x]+eig_vals[x])
                lines["line{0}".format(x)].set_ydata(eig_vals[x])
            fig.canvas.draw_idle()

        E_J_slider.on_changed(update)
        E_C_slider.on_changed(update)

        # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
        resetax = fig.add_axes([0.8, 0.0, 0.1, 0.04])
        button = Button(resetax, 'Reset', hovercolor='0.975')


        def reset(event):
            E_J_slider.reset()
            E_C_slider.reset()
        button.on_clicked(reset)


# ax.legend()
plt.tight_layout()
plt.show()

if save_plots == True: 
    path = r'C:\Users\jiaop\OneDrive\Skrivebord\Fluxonium_vrs.3\Report\Images'
    name_file = 'energy_levels_transmon.pdf'
    fig.savefig(os.path.join(path, name_file), bbox_inches='tight')






# #-------------------------------------------------------------------------------------------------------------
# # The energy simulation of a transmon qubito
# # Define the potential
# def potential(E_J,phi):
#     return -E_J*np.cos(phi) 

# def hamiltonian(E_J, E_C, N, phi, n_g):
#     # e = 1.60217663*10**(-19) # electron charge in coulombs
#     e = 1
#     delta = phi[3]-phi[2]
#     # # make our matrix phi
#     Phi = np.zeros((N,N))
#     for i in range(N):
#         Phi[i][i]= phi[i]

#     # # q^2 approximated: 
#     a = np.ones((1, N-1))[0]
#     b = np.ones((1,N))[0]
#     q_2 = np.dot(( np.diag(-2*b,0) + np.diag(a, -1) + np.diag(a, 1)), (-(1))/(delta**2))
#     # n_2 = np.square(1/(2*e))*q_2
#     N_g = np.zeros((N,N))
#     for i in range(N):
#         N_g[i][i]= n_g

        
#     # Conductor term: kinetic energy
#     C = np.dot(4*E_C,q_2)

#     # JJ term: should be a positive diagonal matrix. 
#     JJ = np.zeros((N,N))
#     for i in range(N):
#         JJ[i][i] = - E_J*np.cos(Phi[i][i])

#     #Define the Hamiltonian
#     Hamiltonian = C + JJ

#     eig_vals, eig_vec = sp.linalg.eigh(Hamiltonian)  
#     # print(f"eigenvalues",eig_vals)
#     return eig_vals,eig_vec


# # Define the initial parameters 
# init_E_J = 1
# init_E_C = 0.5

# N = 1000+1
# periodicity = np.pi
# phi = np.linspace(-periodicity, periodicity, N)

# eig_vals, eig_vec = hamiltonian(init_E_J,init_E_C, N, phi)
# # eig_vals = eig_vals *0.2

# line = {}
# for x in range(1, 5):
#     line['line{0}'.format(x)] = (eig_vec.T[x]+eig_vals[x])


# if show_plots == True: 
#     #Define the range of the variables
#     # create figure 
#     fig, ax = plt.subplots()
#     line, = ax.plot(phi, potential(init_E_J,phi))
#     # line2, = ax.plot(phi, fluxonium_potential( init_E_J,phi, init_phi_ext), lw=2)
#     # line3, = ax.plot(phi,fluxonium_potential( init_E_J,phi, init_phi_ext),'ko')

#     eig_vals, eig_vec = hamiltonian(init_E_J,init_E_C,N, phi)
#     eig_vals = eig_vals *0.3
#     for i in range(0, 4):
#         print(eig_vals[i+1] - eig_vals[i])
    
#     wavefunctions = {}
#     lines = {}
#     for x in range(0, 5):
#         wavefunctions["line{0}".format(x)], = ax.plot(phi, (15*eig_vec.T[x]+eig_vals[x]), label=f"\u03A8_{x}")
#         lines["line{0}".format(x)] = ax.axhline(y=eig_vals[x], color=wavefunctions["line{0}".format(x)].get_color(), linestyle='--', lw = 1)
#         # add aotation to the energy levels |n>
#         label_text = r"$|{0}\rangle$".format(x)
#         ax.text(periodicity + 0.2, eig_vals[x], label_text, color=wavefunctions["line{0}".format(x)].get_color(), )

#     ax.set_xlabel(r'$\phi$')
#     ax.set_ylabel('Energy/h (GHz) ') 
#     ax.set_ylim([0,    25])
#     ax.set_xlim([-periodicity, periodicity])
#     ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: '{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'))
#     ax.xaxis.set_major_locator(plt.MultipleLocator(base=np.pi))
#     # ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
#     # fig.suptitle('Energy levels of LC harmonic oscillator', fontsize=16)
    
#     if adjust == True:
#         # adjust the main plot to make room for the sliders
#         fig.subplots_adjust(left=0.1, bottom=0.3)

#         # Make horizontal sliders to control the induction energy.
#         ax_E_J = fig.add_axes([0.1, 0.1, 0.7, 0.04])
#         E_J_slider = Slider(
#             ax=ax_E_J,
#             label='$E_J$/h (GHz)',
#             valmin=0.,
#             valmax=10,
#             valinit=init_E_J,
#         )

#         # Make horizontal sliders to control the capacitor energy.
#         ax_E_C = fig.add_axes([0.1, 0.05, 0.7, 0.04])
#         E_C_slider = Slider(
#             ax=ax_E_C,
#             label="$E_C$/h (GHz)",
#             valmin=0.,
#             valmax=100,
#             valinit=init_E_C,
#         )


#         # The function to be called anytime a slider's value changes
#         def update(val):
#             line.set_ydata(potential( E_J_slider.val,phi))
#             eig_vals, eig_vec = hamiltonian(E_J_slider.val, E_C_slider.val,N, phi)
#             for x in range(0, 5):
#                 wavefunctions["line{0}".format(x)].set_ydata(10*eig_vec.T[x]+eig_vals[x])
#                 lines["line{0}".format(x)].set_ydata(eig_vals[x])
#             fig.canvas.draw_idle()

#         E_J_slider.on_changed(update)
#         E_C_slider.on_changed(update)

#         # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
#         resetax = fig.add_axes([0.8, 0.0, 0.1, 0.04])
#         button = Button(resetax, 'Reset', hovercolor='0.975')


#         def reset(event):
#             E_J_slider.reset()
#             E_C_slider.reset()
#         button.on_clicked(reset)


# # ax.legend()
# plt.tight_layout()
# plt.show()

# if save_plots == True: 
#     path = r'C:\Users\jiaop\OneDrive\Skrivebord\Fluxonium_vrs.3\Report\Images'
#     name_file = 'Transmon_energy_levels.pdf'
#     fig.savefig(os.path.join(path, name_file), bbox_inches='tight')



class transmon_charge:
    def __init__(self, E_J1, E_J2,E_C,phi_e,k, n_cutoff,n_g):
        self.E_J = E_J
        self.E_J1 = E_J1
        self.E_J2 = E_J2
        self.E_C = E_C
        self.phi_e = phi_e
        self.k = k
        self.n_cutoff = n_cutoff
        self.n_g = n_g
        
    # We make our phi matrix
    phi = sp.sparse.csr_matrix((k,k))
    phi.setdiag(1, k= -1)

    # We define our n-matrix with cutoff. 
    n_array = np.linspace(-n_cutoff,n_cutoff, k)
    n = sp.sparse.csr_matrix((k, k))
    n.setdiag(n_array)
    #we define our n2 matrix
    n2 = n @ n

    # we define our charge offset matrix
    n_g_array = np.linspace(-n_g,n_g, k)
    n_g = sp.sparse.csr_matrix((k, k))
    n_g.setdiag(n_g_array)

    # we define our phi matrix
    phi_inv = sp.sparse.csr_matrix((k,k))
    phi_inv.setdiag(1, k= 1)

    # we define our cos term
    cos = phi + phi_inv

    # calculate the capacitor term
    n_hat = n2 - 2*n_g
    Capacitor = n_hat*(4*E_C)

    # Calculate the Josephson term
    gamma = E_J2/E_J1
    d = (gamma-1)/(gamma)
    E_J = (E_J1+E_J2)*np.sqrt(math.cos(phi_e)**2 +d**2* (math.sin(phi_e)**2)) # a new Josephson energy
    
    JJ = cos.multiply(E_J)

    def Hamiltonian(self): 
        return Capacitor - JJ

    eig_vals, eig_vec = sp.linalg.eigh(Hamiltonian)

    w_q = eig_vals[1]-eig_vals[0]
    
    return eig_vals, eig_vec, Hamiltonian, n.toarray(), E_J, E_C