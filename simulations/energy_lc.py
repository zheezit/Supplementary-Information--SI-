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

cm = 1/2.54 # centimeters in inches
# rcParams.update({
#                     'axes.prop_cycle' : (cycler(color=cmc.actonS.colors)
#                                       + cycler(linestyle=['-', '--', ':', '-.'])*int(len(cmc.batlowKS.colors)//4)),  #https://matplotlib.org/cycler/
#                 #  'axes.grid' : True, #Good for 1D plots, not for 2D plots
#                 #  'text.usetex' : False,  #uses your local latex compiler to render the text. Slow, but looks great
#                  'font.family' : 'Times New Roman',  #Ferdinands standard
#                  'font.weight' : 'normal',
#                  'font.stretch' : 'normal',
#                  'font.size' : 13,  #Ferdinands standard is 7
#                  'axes.titlesize' : 10,   #personal choice
#                 #  'figure.titlesize': 11,  #personal choice
#                 #  'figure.constrained_layout.use' : True,  #the best layout algorithm'
#                 #  'figure.dpi' : 600,  #minimum resolution requested in the prx style guide (link below)
#                 #  'figure.figsize' : (7.0, 5.25),  # in inches. This is the small width prefered in https://cdn.journals.aps.org/files/styleguide-pr.pdf,
#                 # #                                 #with the same side ratio as the original matplotlib plot. It is supposed to fill a single column.
#                 # #                                 #The full width figure would be (7.0, 5.25), again with 7.0 being the important spec.
#                 'figure.figsize' : (12.0*cm, 10.0*cm),  # in inches. This is the small width prefered in https://cdn.journals.aps.org/files/styleguide-pr.pdf,
#                 #                                 #with the same side ratio as the original matplotlib plot. It is supposed to fill a single column.
#                 #                                 #The full width figure would be (7.0, 5.25), again with 7.0 being the important spec.
#                 # 'axes.linewidth' : 1,
#                 # 'lines.linewidth' : 0.5,
#                 # 'lines.markersize' : 2,
#                 })



save_plots = True
show_plots = True
adjust = False
# colors = plt.cm.Set3(np.linspace(0.3, 1.1, 8)) 

#-------------------------------------------------------------------------------------------------------------

# The energy simulation of an LC hamonic oscillator
# Define the potential

def LC_potential(E_L,phi):
    return 1/2*E_L*phi**2 -2

def hamiltonian(E_L, E_C, N, phi):
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

    # # Inductor term: positiv diagonal matrix
    inductor = np.zeros((N,N))
    for i in range(N):
        inductor[i][i] = 1/2*E_L*(Phi[i][i])**2

    #Define the Hamiltonian
    Hamiltonian = C + inductor

    eig_vals, eig_vec = sp.linalg.eigh(Hamiltonian)  
    # print(f"eigenvalues",eig_vals)
    return eig_vals,eig_vec


# Define the initial parameters 
init_E_L = 1
init_E_C = 1

periodicity = 2*np.pi
number_levels = 4

N = 1000+1
phi = np.linspace(-periodicity, periodicity, N)


eig_vals, eig_vec = hamiltonian(init_E_L,init_E_C, N, phi)

line = {}
for x in range(0, 5):
    line['line{0}'.format(x)] = (eig_vec.T[x]+eig_vals[x])


if show_plots == True: 
    #Define the range of the variables
    # create figure 
    fig, ax = plt.subplots()
    line, = ax.plot(phi, LC_potential( init_E_L,phi))
    # line2, = ax.plot(phi, fluxonium_potential( init_E_L,phi, init_phi_ext), lw=2)
    # line3, = ax.plot(phi,fluxonium_potential( init_E_L,phi, init_phi_ext),'ko')

    eig_vals, eig_vec = hamiltonian(init_E_L,init_E_C,N, phi)
    eig_vals = eig_vals -1.5
    for i in range(0, 5):
        print(eig_vals[i+1] - eig_vals[i])
    
    wavefunctions = {}
    lines = {}
    for x in range(0, 5):
        wavefunctions["line{0}".format(x)], = ax.plot(phi, (20*eig_vec.T[x]+eig_vals[x]), label=f"\u03A8_{x}")
        lines["line{0}".format(x)] = ax.axhline(y=eig_vals[x], color=wavefunctions["line{0}".format(x)].get_color(), linestyle='--', lw = 1)
        # add aotation to the energy levels |n>
        label_text = r"$|{0}\rangle$".format(x)
        ax.text(periodicity + 0.2, eig_vals[x], label_text, color=wavefunctions["line{0}".format(x)].get_color(), )


    ax.set_xlabel(r'$\phi$')
    ax.set_ylabel(r'Energy/$h$ (GHz) ') 
    ax.set_ylim([-2,    18])
    ax.set_xlim([-periodicity, periodicity])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, pos: '{:.0g}$\pi$'.format(val/np.pi) if val !=0 else '0'))
    ax.xaxis.set_major_locator(plt.MultipleLocator(base=np.pi))
    # ax.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5))
    # fig.suptitle('Energy levels of LC harmonic oscillator', fontsize=16)
    
    if adjust == True:
        # adjust the main plot to make room for the sliders
        fig.subplots_adjust(left=0.1, bottom=0.3)

        # Make horizontal sliders to control the induction energy.
        ax_E_L = fig.add_axes([0.1, 0.1, 0.7, 0.04])
        E_L_slider = Slider(
            ax=ax_E_L,
            label='$E_L$/h (GHz)',
            valmin=0.,
            valmax=10,
            valinit=init_E_L,
        )

        # Make horizontal sliders to control the capacitor energy.
        ax_E_C = fig.add_axes([0.1, 0.05, 0.7, 0.04])
        E_C_slider = Slider(
            ax=ax_E_C,
            label="$E_C$/h (GHz)",
            valmin=0.,
            valmax=100,
            valinit=init_E_C,
        )


        # The function to be called anytime a slider's value changes
        def update(val):
            line.set_ydata(LC_potential( E_L_slider.val,phi))
            eig_vals, eig_vec = hamiltonian(E_L_slider.val, E_C_slider.val,N, phi)
            for x in range(0, 5):
                wavefunctions["line{0}".format(x)].set_ydata(10*eig_vec.T[x]+eig_vals[x])
                lines["line{0}".format(x)].set_ydata(eig_vals[x])
            fig.canvas.draw_idle()

        E_L_slider.on_changed(update)
        E_C_slider.on_changed(update)

        # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
        resetax = fig.add_axes([0.8, 0.0, 0.1, 0.04])
        button = Button(resetax, 'Reset', hovercolor='0.975')


        def reset(event):
            E_L_slider.reset()
            E_C_slider.reset()
        button.on_clicked(reset)


# ax.legend()
plt.tight_layout()
plt.show()

if save_plots == True: 
    path = r'C:\Users\jiaop\OneDrive\Skrivebord\Fluxonium_vrs.3\Report\Images'
    name_file = 'energy_levels_lc.pdf'
    fig.savefig(os.path.join(path, name_file), bbox_inches='tight')



