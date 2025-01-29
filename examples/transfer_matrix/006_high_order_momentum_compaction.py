# copyright ############################### #
# This file is part of the Xtrack Package.  #
# Copyright (c) CERN, 2021.                 #
# ######################################### #

import numpy as np
from matplotlib import pyplot as plt
from scipy import constants

import xobjects as xo
context = xo.ContextCpu(omp_num_threads=0)
import xtrack as xt
import xpart as xp

muonMass = constants.value('muon mass energy equivalent in MeV')*1E6

n_macroparticles = 40

circumference = 10000
energy = 5000 # [GeV]
p0c = np.sqrt((energy*1E9)**2-muonMass**2)
gamma = np.sqrt(1+(p0c/muonMass)**2)
beta = np.sqrt(1-1/gamma**2)

voltage_rf = 820.0e6
harmonic_rf = 43478 # for a 1.3 GHz RF cavity
frequency_rf = harmonic_rf * beta * constants.c / circumference
lag_rf = 180
alphap_0 = 1e-6

nTurn = 1000
energy_increment = 0

for ii_high_order_alphap, high_order_alphap in enumerate([(0, 0), (1e-6, 0), (1e-5, 0), (1e-4, 0), (0, 1e-3)]):

    line_matrix = xt.LineSegmentMap(
            length=circumference,
            alfx=[0, 0],
            betx=[100, 100],
            dx=[0, 0],
            alfy=[0, 0],
            bety=[100, 100],
            qx=0.155, qy=0.16,
            longitudinal_mode='nonlinear',
            voltage_rf=[voltage_rf],
            frequency_rf=[frequency_rf],
            lag_rf=[lag_rf],
            momentum_compaction_factor=alphap_0,
            high_order_momentum_compaction_factor=high_order_alphap,
            energy_ref_increment=energy_increment,
            energy_increment=0)

    line = xt.Line(elements=[line_matrix])
    line.particle_ref = xp.Particles(p0c=p0c, q0=1.0, x=0.0, px=0.0, y=0.0, py=0.0, zeta=0.0, delta=0.0)
    line.build_tracker()

    # Create the particle distribution
    z_particles = np.zeros(n_macroparticles)
    z_particles = np.linspace(-0.1, 0.1, n_macroparticles)
    delta_particles = np.linspace(-0.1, 0.1, n_macroparticles)

    particles = xt.Particles(_context=context,
                            q0=1,
                            mass0=muonMass,
                            p0c=p0c,
                            x=np.zeros(n_macroparticles),
                            px=np.zeros(n_macroparticles),
                            y=np.zeros(n_macroparticles),
                            py=np.zeros(n_macroparticles),
                            zeta=z_particles,
                            delta=delta_particles,
                            )

    # Track the particles
    line.track(particles, num_turns=nTurn, turn_by_turn_monitor=True)

    # Plot the longitudinal phase space
    plt.figure(ii_high_order_alphap)
    plt.scatter(line.record_last_track.zeta, line.record_last_track.pzeta, s=1)
    plt.xlabel('zeta [m]')
    plt.ylabel('pzeta [eV/c]')
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.11, 0.11)
    plt.title(f'$\\alpha_{{p, 0}}$ = {alphap_0} $\\alpha_{{p, 1}}$ = {high_order_alphap[0]} $\\alpha_{{p, 2}}$ = {high_order_alphap[1]}')

plt.show()