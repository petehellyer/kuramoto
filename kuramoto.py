# python kuramoto
# based on the cabral flavour of the Kuramoto Model.
import numpy as np
import scipy.io
from pylab import *

# Network simulation of Kuramoto oscillators with time delays

mat = scipy.io.loadmat('Human_66.mat')
C = mat['C']
C = (C + C.transpose()) / 2  # Matrix of coupling weights (NxN) between pairs of regions (can be directed and weighted)
D = mat['L']  # Matrix of distances (in mm) (NxN)
N = C.shape[0]
IC = (np.random.rand(N) * (2 * np.pi)) - np.pi  # Initial Conditions
frequency_mean = 60  # Neural populations' average intrinsic frequency (Hz)
f_std = 10  # Standard deviation of intrinsic frequencies across regions, Can be 0 if all equal oscillators.
f_dist = np.zeros((N,
                   1))  # Vector containing the distribution of intrinsic frequencies. if all equal: f_dist=zeros(N,1)
# if normally distributed: f_dist=randn(N,1)
k = 1  # scaling of C
tau = 1  # this is the scaling for D
t_max = 240  # Total time of simulated activity (seconds)
dt = 0.0001  # integration step (smaller than smaller delays) ( in seconds) (eg. 0.0001 s = 0.1 ms)
sampling = 10  # sampling for saving simulated activity (eg. 10)  => 10*dt  = 1ms
sig_n = 1  # standard deviation of noise in the phases (can be zero, in radians)
trim = 30  # number of seconds to trim off of simulation

n_t_total = np.ceil(t_max / dt)  # total number of time steps e.g 300*0.0001 = 3000000
n_ts = np.ceil(n_t_total / sampling)  # same with downsampling ex. 300000
n_tcycle = 10000  # number of time steps in each cycle (for RAM memory reasons)
n_tcs = n_tcycle / sampling  # same with downsampling ex. 1000
n_cycles = np.ceil(n_t_total / n_tcycle)  # number of cycles ex. 300

C = k * (C / np.mean(C[:]))  # Normalize so that mean(C(:))=k;
C = dt * C  # Scale the coupling strengths per time step
# I          = C>0;            #
# d_m        = mean(D(I));           #
# v          = d_m/tau;              # conduction velocity (m/s)
stepsDelay = np.round(D / (dt * 1e3))  # number of time steps for delays
sig_noise = sig_n * np.sqrt(dt)  # Scale noise per time step
f_diff = f_dist * f_std  # define intrinsinc node frequencies.
omega_step = 2 * np.pi * frequency_mean * dt  # 0.0251radians if f=40Hz and dt = 0.0001.
omega_diff = 2 * np.pi * f_diff * dt
omegas = omega_step + omega_diff  # Natural phase increment at each time step

n_td = np.ceil(np.max(stepsDelay[:])) + 10  # number of time steps for maximal delays
n_tp = n_td + n_tcycle  # number of time steps in one cycle with the time for delays

th = np.zeros((N, n_tp))  # initialize phase timeseries for one cycle
th_p = np.zeros((N, n_tp))
ths = np.zeros((N, n_ts))  # initialize phase timeseries to save

# Initialization

th[:, 1] = IC
for n in np.arange(0, N):
    th[n, 0:n_td] = th[n, 0] + np.arange(0, n_td * omegas[n], omegas[n]) + sig_noise * np.random.randn(n_td)
    th[n, 0:n_td] = np.mod(th[n, 0:n_td], 2 * np.pi)
    th_p[n, 1:n_td] = np.diff(np.sign(th[n, 0:n_td])) > 0  # calulcate pulses

# Equations integration
for c in np.arange(0, n_cycles):
    th[:, n_td + 1:n_tp] = 0
    th_p[:, n_td + 1:n_tp] = 0

    if c < n_cycles:  # total number os steps in this cycle
        n_tpc = n_tp  # normal nr of steps 10000
    else:
        n_tpc = n_t_total - (n_cycles - 1) * n_tcycle + n_td

    for t in np.arange(n_td, n_tpc - 1):
        dth = squeeze(omegas) + sig_noise * np.random.randn(N)
        for n in np.arange(0, N):
            for p in np.arange(0, N):
                if C[n, p] > 0:
                    dth[n] = dth[n] + C[n, p] * np.sin(th_p[p, t - stepsDelay[n, p]] - th_p[n, t])

        th[:, t + 1] = np.mod(th[:, t] + dth, 2 * np.pi)
        th_p[:, t + 1] = (np.sign(th[:, t + 1]-pi) - np.sign(th[:, t]-pi)) > 0
    ni = (c - 1) * n_tcs
    ns = np.ceil((n_tpc - n_td) / sampling)
    #save everything to disk. THis is a RAM ISSUE!
    scipy.io.savemat('test_%s.mat' % c, {'th': th[:, n_td:n_tpc-1:sampling], 'p': th_p[:, n_td:n_tpc-1:sampling]})
    th[:, 1:n_td] = th[:, n_tp - n_td + 1:n_tp]
    th_p[:, 1:n_td] = th_p[:, n_tp - n_td + 1:n_tp]
    if np.mod(c, 10) == 0:
        print 'Cycle = ' + str(c) + ' of ' + str(n_cycles)
