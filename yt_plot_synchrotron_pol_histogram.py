#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
import MPI_taskpull2
import os
import sys
import yt
import numpy as np
import matplotlib.pyplot as plt
from yt_synchrotron_emissivity import *
import util
import logging
logging.getLogger('yt').setLevel(logging.ERROR)

dirs = ['/home/ychen/data/0only_0529_h1',\
        '/home/ychen/data/0only_0605_hinf',\
        '/home/ychen/data/0only_0605_h0']
regex = 'MHD_Jet*_hdf5_plt_cnt_[0-9][0-9]00'
files = None
proj_axis = 'x'
ptype = 'lobe'
nu = (150, 'MHz')

def rescan(dir, printlist=False):
    files = util.scan_files(dir, regex=regex, walk=True, printlist=printlist, reverse=False)
    return files


def worker_fn(file):
    ds = yt.load(file.fullpath)
    pars = add_synchrotron_pol_emissivity(ds, ptype=ptype, nu=nu, proj_axis=proj_axis)
    fieldi = ('deposit', ('nn_emissivity_i_%s_%%.1f%%s' % ptype) % nu)
    fieldq = ('deposit', ('nn_emissivity_q_%s_%%.1f%%s' % ptype) % nu)
    fieldu = ('deposit', ('nn_emissivity_u_%s_%%.1f%%s' % ptype) % nu)
    proj = yt.ProjectionPlot(ds, proj_axis, [fieldi, fieldq, fieldu], center=[0,0,0], width=((40,'kpc'),(80,'kpc')))

    frb_I = proj.frb.data[fieldi].v
    frb_Q = proj.frb.data[fieldq].v
    frb_U = proj.frb.data[fieldu].v
    #null = plt.hist(np.log10(arri.flatten()), range=(-15,3), bins=100)

    factor = 1
    nx = 800/factor
    ny = 800/factor

    I_bin = frb_I.reshape(nx, factor, ny, factor).sum(3).sum(1)
    Q_bin = frb_Q.reshape(nx, factor, ny, factor).sum(3).sum(1)
    U_bin = frb_U.reshape(nx, factor, ny, factor).sum(3).sum(1)

    # angle between the polarization and horizontal axis
    # (or angle between the magnetic field and vertical axis
    psi = 0.5*np.arctan2(U_bin, Q_bin)
    frac = np.sqrt(Q_bin**2+U_bin**2)/I_bin

    fig = plt.figure(figsize=(16,4))

    ax1 = fig.add_subplot(131)
    null = ax1.hist(frac[I_bin.nonzero()].flatten()*100, range=(0,80), bins=40)
    ax1.set_xlabel('Polarization fraction (%)')
    ax1.set_xlim(0, 80)
    ax1.set_title(file.pathname)

    ax2  = fig.add_subplot(132)
    null = ax2.hist(psi[I_bin.nonzero()].flatten(), bins=50, range=(-0.5*np.pi, 0.5*np.pi))
    x_tick = np.linspace(-0.5, 0.5, 5, endpoint=True)

    x_label = [r"$-\pi/2$", r"$-\pi/4$", r"$0$", r"$+\pi/4$", r"$+\pi/2$"]
    ax2.set_xlim(-0.5*np.pi, 0.5*np.pi)
    ax2.set_xticks(x_tick*np.pi)
    ax2.set_xticklabels(x_label)
    ax2.set_title(ds.basename + '  %.1f %s' % nu)

    ax3 = fig.add_subplot(133)
    null = ax3.hist(np.abs(psi[I_bin.nonzero()].flatten()), bins=25, range=(0.0, 0.5*np.pi))
    ax3.set_xlim(0.0, 0.5*np.pi)
    ax3.set_xticks([x_tick[2:]*np.pi])
    ax3.set_xticks(x_tick[2:]*np.pi)
    ax3.set_xticklabels(x_label[2:])

    maindir = os.path.join(file.pathname, 'pol_synchrotron_QU_nn_%s/' % ptype)
    histdir = os.path.join(maindir, 'histogram')
    fig.savefig(histdir + '/' + ds.basename)

    return file.pathname, ds.basename[-4:]

def tasks_gen(dirs, i0):
    for dir in dirs:
        files = rescan(dir, False)[i0:]
        for file in files:
            yield file

def init():
    for dir in dirs:
        maindir = os.path.join(dir, 'pol_synchrotron_QU_nn_%s/' % ptype)
        histdir = os.path.join(maindir, 'histogram')
        for subdir in [maindir, histdir]:
            if not os.path.exists(subdir):
                os.mkdir(subdir)

i0 = int(sys.argv[1]) if len(sys.argv) > 1 else 0
tasks = tasks_gen(dirs, i0)

results = MPI_taskpull2.taskpull(worker_fn, tasks, initialize=init, print_result=True)
