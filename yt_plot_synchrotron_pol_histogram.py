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
from scipy.ndimage import gaussian_filter

yt.mylog.setLevel('ERROR')

dirs = [
        '/home/ychen/data/00only_0529_h1',\
        '/home/ychen/data/00only_0605_hinf',\
        '/home/ychen/data/00only_0605_h0'
        ]
regex = 'MHD_Jet*_hdf5_plt_cnt_??00'
files = None
proj_axis = 'x'
north_vector = [0,0,1]
ptype = 'lobe'
nus = [(150, 'MHz'), (1.4, 'GHz')]
zoom_fac = 8
gc = 32
sigma = 4


def rescan(dir, printlist=False):
    files = util.scan_files(dir, regex=regex, walk=True, printlist=printlist, reverse=False)
    return files


def worker_fn(file):
    ds = yt.load(file.fullpath)
    #for nu in nus:
    #    write_synchrotron_hdf5(ds, ptype, nu, proj_axis)#, extend_cells=None)

    if not os.path.isfile(synchrotron_file_name(ds, extend_cells=gc)):
        return ds.directory, 'sync file not found %s' % synchrotron_file_name(ds, extend_cells=gc)
    maindir = os.path.join(file.pathname, 'cos_synchrotron_QU_nn_%s/' % ptype)
    if proj_axis != 'x':
        maindir = os.path.join(maindir, '%i_%i_%i' % tuple(proj_axis))
        histdir = os.path.join(maindir, 'histogram_gaussian%i_%i%i%i' % (sigma, *proj_axis))
    else:
        histdir = os.path.join(maindir, 'histogram_gaussian%i_%s' % (sigma, proj_axis))

    ds_sync = yt.load(synchrotron_file_name(ds, extend_cells=gc))
    width = ds_sync.domain_width[1:]/zoom_fac
    res = ds_sync.domain_dimensions[1:]*ds_sync.refine_by**ds_sync.index.max_level//zoom_fac//2
    psi, frac = {}, {}
    I_bin = {}
    for nu in nus:
        stokes = StokesFieldName(ptype, nu, proj_axis, field_type='flash')
        if proj_axis == 'x':
            proj = yt.ProjectionPlot(ds_sync, proj_axis, stokes.IQU, center=[0,0,0],
                width=width).set_buff_size(res)
        else:
            proj = yt.OffAxisProjectionPlot(ds_sync, proj_axis, stokes.IQU ,width=width,
                                       north_vector=north_vector).set_buff_size(res)

        frb_I = proj.frb.data[stokes.I].v
        frb_Q = proj.frb.data[stokes.Q].v
        frb_U = proj.frb.data[stokes.U].v
        #null = plt.hist(np.log10(arri.flatten()), range=(-15,3), bins=100)


        frb_I = gaussian_filter(frb_I, sigma)
        frb_Q = gaussian_filter(frb_Q, sigma)
        frb_U = gaussian_filter(frb_U, sigma)

        factor = 1
        nx = res[0]//factor
        ny = res[1]//factor

        I_bin[nu] = frb_I.reshape(ny, factor, nx, factor).sum(3).sum(1)
        Q_bin = frb_Q.reshape(ny, factor, nx, factor).sum(3).sum(1)
        U_bin = frb_U.reshape(ny, factor, nx, factor).sum(3).sum(1)

        # angle between the polarization and horizontal axis
        # (or angle between the magnetic field and vertical axis
        psi[nu] = 0.5*np.arctan2(U_bin, Q_bin)
        frac[nu] = np.sqrt(Q_bin**2+U_bin**2)/I_bin[nu]

        fig = plt.figure(figsize=(8,16))
        i_plot = fig.add_subplot(111)
        i_plot.imshow(np.log10(frb_I+1e-2), vmin=-1, vmax=1, origin='lower')

        xx0, xx1 = i_plot.get_xlim()
        yy0, yy1 = i_plot.get_ylim()
        X,Y = np.meshgrid(np.linspace(xx0,xx1,nx,endpoint=True),
                  np.linspace(yy0,yy1,ny,endpoint=True))

        mask = I_bin[nu] < 0.1

        frac[nu][mask] = 0
        psi[nu][mask] = 0

        pixX = frac[nu]*np.cos(psi[nu]) # X-vector 
        pixY = frac[nu]*np.sin(psi[nu]) # Y-vector

        # keyword arguments for quiverplots
        quiveropts = dict(headlength=0, headwidth=1, pivot='middle')
        i_plot.quiver(X, Y, pixX, pixY, scale=8, **quiveropts)
        nu_str = '%.1f%s' % nu
        fig.savefig(histdir + '/' + ds.basename + '_I_%s.png' % nu_str)

    fig = plt.figure(figsize=(16,4))

    def plot_polarization_histogram(frac, psi, I_bin, fig=None, label=None):

        if not fig:
            fig = plt.figure(figsize=(16,4))

        ax1 = fig.axes[0]
        null = ax1.hist(frac[I_bin.nonzero()].flatten()*100, range=(0,80),
                        bins=40, alpha=0.5,
                        weights=I_bin[I_bin.nonzero()].flatten(),
                        normed=True)
        ax1.set_xlabel('Polarization fraction (%)')
        ax1.set_xlim(0, 80)

        ax2 = fig.axes[1]
        null = ax2.hist(psi[I_bin.nonzero()].flatten(), bins=50,
                        range=(-0.5*np.pi, 0.5*np.pi), alpha=0.5,
                        weights=I_bin[I_bin.nonzero()].flatten(),
                        normed=True)
        x_tick = np.linspace(-0.5, 0.5, 5, endpoint=True)

        x_label = [r"$-\pi/2$", r"$-\pi/4$", r"$0$", r"$+\pi/4$", r"$+\pi/2$"]
        ax2.set_xlim(-0.5*np.pi, 0.5*np.pi)
        ax2.set_xticks(x_tick*np.pi)
        ax2.set_xticklabels(x_label)
        #ax2.set_title(ds.basename + '  %.1f %s' % nu)

        ax3 = fig.axes[2]
        null = ax3.hist(np.abs(psi[I_bin.nonzero()].flatten()), bins=25,
                        range=(0.0, 0.5*np.pi), alpha=0.5,
                        label=label)
        ax3.legend()
        ax3.set_xlim(0.0, 0.5*np.pi)
        ax3.set_xticks([x_tick[2:]*np.pi])
        ax3.set_xticks(x_tick[2:]*np.pi)
        ax3.set_xticklabels(x_label[2:])

        return fig

    fig = plt.figure(figsize=(16,4))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)


    for nu in nus:
        nu_str = '%.1f%s' % nu
        fig = plot_polarization_histogram(frac[nu], psi[nu], I_bin[nu], fig=fig, label=nu_str)

    ax1.set_title(file.pathname)
    ax2.set_title(ds.basename + '  %.1f %s' % nu + ' gaussian %i' % sigma)


    fig.savefig(histdir + '/' + ds.basename)

    return file.pathname, ds.basename[-4:]

def tasks_gen(dirs, i0):
    for dir in dirs:
        files = rescan(dir, False)[i0:]
        for file in files:
            yield file

def init():
    for dir in dirs:
        maindir = os.path.join(file.pathname, 'cos_synchrotron_QU_nn_%s/' % ptype)
        if proj_axis != 'x':
            maindir = os.path.join(maindir, '%i_%i_%i' % tuple(proj_axis))
            histdir = os.path.join(maindir, 'histogram_gaussian%i_%i%i%i' % (sigma, *proj_axis))
        else:
            histdir = os.path.join(maindir, 'histogram_gaussian%i_%s' % (sigma, proj_axis))
        for subdir in [maindir, histdir]:
            if not os.path.exists(subdir):
                os.mkdir(subdir)

i0 = int(sys.argv[1]) if len(sys.argv) > 1 else 0
tasks = tasks_gen(dirs, i0)

results = MPI_taskpull2.taskpull(worker_fn, tasks, initialize=init, print_result=True)
