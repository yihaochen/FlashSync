#!/usr/bin/env python
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'stixgeneral'
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['figure.figsize'] = (7,3.5)
import matplotlib.pyplot as plt
import yt
yt.enable_parallelism()
yt.mylog.setLevel('INFO')
from mpl_toolkits.axes_grid1 import make_axes_locatable
from synchrotron.yt_synchrotron_emissivity import\
        setup_part_file,\
        synchrotron_filename,\
        synchrotron_fits_filename,\
        StokesFieldName
from scipy.ndimage import gaussian_filter
from astropy.io import fits

dir = './'
try:
    ind = int(sys.argv[1])
    ts = yt.DatasetSeries(os.path.join(dir,'data/*_hdf5_plt_cnt_%04d' % ind), parallel=1)
except IndexError:
    ts = yt.DatasetSeries(os.path.join(dir,'data/*_hdf5_plt_cnt_????'), parallel=20, setup_function=setup_part_file)

format = 'png'

#proj_axis = [1,0,2]
proj_axis = 'x'
nus = [(nu, 'MHz') for nu in [100,300,600,1400,8000]]
ptype = 'lobe'
gc = 8
plot_emissivity_i= True
plot_polline     = True
plot_spectralind = True


maindir = os.path.join(dir, 'synchrotron_%s/' % ptype)
if proj_axis != 'x':
    maindir = os.path.join(maindir, '%i_%i_%i' % tuple(proj_axis))
polline = os.path.join(maindir, 'polline')
emissivity_i_dir = os.path.join(maindir, 'emissivity_i')
spectral_index_dir = os.path.join(maindir, 'spectral_index')
if yt.is_root():
    for subdir in [maindir, polline, emissivity_i_dir, spectral_index_dir]:
        if not os.path.exists(subdir):
            os.mkdir(subdir)


def setup_plot(plot, cbar_label):
    plot.set_xlim(120,-120)
    plot.set_ylim(-60,60)
    plot.set_xlabel('z (kpc)')
    plot.set_ylabel('y (kpc)')
    plot.tick_params(direction='in')
    divider = make_axes_locatable(plot)
    cax = divider.append_axes("right", size="3%", pad=0)
    cbar = plt.colorbar(im, cmap=cmap, cax=cax)
    cbar.set_label(cbar_label)
    cbar.ax.tick_params(direction='in', which='both', width=0.5, length=3)


def to_fname(proj_axis):
    if proj_axis == 'x':
        return '_x_'
    else:
        return '%i_%i_%i_' % tuple(proj_axis)


for ds in ts.piter():
    if '0000' in ds.basename: continue

    projs, frb_I, frb_Q, frb_U = {}, {}, {}, {}
    for nu in nus:
        norm = yt.YTQuantity(*nu).in_units('GHz').value**0.5

        fitsname = synchrotron_fits_filename(ds, dir, ptype, proj_axis)
        if not os.path.isfile(fitsname): continue
        hdulist = fits.open(fitsname)
        stokes = StokesFieldName(ptype, nu, proj_axis, field_type='flash')
        frb_I[nu] = hdulist[stokes.I[1]].data
        frb_Q[nu] = hdulist[stokes.Q[1]].data
        frb_U[nu] = hdulist[stokes.U[1]].data
        header = hdulist[stokes.I[1]].header
        xr = -header['CRPIX1']*header['CDELT1'] + header['CRVAL1']
        xl = (header['NAXIS1'] - header['CRPIX1'])*header['CDELT1'] + header['CRVAL1']
        yr = -header['CRPIX2']*header['CDELT2'] + header['CRVAL2']
        yl = (header['NAXIS2'] - header['CRPIX2'])*header['CDELT2'] + header['CRVAL2']
        ext = ds.arr([yr, yl, xr, xl], input_units='cm').in_units('kpc')


        # Convolved image arrary. Do nothing for now.
        convolved_image = frb_I[nu] + 1E-10
        convolved_image_q = frb_Q[nu]
        convolved_image_u = frb_U[nu]

        if plot_emissivity_i or plot_polline:
            fig = plt.figure()
            plot = fig.add_subplot(111)
            vmin = np.log10(1E-4/norm)
            vmax = np.log10(1E-2/norm)
            cmap = plt.cm.hot
            cmap.set_bad('#0a0000')
            im = plot.imshow(np.log10(convolved_image.transpose()), vmin=vmin, vmax=vmax,
                             extent=ext, origin='lower', cmap=cmap)

            plot.set_facecolor('#0a0000')
            clabel = r'log Synchrotron Intensity ($\frac{\rm{Jy}}{\rm{arcsec}^2}}$)'
            setup_plot(plot, clabel)
            #ticks = np.log10([3E-3,4E-3,5E-3,6E-3,7E-3,8E-3,9E-3,
            #                  1E-2,2E-2,3E-2,4E-2,5E-2,6E-2,7E-2,8E-2,9E-2,
            #                  0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,
            #                  1,2,3,4,5,6,7,8,9,
            #                  1E1,2E1])
            #cbar.ax.set_yticks(ticks)
            #cbar.ax.set_yticklabels(['','','','','','','',
            #             r'$10^{-2}$','','','','','','','','',
            #             r'$10^{-1}$','','','','','','','','',
            #             1,'','','','','','','','',
            #             r'$10^{1}$',''])


            #plot.text(0.95, 0.90, labels[i], horizontalalignment='right',
            #          color='w', transform=i_plot.transAxes)

            #p.set_font_size(9)

            plot.text(0.05, 0.90, '%.1f Myr' % ds.current_time.in_units('Myr'),
                      color='w', transform=plot.transAxes)
            plot.text(0.80, 0.90, '%i %s' % nu,
                      color='w', transform=plot.transAxes)

            plt.tight_layout()
            figfname = ds.basename
            figfname += to_fname(proj_axis)
            nu_str = '%i%s' % nu
            figfname += 'emissivity_i_%s_%s.%s' % (ptype, nu_str, format)
            plt.savefig(os.path.join(emissivity_i_dir, figfname), format=format, dpi=240)

        if plot_polline:
            # ranges of the axis. Note that the image is transposed
            yy0, yy1, xx0, xx1 = im.get_extent()

            # binning factor
            factor = [4, 4]

            # re-binned number of points in each axis
            nx_new = convolved_image.shape[1] // factor[0]
            ny_new = convolved_image.shape[0] // factor[1]

            # These are the positions of the quivers
            # Note that the image is transposed
            Y,X = np.meshgrid(np.linspace(xx0,xx1,nx_new,endpoint=True),
                              np.linspace(yy0,yy1,ny_new,endpoint=True))

            # bin the data
            I_bin = convolved_image.reshape(nx_new, factor[0], ny_new, factor[1]).sum(3).sum(1)
            Q_bin = convolved_image_q.reshape(nx_new, factor[0], ny_new, factor[1]).sum(3).sum(1)
            U_bin = convolved_image_u.reshape(nx_new, factor[0], ny_new, factor[1]).sum(3).sum(1)

            # polarization angle
            psi = 0.5*np.arctan2(U_bin, Q_bin)

            # polarization fraction
            frac = np.sqrt(Q_bin**2+U_bin**2)/I_bin

            # mask for low signal area
            mask = I_bin < 1E-3

            frac[mask] = 0
            psi[mask] = 0

            pixX = frac*np.cos(psi) # X-vector 
            pixY = frac*np.sin(psi) # Y-vector

            im.set_cmap('viridis')
            plot.set_facecolor('#440154')

            # keyword arguments for quiverplots
            quiveropts = dict(headlength=0, headwidth=1, headaxislength=0, pivot='middle', linewidth=0.2, width=0.001)
            Q = plot.quiver(X, Y, pixX, pixY, scale=48, **quiveropts)

            plot.quiverkey(Q, 0.93, 0.10, 0.5, '50%', color='lightgray', labelcolor='lightgray', labelpos='S')

            pollinefname = ds.basename
            pollinefname += to_fname(proj_axis)
            nu_str = '%i%s' % nu
            pollinefname += 'polarization_quiver_%s_%s.%s' % (ptype, nu_str, format)
            plt.savefig(os.path.join(polline, pollinefname), suffix=format, dpi=240)



    if plot_spectralind:
        sigma = 1
        nu1, nu2 = nus[0], nus[-1]
        I1 = gaussian_filter(frb_I[nu1], sigma)
        I2 = gaussian_filter(frb_I[nu2], sigma)
        alpha = np.log10(I2/I1)/np.log10(nu2[0]/nu1[0])
        alpha = np.ma.masked_where(I2<1E-6, np.array(alpha))
        fig = plt.figure()
        plot = fig.add_subplot(111)

        cmap = plt.cm.jet
        cmap.set_bad('navy')
        im = plot.imshow(alpha.transpose(), cmap=cmap, vmin=-1.0, vmax=-0.5,
                         extent=ext, origin='lower', aspect='equal')
        plot.set_facecolor('navy')
        #pickle.dump(projs, open(dir+'projs/%s_projs.pickle' % ds.basename, 'wb'))
        #projs[(1.4, 'GHz')].save_object('proj_1.4GHz', dir+'projs/'+ds.basename+'_projs.cpkl')
        #projs[(150, 'MHz')].save_object('proj_150MHz', dir+'projs/'+ds.basename+'_projs.cpkl')

        #dirnamesplit = ds.directory.strip('/').split('_')
        #if dirnamesplit[-1] in ['h1','hinf', 'h0']:
        #    if dirnamesplit[-1] == 'h1':
        #        x = 0.80
        #        sim_name = 'helical'
        #    else:
        #        x = 0.85
        #        sim_name = dirnamesplit[-1]
        #else:
        #    x = 0.80
        #    sim_name = dirnamesplit[-2] + '_' + dirnamesplit[-1]
        #plt.annotate(sim_name, (1,1), xytext=(x, 0.96),  textcoords='axes fraction',\
        #            horizontalalignment='left', verticalalignment='center')
        plot.text(0.05, 0.90, '%.1f Myr' % ds.current_time.in_units('Myr'),
                  color='w', transform=plot.transAxes)

        clabel = 'Spectral Index (%s) (%i%s/%i%s)' % ((ptype,)+nu2+nu1)
        setup_plot(plot, clabel)
        plt.tight_layout()
        sifname = ds.basename
        sifname += to_fname(proj_axis)
        sifname += 'proj_spectral_index.%s' % format
        plt.savefig(os.path.join(spectral_index_dir, sifname), format=format, dpi=240)

