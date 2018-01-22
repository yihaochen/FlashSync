#!/usr/bin/env python
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import yt
yt.enable_parallelism()
yt.mylog.setLevel('INFO')
from itertools import chain
from yt import FITSImageData
from yt.visualization.volume_rendering.off_axis_projection import \
        off_axis_projection
from yt.visualization.fits_image import\
        construct_image,\
        FITSProjection,\
        FITSOffAxisProjection
from astropy.wcs import WCS
from synchrotron.yt_synchrotron_emissivity import\
        setup_part_file,\
        write_synchrotron_hdf5,\
        synchrotron_filename,\
        synchrotron_fits_filename,\
        StokesFieldName


#dir = '/d/d5/ychen/2015_production_runs/0204_h0_10Myr'
dir = './'
#dir = '/home/ychen/data/0only_0605_h0/'
#dir = '/home/ychen/data/0only_1022_h1_10Myr/'
#dir = '/d/d5/ychen/2015_production_runs/1022_h1_10Myr'
#ts = yt.DatasetSeries(os.path.join(dir,'*_hdf5_plt_cnt_0640'), parallel=1, setup_function=setup_part_file)
#ts = yt.DatasetSeries(os.path.join(dir,'*_hdf5_plt_cnt_0910'), parallel=1, setup_function=setup_part_file)
try:
    ind = int(sys.argv[1])
    ts = yt.DatasetSeries(os.path.join(dir,'data/*_hdf5_plt_cnt_%03d0' % ind), parallel=1, setup_function=setup_part_file)
except IndexError:
    ts = yt.DatasetSeries(os.path.join(dir,'data/*_hdf5_plt_cnt_???0'), parallel=5, setup_function=setup_part_file)

mock_observation = False

if mock_observation:
    # Assumed distance to the object
    dist_obj = 165.95*yt.units.Mpc
    # Assumed coordinate of the object
    coord = [229.5, 42.82]

#nus = [(nu, 'MHz') for nu in chain(range(100,200,25), range(200,900,50), range(900,1500,100))]
nus = [(nu, 'MHz') for nu in [100,150,300,600,1400,8000]]

zoom_fac = 8
proj_axis = [1,0,2]
#proj_axis = 'x'
ptype = 'lobe'
gc = 32
maindir = os.path.join(dir, 'cos_synchrotron_QU_nn_%s/' % ptype)
fitsdir = 'fits_obs/' if mock_observation else 'fits/'
fitsdir = os.path.join(maindir, fitsdir)

if yt.is_root():
    for subdir in [maindir, fitsdir]:
        if not os.path.exists(subdir):
            os.mkdir(subdir)

for ds in ts.piter():
    fields = []
    ds_sync = yt.load(synchrotron_filename(ds, extend_cells=gc))
    flist = ds_sync.field_list

    width = ds_sync.domain_width[1:]/zoom_fac
    #res = ds_sync.domain_dimensions[1:]*ds_sync.refine_by**ds_sync.index.max_level//zoom_fac
    res = [512, 1024] if zoom_fac == 8 else [1024, 2048]

    for nu in nus:
        stokes = StokesFieldName(ptype, nu, proj_axis, field_type='flash')
        if mock_observation:
            fields.append(stokes.I)
        else:
            fields += stokes.IQU
        for field in stokes.IQU:
            ds_sync.field_info[field].units = 'Jy/cm/arcsec**2'
            ds_sync.field_info[field].output_units = 'Jy/cm/arcsec**2'
    if mock_observation:
        # Setting up mock observation FITS
        #  - Configure wcs coordinate
        #  - Convert the unit from Jy/arcsec**2 to Jy/beam
        rad = yt.units.rad
        cdelt1 = (width[0]/dist_obj/res[0]*rad).in_units('deg')
        cdelt2 = (width[1]/dist_obj/res[1]*rad).in_units('deg')

        # Setting up wcs header
        w = WCS(naxis=2)
        # reference pixel coordinate
        w.wcs.crpix = [res[0]/2,res[1]/2]
        # sizes of the pixel in degrees
        w.wcs.cdelt = [cdelt1.base, cdelt2.base]
        # converting ra and dec into degrees
        w.wcs.crval = coord
        # the units of the axes are in degrees
        w.wcs.cunit = ['deg']*2
        w.wcs.equinox = 2000
        wcs_header = w.to_header()

        # Assuming beam area = 1 pixel^2
        beam_area = cdelt1*cdelt2
        beam_axis = np.sqrt(beam_area/2/np.pi)*2*np.sqrt(2*np.log(2))
        # Major and minor beam axes
        beam_axis = float(beam_axis.in_units('deg').v)

        header_dict = {
                   'CTYPE1': 'RA---SIN',
                   'CTYPE2': 'DEC--SIN',
                   'CROTA1': (0, 'Rotation in degrees.'),
                   'CROTA2': (0, 'Rotation in degrees.'),
                   'CTYPE3': 'FREQ',
                   'CUNIT3': 'Hz',
                   'BMAJ': (beam_axis, 'Beam major axis (deg)'),
                   'BMIN': (beam_axis, 'Beam minor axis (deg)'),
                   'BPA': (0.0, 'Beam position angle (deg)')
                  }

        if proj_axis in ['x', 'y', 'z']:
            # On-Axis Projections
            prj = ds_sync.proj(stokes.I, proj_axis)
            buf = prj.to_frb(width[0], res, height=width[1])
        else:
            # Off-Axis Projections
            buf = {}
            width = ds_sync.coordinates.sanitize_width(proj_axis, width, (1.0, 'unitary'))
            wd = tuple(w.in_units('code_length').v for w in width)
            for field in fields:
                buf[field] = off_axis_projection(ds_sync, [0,0,0], proj_axis, wd,
                                res, field, north_vector=[1,0,0], num_threads=0).swapaxes(0,1)
        fits_image = FITSImageData(buf, fields=fields, wcs=w)
        for nu in nus:
            stokes = StokesFieldName(ptype, nu, proj_axis, field_type='flash')
            field = stokes.I[1]
            fits_image[field].data.units.registry.add('beam', float(beam_area.in_units('rad**2').v),
                              dimensions=yt.units.dimensions.solid_angle, tex_repr='beam')
            fits_image.set_unit(field, 'Jy/beam')
            nu = yt.YTQuantity(*nu)
            header_dict.update({
                   'OBJECT': 'Simulation %i %s' % (nu.v, nu.units),
                   'CRVAL3': int(nu.in_units('Hz').v)
                    })
            fits_image[field].header.update(header_dict)
            fits_image[field].header.update(wcs_header)
    else:
        # Use stock yt FITSProjection and FITSOffAxisProjection
        if proj_axis in ['x', 'y', 'z']:
            fits_image = FITSProjection(ds_sync, proj_axis, fields,
                        center=[0,0,0], width=width, image_res=res)
        else:
            fits_image = FITSOffAxisProjection(ds_sync, proj_axis, fields,
                         center=[0,0,0], north_vector=[1,0,0], width=width, image_res=res)

    #    fits_proj = FITSProjection(ds_sync, proj_axis, fields,
    #            center=[0,0,0], width=width, image_res=res)
    #else:
    #    fits_proj = FITSOffAxisProjection(ds_sync, proj_axis, fields,
    #            center=[0,0,0], north_vector=[1,0,0], width=width, image_res=res)
    fitsfname = synchrotron_fits_filename(ds, dir, ptype, proj_axis, mock_observation)
    fits_image.writeto(fitsfname, clobber=True)
