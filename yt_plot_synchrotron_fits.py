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
from tools import setup_cl

dir = './'
try:
    ind = int(sys.argv[1])
    ts = yt.DatasetSeries(os.path.join(dir,'data/*_hdf5_plt_cnt_%04d' % ind), parallel=1, setup_function=setup_part_file)
except IndexError:
    ts = yt.DatasetSeries(os.path.join(dir,'data/*_hdf5_plt_cnt_????'), parallel=8, setup_function=setup_part_file)

mock_observation = True

#nus = [(nu, 'MHz') for nu in [100,300,600,1400,8000]]
nus = [(nu, 'MHz') for nu in [100,1400,8000]]

zoom_fac = 4
#proj_axis = [1,0,2]
proj_axis = 'x'
ptype = 'lobe'
gc = 8
maindir = os.path.join(dir, 'synchrotron_%s/' % ptype)
fitsobsdir = 'fits_obs/'
fitsobsdir = os.path.join(maindir, fitsobsdir)
fitsdir = 'fits/'
fitsdir = os.path.join(maindir, fitsdir)

# Assumed distance to the object
dist_obj = 500*yt.units.Mpc
# Assumed coordinate of the object
#coord = [229.5, 42.82]
coord = [0, 0]

if yt.is_root():
    for subdir in [maindir, fitsdir, fitsobsdir]:
        if not os.path.exists(subdir):
            os.mkdir(subdir)

for ds in ts.piter():
    if '0000' in ds.basename: continue
    color, label = setup_cl([os.path.abspath(ds.directory)])
    ds_sync = yt.load(synchrotron_filename(ds, extend_cells=gc))
    flist = ds_sync.field_list

    width = ds_sync.domain_width[1:]/zoom_fac
    #res = ds_sync.domain_dimensions[1:]*ds_sync.refine_by**ds_sync.index.max_level//zoom_fac
    fields = []
    fieldsobs = []
    for nu in nus:
        stokes = StokesFieldName(ptype, nu, proj_axis, field_type='flash')
        fieldsobs.append(stokes.I)
        fields += stokes.IQU
        for field in stokes.IQU:
            try:
                ds_sync.field_info[field].units = 'Jy/cm/arcsec**2'
                ds_sync.field_info[field].output_units = 'Jy/cm/arcsec**2'
            except KeyError:
                raise KeyError("No field name %s in %s" % (field[1], ds.basename))

    if mock_observation:
        # Setting up mock observation FITS
        #  - Configure wcs coordinate
        #  - Convert the unit from Jy/arcsec**2 to Jy/beam
        res = [512, 1024]
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
        dist = float(dist_obj.in_units('Mpc').v)

        header_dict = {
                   'CTYPE1': 'RA---SIN',
                   'CTYPE2': 'DEC--SIN',
                   'CROTA1': (0, 'Rotation in degrees.'),
                   'CROTA2': (0, 'Rotation in degrees.'),
                   'CTYPE3': 'FREQ',
                   'CUNIT3': 'Hz',
                   'BMAJ': (beam_axis, 'Beam major axis (deg)'),
                   'BMIN': (beam_axis, 'Beam minor axis (deg)'),
                   'BPA': (0.0, 'Beam position angle (deg)'),
                   'DISTANCE': (dist, 'Assumed distance to the object (Mpc)')
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
                   'OBJECT': 'Simulation %s - %.1f Myr - %i %s'
                    % (list(label.values())[0], ds.current_time.in_units('Myr'), nu.v, nu.units),
                   'CRVAL3': int(nu.in_units('Hz').v)
                    })
            fits_image[field].header.update(header_dict)
            fits_image[field].header.update(wcs_header)
        fitsfname = synchrotron_fits_filename(ds, dir, ptype, proj_axis, mock_observation)
        fits_image.writeto(fitsfname, clobber=True)
        # End of if mock_observation


    # Use stock yt FITSProjection and FITSOffAxisProjection
    res = [1024, 2048]
    if proj_axis in ['x', 'y', 'z']:
        fits_image = FITSProjection(ds_sync, proj_axis, fields,
                    center=[0,0,0], width=width, image_res=res)
    else:
        fits_image = FITSOffAxisProjection(ds_sync, proj_axis, fields,
                     center=[0,0,0], north_vector=[1,0,0], width=width, image_res=res)
    fitsfname = synchrotron_fits_filename(ds, dir, ptype, proj_axis)
    fits_image.writeto(fitsfname, clobber=True)

