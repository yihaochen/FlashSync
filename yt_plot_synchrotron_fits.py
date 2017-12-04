#!/usr/bin/env python
import os
import sys
import matplotlib
matplotlib.use('Agg')
import yt
from yt_synchrotron_emissivity import *
yt.enable_parallelism()
yt.mylog.setLevel('INFO')
from yt import FITSProjection, FITSOffAxisProjection
from synchrotron.yt_synchrotron_emissivity import\
        setup_part_file,\
        write_synchrotron_hdf5,\
        synchrotron_file_name,\
        StokesFieldName


dir = '/d/d5/ychen/2015_production_runs/0204_h0_10Myr'
#dir = './'
#dir = '/home/ychen/data/0only_0605_h0/'
#dir = '/home/ychen/data/0only_1022_h1_10Myr/'
#dir = '/d/d5/ychen/2015_production_runs/1022_h1_10Myr'
#ts = yt.DatasetSeries(os.path.join(dir,'*_hdf5_plt_cnt_0640'), parallel=1, setup_function=setup_part_file)
#ts = yt.DatasetSeries(os.path.join(dir,'*_hdf5_plt_cnt_0910'), parallel=1, setup_function=setup_part_file)
try:
    ind = int(sys.argv[1])
    ts = yt.DatasetSeries(os.path.join(dir,'*_hdf5_plt_cnt_%03d0' % ind), parallel=1, setup_function=setup_part_file)
except IndexError:
    ts = yt.DatasetSeries(os.path.join(dir,'*_hdf5_plt_cnt_???0'), parallel=1, setup_function=setup_part_file)


zoom_fac = 8

proj_axis = [1,0,2]
#proj_axis = 'x'
ptype = 'lobe'
gc = 32
maindir = os.path.join(dir, 'cos_synchrotron_QU_nn_%s/' % ptype)
fitsdir = os.path.join(maindir, 'fits/')
if yt.is_root():
    for subdir in [maindir, fitsdir]:
        if not os.path.exists(subdir):
            os.mkdir(subdir)

for ds in ts.piter():
    width = ds.domain_width[1:]/zoom_fac
    res = ds.domain_dimensions[1:]*ds.refine_by**ds.index.max_level//zoom_fac//2

    #for nu in [(150, 'MHz')]:
    #for nu in [(150, 'MHz'), (1400, 'MHz')]:
    fields = []
    for nu in [(150, 'MHz'), (233, 'MHz'), (325, 'MHz'), (610, 'MHz'), (1400, 'MHz')]:
        stokes = StokesFieldName(ptype, nu, proj_axis, field_type='flash')
        fields.append(stokes.I)
    #pars = add_synchrotron_dtau_emissivity(ds, ptype=ptype, nu=nu, proj_axis=proj_axis, extend_cells=8)
    ds_sync = yt.load(synchrotron_file_name(ds, extend_cells=gc))
    ds_sync.field_list
    for field in fields:
        ds_sync.field_info[field].units = 'Jy/cm/arcsec**2'
        ds_sync.field_info[field].output_units = 'Jy/cm/arcsec**2'
    if proj_axis != 'x':
        fits_proj = FITSOffAxisProjection(ds_sync, proj_axis, fields,
                center=[0,0,0], north_vector=[1,0,0], width=width, image_res=res)
        fitsfname = 'synchrotron_%s_%i_%i_%i_%s.fits' %\
                (ptype, *proj_axis, ds.basename[-4:])
    else:
        fits_proj = FITSProjection(ds_sync, proj_axis, fields,
                center=[0,0,0], width=width, image_res=res)
        fitsfname = 'synchrotron_%s_%s.fits' % (ptype, ds.basename[-4:])
    fitsfname = os.path.join(fitsdir, fitsfname)
    fits_proj.writeto(fitsfname, clobber=True)
