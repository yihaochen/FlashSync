#!/usr/bin/env python
import os
import sys
import matplotlib
matplotlib.use('Agg')
import yt
yt.enable_parallelism()
yt.mylog.setLevel('INFO')

dir = './'

try:
    ind = int(sys.argv[1])
    ts = yt.DatasetSeries(os.path.join(dir,'data/*_hdf5_plt_cnt_%04d' % ind), parallel=1)
except IndexError:
    ts = yt.DatasetSeries(os.path.join(dir,'data/*_hdf5_plt_cnt_??00'), parallel=1)

zoom_fac = 4
#proj_axis = [1,0,2]
proj_axis = 'x'
ptype = 'lobe'
fields = ['magnetic_field_strength']

maindir = os.path.join(dir, 'synchrotron_%s/' % ptype)
fitsdir = 'magnetic_field/'
fitsdir = os.path.join(maindir, fitsdir)

if yt.is_root():
    for subdir in [maindir, fitsdir]:
        if not os.path.exists(subdir):
            os.mkdir(subdir)

for ds in ts.piter():
    print(ds.current_time.in_units('Myr'))
    flist = ds.field_list
    width = ds.domain_width[1:]/zoom_fac
    #res = ds.domain_dimensions[1:]*ds.refine_by**ds.index.max_level//zoom_fac
    res = [512, 1024] if zoom_fac == 8 else [1024, 2048]

    if proj_axis in ['x', 'y', 'z']:
        fits_image = yt.FITSSlice(ds, proj_axis, fields,
                    center=[0,0,0], width=width, image_res=res)
    else:
        fits_image = yt.FITSOffAxisSlice(ds, proj_axis, fields,
                     center=[0,0,0], north_vector=[1,0,0], width=width, image_res=res)

    fitsfname = ds.basename + '_' + proj_axis + '_%s.fits' % fields[0]
    fits_image.writeto(os.path.join(fitsdir,fitsfname), clobber=True)
