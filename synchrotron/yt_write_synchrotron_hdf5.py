#!usr/bin/env python
import sys
import os
import yt
yt.mylog.setLevel("INFO")
import yt_synchrotron_emissivity as sync

yt.enable_parallelism(suppress_logging=True)

dir = './'
ptype = 'lobe'
proj_axis = 'x'
#proj_axis = [1,0,2]
extend_cells = 8
nus = [(nu, 'MHz') for nu in [100, 1400, 8000]]

try:
    ind = int(sys.argv[1])
    #ts = yt.DatasetSeries(os.path.join(dir,'*_hdf5_plt_cnt_%04d' % ind), parallel=1, setup_function=sync.setup_part_file)

    # This works for 1 file at a time. Thus parallel=1
    ts = yt.DatasetSeries(os.path.join(dir,'data/*_hdf5_plt_cnt_%04d' % ind), parallel=1)
except IndexError:
    ts = yt.DatasetSeries(os.path.join(dir,'data/*_hdf5_plt_cnt_????'), parallel=1)

for ds in ts.piter():
    if '0000' in ds.basename: continue
    for nu in nus:
        # The two projection axes cannot be completed at the same time
        # Remember to comment out one of the following lines

        pars = sync.add_synchrotron_dtau_emissivity(ds, ptype=ptype, nu=nu,
                                           proj_axis=proj_axis, extend_cells=extend_cells)
        # Field names that we are going to write to the new hdf5 file
        stokes = sync.StokesFieldName(ptype, nu, proj_axis)
        # Take only the field name, discard "deposit" field type
        write_fields = [f for ftype, f in stokes.IQU]

        # Do the actual calculation and write to HDF5
        sync.write_synchrotron_hdf5(ds, write_fields, extend_cells=extend_cells)#, sanitize_fieldnames=True)
