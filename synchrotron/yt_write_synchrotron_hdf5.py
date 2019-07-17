#!usr/bin/env python
import sys
import os
import yt
yt.mylog.setLevel("INFO")
import synchrotron.yt_synchrotron_emissivity as sync
from itertools import chain

yt.enable_parallelism(suppress_logging=True)

dir = './data/'
ptype = 'lobe'
proj_axis = 'x'
#proj_axis = [1,0,2]
extend_cells = 8

try:
    ind = int(sys.argv[1])
    #ts = yt.DatasetSeries(os.path.join(dir,'*_hdf5_plt_cnt_%04d' % ind), parallel=1, setup_function=sync.setup_part_file)
    ts = yt.DatasetSeries(os.path.join(dir,'*_hdf5_plt_cnt_%04d' % ind), parallel=1)
except IndexError:
    ts = yt.DatasetSeries(os.path.join(dir,'*_hdf5_plt_cnt_????'), parallel=1)

for ds in ts.piter():
    #if '0000' in ds.basename: continue
    #sync.write_synchrotron_hdf5(ds, 'jetp', (150, 'MHz'), 'x', extend_cells=0)
    #sync.write_synchrotron_hdf5(ds, 'lobe', (150, 'MHz'), 'x', extend_cells=32)
    #sync.write_synchrotron_hdf5(ds, 'lobe', (1.4, 'GHz'), 'x', extend_cells=32)
    #nus = chain(range(100,200,25), range(200,900,50), range(900,1500,100))
    nus = [100, 300, 600, 1400, 8000]
    #nus = [100, 1400]
    for nu in [(nu, 'MHz') for nu in nus]:
    #for nu in [(150, 'MHz'), (1400, 'MHz')]:
        # The two projection axes cannot be completed at the same time
        # Remember to comment out one of the following lines
        #sync.write_synchrotron_hdf5(ds, 'lobe', nu, [1,0,2], extend_cells=32, sanitize_fieldnames=True)
        pars = sync.add_synchrotron_dtau_emissivity(ds, ptype=ptype, nu=nu,
                                           proj_axis=proj_axis, extend_cells=extend_cells)
        # Field names that we are going to write to the new hdf5 file
        stokes = sync.StokesFieldName(ptype, nu, proj_axis)
        # Take only the field name, discard "deposit" field type
        write_fields = [f for ftype, f in stokes.IQU]
        sync.write_synchrotron_hdf5(ds, write_fields, extend_cells=extend_cells)#, sanitize_fieldnames=True)
