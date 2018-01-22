#!usr/bin/env python
import sys
import os
import yt
yt.mylog.setLevel("INFO")
import synchrotron.yt_synchrotron_emissivity as sync
from itertools import chain

yt.enable_parallelism(suppress_logging=True)

dir = './data/'

try:
    ind = int(sys.argv[1])
    ts = yt.DatasetSeries(os.path.join(dir,'*_hdf5_plt_cnt_%03d0' % ind), parallel=1, setup_function=sync.setup_part_file)
except IndexError:
    ts = yt.DatasetSeries(os.path.join(dir,'*_hdf5_plt_cnt_???0'), parallel=1, setup_function=sync.setup_part_file)

for ds in ts.piter():
    #if '0000' in ds.basename: continue
    #sync.write_synchrotron_hdf5(ds, 'jetp', (150, 'MHz'), 'x', extend_cells=0)
    #sync.write_synchrotron_hdf5(ds, 'lobe', (150, 'MHz'), 'x', extend_cells=32)
    #sync.write_synchrotron_hdf5(ds, 'lobe', (1.4, 'GHz'), 'x', extend_cells=32)
    #nus = chain(range(100,200,25), range(200,900,50), range(900,1500,100))
    nus = [100,150,300,600,1400,8000]
    for nu in [(nu, 'MHz') for nu in nus]:
    #for nu in [(150, 'MHz'), (1400, 'MHz')]:
        # The two projection axes cannot be completed at the same time
        # Remember to comment out one of the following lines
        #sync.write_synchrotron_hdf5(ds, 'lobe', nu, [1,0,2], extend_cells=32, sanitize_fieldnames=True)
        sync.write_synchrotron_hdf5(ds, 'lobe', nu, 'x', extend_cells=32, sanitize_fieldnames=True)
