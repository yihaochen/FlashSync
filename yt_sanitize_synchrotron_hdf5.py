#!usr/bin/env python
# This script is used to removed all "1.4 GHz" fields in the hdf5 file
import sys
import os
import yt
yt.mylog.setLevel("INFO")
import h5py
import synchrotron.yt_synchrotron_emissivity as sync

yt.enable_parallelism(suppress_logging=True)

dir = './data/'

try:
    ind = int(sys.argv[1])
    ts = yt.DatasetSeries(os.path.join(dir,'*_hdf5_plt_cnt_%03d0' % ind), parallel=1, setup_function=sync.setup_part_file)
except IndexError:
    ts = yt.DatasetSeries(os.path.join(dir,'*_hdf5_plt_cnt_???0'), parallel=1, setup_function=sync.setup_part_file)


for ds in ts.piter():
    stokes = sync.StokesFieldName('lobe', (1.4, 'GHz'), 'x')
    sfname = sync.synchrotron_file_name(ds, extend_cells=32)
    with h5py.File(sfname, 'a') as h5f:
        for field in stokes.IQU:
            if field[1] in h5f.keys():
                del h5f[field[1]]
        for k in h5f.keys():
            print(k)
        print('unknown names', h5f['unknown names'].value)

