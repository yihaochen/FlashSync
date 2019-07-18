# FlashSync

Scripts to generate synchrotron maps from the customized FLASH simulation.

## How to use

### 0. You will need:
1. FLASH plot and particle files
    * `XXX_plt_cnt_####`
    * `XXX_part_####`
2. python installation, including `h5py` and `astropy`
3. yt installation
4. MPI and HDF5 installaion
5. The scripts in this repository

Everything should be setup on cetus. You will need to use my Anaconda
environment on cetus.

```
source /home/ychen/anaconda3/etc/profile.d/conda.sh
```

Then you need to make symlinks of the three scripts to the directory
where the FLASH outputs are located `yt_write_synchrotron_hdf5.py`,
`yt_plot_synchrotron_fits.py`, and `yt_plot_synchrotron_from_its.py`.

### 1. Calculate and write the synchrotron emissivity in each cell to an HDF5 file

Set the frequencies `nus`, projection axis `proj_axis` (could be `x`, `y`, `z`,
or a vector specifying the normal direction, e.g. `[0,1,2]` in the script
`yt_write_synchrotron_hdf5.py`.

```
python yt_write_synchrotron_hdf5.py [file#]
```

This step will take the longest time. So it would be a good idea to run it in
parallel. This also take a lot of memory for each process.

For example:

```
mpirun -hostfile ~/hostfile/hostfile -np 32 python yt_write_synchrotron_hdf5.py 2200
```

This will generate a hdf5 file `XXX_plt_cnt_####_synchrotron_peak_gc8`.


### 2. Make a projection plot to a fits file

Set the same frequencies `nus` and projection axis `proj_axis` as in the
previous step in `yt_plot_synchrotron_fits.py`. Additionally, set the
resolution `res` and `res_obs` of the output image and a zoom-in factor
`zoom_fac`. For mock observation fits, set the distance `dist_obj` and
the coordinate `coord` [RA, Dec] to the object.

```
python yt_plot_synchrotron_fits.py [file#]
```

This will generate fits file in `synchortron_lobe/fits` and the
mock observation fits file in `synchrotron_lobe/fits_obs`.


### 3. Make images from the fits file

Set the same frequencies `nus` and projection axis `proj_axis` as in the
previous step in `yt_plot_synchrotron_fits.py`. You can also fine-tune the
output figures in the code, e.g. field of view, colormap, or range of
the emissivity. There is also option to do a convolved image.

```
python yt_plot_synchrotron_from_its.py [file#]
```

This can generate emissivity, polarization, and spectral index figures.
