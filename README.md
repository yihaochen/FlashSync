# FlashSync

Scripts to generate synchrotron maps from the customized FLASH simulation.

## How to use

### You will need:
1. FLASH plot and particle files
    * XXX_plt_cnt_####
    * XXX_part_####
2. yt installation
3. MPI and HDF5 installaion
4. The scripts in this repo

Everything should be setup on cetus. You will need to use my Anaconda environment on cetus.

```source /home/ychen/anaconda3/etc/profile.d/conda.sh```

Then you need to copy (or make symlinks) of the three scripts to the directory where the FLASH outputs are located
`yt_write_synchrotron_hdf5.py`, `yt_plot_synchrotron_fits.py`, and `yt_plot_synchrotron_from_its.py`.

1.Calculate and write the synchrotron emissivity in each cell to an HDF5 file
----
This step will take the longest time. So it would be a good idea to run it in parallel.

For example:

```mpirun -hostfile ~/hostfile/hostfile -np 32 python yt_write_synchrotron_hdf5.py 2200```

2.Make a projection plot to a fits file
----
```python yt_plot_synchrotron_fits.py [file#]```

2.Make images from the fits file
----
```python yt_plot_synchrotron_from_its.py [file#]```
