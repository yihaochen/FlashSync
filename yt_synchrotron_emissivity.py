import os
import sys
import numpy as np
import h5py
import itertools
import yt
from yt.fields.field_detector import FieldDetector
from yt.fields.derived_field import \
    ValidateSpatial
from yt.funcs import mylog, only_on_root
from yt.utilities.file_handler import HDF5FileHandler
from yt.utilities.parallel_tools.parallel_analysis_interface import \
    parallel_objects, \
    communication_system, \
    parallel_blocking_call, \
    get_mpi_type, \
    op_names
#from yt.utilities.parallel_tools.task_queue import dynamic_parallel_objects
from particles.particle_filters import *
import time
import re


class StokesFieldName(object):
    """
    Define field names for Stokes I, Q, and U for easy access.
    """
    def __init__(self, ptype, nu, proj_axis, field_type='deposit', fieldname_prefix='nn_emissivity'):
        # particle type
        self.ptype = ptype
        # frequency tuple
        self.nu = nu
        # frequency in string
        self.nu_str = '%.1f%s' % nu
        # projection axis could be x,y,z or a vector
        if type(proj_axis) is str:
            postfix = proj_axis
        elif type(proj_axis) is list:
            postfix = '%.1f_%.1f_%.1f' % tuple(proj_axis)
        self.I = (field_type, fieldname_prefix+'_i_%s_%s_%s' % (self.ptype, self.nu_str, postfix))
        self.Q = (field_type, fieldname_prefix+'_q_%s_%s_%s' % (self.ptype, self.nu_str, postfix))
        self.U = (field_type, fieldname_prefix+'_u_%s_%s_%s' % (self.ptype, self.nu_str, postfix))
        self.IQU = [self.I, self.Q, self.U]

    def display_name(self, IQU):
        return '%s NN Emissivity %s (%s)' % (self.nu_str, IQU, self.ptype)


def _jet_volume_fraction(field, data):
    from yt.units import g, cm, Kelvin
    mH = yt.utilities.physical_constants.mass_hydrogen
    k  = yt.utilities.physical_constants.boltzmann_constant

    rhoCore = data.ds.parameters['sim_rhocore']*g/cm**3
    rCore   = data.ds.parameters['sim_rcore']*cm
    densitybeta = data.ds.parameters['sim_densitybeta']
    Tout    = data.ds.parameters['sim_tout']*Kelvin
    Tcore   = data.ds.parameters['sim_tcore']*Kelvin
    rCoreT  = data.ds.parameters['sim_rcoret']*cm
    gammaICM= data.ds.parameters['sim_gammaicm']
    mu      = data.ds.parameters['sim_mu']

    if not isinstance(data, FieldDetector):
        data.set_field_parameter('center', (0,0,0))
    r = data['index', 'spherical_radius']

    density0 = rhoCore*(1.0 + (r/rCore)**2)**(-1.5*densitybeta)
    T0 = Tout*(1.0+(r/rCoreT)**3)/(Tout/Tcore+(r/rCoreT)**3)
    P0 = density0/mu/mH*k*T0
    icm_mass_fraction = 1.0 - data['flash', 'jet ']
    P = data['gas', 'pressure']
    density = data['gas', 'density']

    icm_volume_fraction = (P0/P)**(1/gammaICM)*icm_mass_fraction*density/density0

    icm_volume_fraction = np.where(icm_volume_fraction < 1.0, icm_volume_fraction, 1.0)

    # Cutoff where jet volume fraction is smaller than 5%
    icm_volume_fraction = np.where(icm_volume_fraction > 0.95, 1.0, icm_volume_fraction)

    return 1.0 - icm_volume_fraction


def add_synchrotron_dtau_emissivity(ds, ptype='lobe', nu=(1.4, 'GHz'),
                                    method='nearest_weighted', proj_axis='x',
                                    extend_cells=32):
    me = yt.utilities.physical_constants.mass_electron #9.109E-28
    c  = yt.utilities.physical_constants.speed_of_light #2.998E10
    e  = yt.utilities.physical_constants.elementary_charge #4.803E-10 esu

    gamma_min = yt.YTQuantity(10, 'dimensionless')
    # Index for electron power law distribution
    p = 2.0
    pol_ratio = (p+1.)/(p+7./3.)
    # Fitted constants for the approximated power-law + exponential spectra
    # Integral of 2*F(x) -> tot_const*(nu**-2)*exp(-nu/nuc)
    # 2*F(x) for the total intensity (parallel + perpendicular)
    tot_const = 4.1648

    stokes = StokesFieldName(ptype, nu, proj_axis)
    nu = yt.YTQuantity(*nu)

    if proj_axis=='x':
        los = [1.,0.,0.]
        xvec = [0., 1., 0.]
        yvec = [0., 0., 1.]
    elif proj_axis=='y':
        los = [0.,1.,0.]
        xvec = [0., 0., 1.]
        yvec = [1., 0., 0.]
    elif proj_axis=='z':
        los = [0.,0.,1.]
        xvec = [1., 0., 0.]
        yvec = [0., 1., 0.]
    elif type(proj_axis) is list :
        los = proj_axis
        if los[0] != 0.: # not perpendicular to z-axis
            xvec = [0., 1., 0.]
            yvec = [0., 0., 1.]
        # TODO: xvec and yvec for arbitrary proj_axis
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError
    los = np.array(los)
    xvec = np.array(xvec)
    yvec = np.array(yvec)
    los = los/np.sqrt(np.sum(los*los))

    # Update the particle file handler in yt; raise exception if not successful
    success = setup_part_file(ds)
    if not success:
        raise IOError

    if ('gas', 'jet_volume_fraction') not in ds.derived_field_list:
        ds.add_field(('gas', 'jet_volume_fraction'), function=_jet_volume_fraction,
                     display_name="Jet Volume Fraction", sampling_type='cell')

    def _gamc(field, data):
        # Density when the particle left the jet
        #den1 = data['particle_den1']
        dtau = data['particle_dtau']

        # The new cutoff gamma
        # Note that particle_dens could be negative due to quadratic interpolation!
        gamc = (np.abs(data['particle_dens'] / data['particle_den0']))**(1./3.) \
               / (dtau + np.finfo(np.float64).tiny)
        ind = np.where(gamc < 0.0)[0]
        if ind.shape[0] > 0:
            print(ind)
            print(gamc)

        return gamc

    pfname = 'particle_gamc_dtau'
    ds.add_field(pfname, function=_gamc, sampling_type='particle',
                 units='', force_override=True)

    #def _synchrotron_spec(field, data):
    #    # To convert from FLASH "none" unit to cgs unit, times the B field from FLASH by sqrt(4*pi)
    #    Bvec = np.array([data['particle_magx'],\
    #                     data['particle_magy'],\
    #                     data['particle_magz']])*np.sqrt(4.0*np.pi)
    #    Bvec = data.apply_units(Bvec, 'gauss')

    #    # Calculate sin(a), in which a is the pitch angle of the electrons relative to B field.
    #    # See _nn_emissivity_i for more comments
    #    cross = np.cross(los, Bvec, axisb=0)
    #    Bsina = np.sqrt(np.sum(cross*cross, axis=-1))
    #    Bsina = data.apply_units(Bsina, 'gauss')
    #    #B = np.sqrt(np.sum(Bvec*Bvec, axis=0))

    #    # Return for the FieldDetector; do nothing
    #    if isinstance(data, FieldDetector):
    #        return data['particle_dens']/data['particle_den1']**(1./3.)/ \
    #                (data['particle_dtau'])

    #    # Density when the particle left the jet
    #    den1 = data['particle_den1']
    #    dtau = data['particle_dtau']

    #    if np.any(dtau < 0.0):
    #        print('negative tau!')
    #        print(data)
    #        print(data['particle_tau'])
    #        print(dtau)

    #    # The new cutoff gamma
    #    # Note that particle_dens could be negative due to quadratic interpolation!
    #    gamc = (np.abs(data['particle_dens'] / den1))**(1./3.) \
    #           / (dtau + np.finfo(np.float64).tiny)
    #    ind = np.where(gamc <= 0.0)[0]
    #    if ind.shape[0] > 0:
    #        print(ind)
    #        print(gamc)

    #    #gamc = data[(ptype, 'particle_gamc')]

    #    # Cutoff frequency
    #    nuc = 3.0*gamc**2*e*Bsina/(4.0*np.pi*me*c)
    #    #nu = data.get_field_parameter("frequency", default=yt.YTQuantity(1.4, 'GHz'))

    #    # B**1.5 is taken from the grid data
    #    norm = 3.0/8.0*e**3.5/(c**2.5*me**1.5*(np.pi)**0.5)
    #    # P is taken from the grid data
    #    N0 = 3.0/me/c/c/(np.log(gamc/gamma_min))/yt.YTQuantity(4.*np.pi, 'sr')

    #    # Fix where the cutoff gamma < 0
    #    N0[ind] = 0.0

    #    return np.clip(N0*norm*nu**(-0.5)*np.exp(-nu/nuc), np.finfo(np.float64).tiny, None)

    ## particle field name
    #pfname = 'particle_sync_spec_%s' % stokes.nu_str
    #ds.add_field(pfname, function=_synchrotron_spec, sampling_type='particle',
    #             units='cm**(3/4)*s**(3/2)/g**(3/4)/sr', force_override=True)
    #try:
    ds.add_particle_filter(ptype)
    #except:
    #    raise NotImplementedError

    ###########################################################################
    ## Nearest Neighbor method
    ###########################################################################
    #fname_nn = ds.add_deposited_particle_field(
    #        (ptype, 'particle_sync_spec_%s' % stokes.nu_str), 'nearest', extend_cells=extend_cells)

    #deposit_field = 'particle_sync_spec_%s' % stokes.nu_str
    deposit_field = 'particle_gamc_dtau'

    sync_unit = ds.field_info[deposit_field].units
    if method == "nearest":
        field_name = "%s_nn_%s"
    elif method == "nearest_weighted":
        field_name = "%s_nnw_%s"
    else:
        raise NotImplementedError
    field_name = field_name % (ptype, deposit_field.replace('particle_', ''))
    ad = ds.all_data()
    #print(ad[ptype, "particle_position"])

    def _nnw_deposit_field(field, data):
        if isinstance(data, FieldDetector):
            jetfluid = data['velocity_magnitude'] > lobe_v
            d = data.deposit(data[ptype, "particle_position"],
                    deposit_field, method=method)
            d[jetfluid] = 0.0
            return d
        #pos = ad[ptype, "particle_position"]
        if ptype == 'lobe':
            # Calling other fields must preceed the more intensive
            # deposit function to prevent repeated iterations
            jetfluid = data['velocity_magnitude'] > lobe_v
        alldata = True
        if alldata:
            pos = ad[ptype, "particle_position"]
            # Deposit using the distance weighted log field
            fields = [np.log(ad[ptype, deposit_field])]
            fields = [np.ascontiguousarray(f) for f in fields]
        else:
            left_edge = np.maximum(data.LeftEdge-data.dds*extend_cells,\
                                   data.ds.domain_left_edge)
            right_edge = np.minimum(data.RightEdge+data.dds*extend_cells,\
                                    data.ds.domain_right_edge)
            box = data.ds.box(left_edge, right_edge)
            pos = box[ptype, "particle_position"]
            fields = [box[ptype, deposit_field]]
        d = data.deposit(pos, fields, method=method,
                         extend_cells=extend_cells)
        if np.all(d == 0.0):
            d = data.ds.arr(d, input_units=sync_unit)
        else:
            # Conver the log back to real value
            d = data.ds.arr(np.exp(d), input_units=sync_unit)
        if ptype == 'lobe':
            d[jetfluid] = 0.0
        return d

    fname_nn = ("deposit", field_name)
    ds.add_field(fname_nn,
        function=_nnw_deposit_field,
        sampling_type="cell",
        units=sync_unit,
        take_log=True,
        force_override=True,
        validators=[ValidateSpatial()])


    def _nn_emissivity_i(field, data):
        '''
        Emissivity using nearest neighbor. Integrate over line of sight to get intensity.
        '''
        Bvec = np.array([data[('gas', 'magnetic_field_x')],\
                         data[('gas', 'magnetic_field_y')],\
                         data[('gas', 'magnetic_field_z')]])

        # Calculate sin(a), in which a is the pitch angle of the electrons relative to B field.
        # We only see the radiation from electrons with pitch angles pointing to line of sight.
        cross = np.cross(los, Bvec, axisb=0)
        # B * sin(alpha) = (B * |(los x Bvec)|/|los|/|Bvec|)
        # = |(los x Bvec)|
        Bsina = np.sqrt(np.sum(cross*cross, axis=-1))
        Bsina = data.apply_units(Bsina, 'gauss')

        frac = data['gas', 'jet_volume_fraction']

        # fname_nn = ptype_nnw_gamc_dtau
        if ('flash', fname_nn) in data.ds.field_list:
            gamc = data[('flash', fname_nn)]
        else:
            gamc = data[('deposit', fname_nn)]
        bad_mask = gamc <= gamma_min
        # Cutoff frequency
        nuc = 3.0*gamc**2*e*Bsina/(4.0*np.pi*me*c)
        #nu = data.get_field_parameter("frequency", default=yt.YTQuantity(1.4, 'GHz'))

        norm = 3.0*Bsina**1.5/8.0*e**3.5/(c**2.5*me**1.5*(np.pi)**0.5)
        N0 = 3.0*data['gas', 'pressure']/me/c/c/(np.log(gamc/gamma_min))/yt.YTQuantity(4.*np.pi, 'sr')
        N0[bad_mask] = 0.0

        return np.clip(frac*N0*norm*tot_const*nu**(-0.5)*np.exp(-nu/nuc), np.finfo(np.float64).tiny, None)


    ds.add_field(stokes.I, function=_nn_emissivity_i, sampling_type='cell',
                 display_name=stokes.display_name('I'),
                 units='Jy/cm/arcsec**2', take_log=True,
                 force_override=True,
                 validators=[ValidateSpatial()])

    def _cos_theta(field, data):
        Bvec = np.stack([data[('gas', 'magnetic_field_x')],\
                         data[('gas', 'magnetic_field_y')],\
                         data[('gas', 'magnetic_field_z')]], axis=-1)
        Bproj = Bvec - np.expand_dims(np.inner(Bvec, los), -1)*los
        # cos = cos(theta), theta is the angle between projected B and xvec
        # Ignore invalid 0/0 warnings
        with np.errstate(invalid='ignore'):
            cos = np.inner(Bproj, xvec)/np.sqrt(np.sum(Bproj*Bproj, axis=-1))
        return cos

    ds.add_field('cos', function=_cos_theta, sampling_type='cell',
                 display_name='cos theta',
                 units='', take_log=False,
                 force_override=True)

    def _nn_emissivity_q(field, data):
        # pol_ratio = (perp - para) / (perp + para)
        # The minus accounts for the perpendicular polarization
        rtn = -data[stokes.I]*pol_ratio*(2*data['cos']**2-1.0)
        rtn[~np.isfinite(data['cos'])] = 0.0
        return rtn

    ds.add_field(stokes.Q, function=_nn_emissivity_q, sampling_type='cell',
                 display_name=stokes.display_name('Q'),
                 units='Jy/cm/arcsec**2', take_log=False,
                 force_override=True)


    def _nn_emissivity_u(field, data):
        sin = np.sqrt(1.0-data['cos']**2)
        rtn = -data[stokes.I]*pol_ratio*2*sin*data['cos']
        rtn[~np.isfinite(data['cos'])] = 0.0
        return rtn

    ds.add_field(stokes.U, function=_nn_emissivity_u, sampling_type='cell',
                 display_name=stokes.display_name('U'),
                 units='Jy/cm/arcsec**2', take_log=False,
                 force_override=True)

    return pfname, fname_nn, stokes.I, stokes.nu_str


def setup_part_file(ds):
    filename = os.path.join(ds.directory,ds.basename)
    updated_pfname = re.sub(r'_synchrotron_.*','',filename).replace('plt_cnt', 'part')+'_updated_peak'
    if os.path.exists(updated_pfname):
        ds._particle_handle = HDF5FileHandler(updated_pfname)
        ds.particle_filename = updated_pfname
        mylog.info('Changed particle files to: %s', ds.particle_filename)
        return True
    else:
        return False


def synchrotron_filename(ds, extend_cells=None):
    postfix = '_synchrotron_peak_gamc'
    postfix = postfix+'_gc%i' % extend_cells if extend_cells else postfix
    return os.path.join(ds.directory, ds.basename + postfix)

def synchrotron_fits_filename(ds, dir, ptype, proj_axis, mock_observation=False):
    if proj_axis in ['x', 'y', 'z']:
        fitsfname = 'synchrotron_%s_%s.fits' % (ptype, ds.basename[-4:])
    else:
        fitsfname = 'synchrotron_%s_%i_%i_%i_%s.fits' %\
                (ptype, *proj_axis, ds.basename[-4:])
    maindir = os.path.join(dir, 'cos_synchrotron_QU_nn_%s/' % ptype)
    fitsdir = 'fits_obs/' if mock_observation else 'fits/'
    fitsdir = os.path.join(maindir, fitsdir)
    return os.path.join(fitsdir, fitsfname)


def prep_field_data(ds, field, offset=1):
    """
    Prepare the grid data. Read the field data grid by grid, remove bad values.
    Return the numpy array with shape [ngrid, nx, ny, nz].
    """
    mylog.info('Calculating field: %s', field)
    comm = communication_system.communicators[-1]
    dtype = 'float64'
    # data.shape should be (ngrid, nxb, nyb, nzb)
    data = np.zeros([ds.index.num_grids, *ds.index.grid_dimensions[0]], dtype=dtype)
    # Go through all the grids in the index
    if comm.rank == 0:
        t0 = time.time()
        t1_prev = t0
    for g in parallel_objects(ds.index.grids, njobs=0):
        if comm.rank == 0:
            t1 = time.time()
            sys.stdout.write('\rWorking on Grid %7i / %7i - %7.3f s'
                             % (g.id, ds.index.num_grids, t1-t1_prev))
            t1_prev = t1
        # Print the grid if nan or inf is in it
        #if np.nan in g[field].v or np.inf in g[field].v:
        #    mylog.warning('Encountered non-physical values in %s', g) # g[field].v)
        # Calculate the field values in each grid
        # Use numpy nan_to_num to convert the bad values anyway
        # Transpose the array since the array in FLASH is fortran-ordered
        #data[g.id-offset] = np.nan_to_num(g[field].in_units('Jy/cm/arcsec**2').v.transpose())
        data[g.id-offset] = np.nan_to_num(g[field].v.transpose())
    if comm.rank == 0:
        sys.stdout.write(' - mpi_reduce')
        t2 = time.time()
    temp = data.copy()
    comm.comm.Reduce([temp,get_mpi_type(dtype)], [data,get_mpi_type(dtype)], op=op_names['sum'])
    if comm.rank == 0:
        t3 = time.time()
        sys.stdout.write(' - Done!\nGrid Calculation: %.1f MPI: %.1f Total: %.1f\n'
                         % (t2-t0, t3-t2, t3-t0))

    return data


@parallel_blocking_call
def write_synchrotron_hdf5(ds, write_fields, sanitize_fieldnames=False, extend_cells=None):
    """
    Calculate the emissivity of Stokes I, Q, and U in each cell. Write them
    to a new HDF5 file and copy metadata from the original HDF5 files.
    The new HDF5 file can then be loaded into yt and make plots.
    """
    # The new file name that we are going to write to
    sfname = synchrotron_filename(ds, extend_cells=extend_cells)

    h5_handle = ds._handle

    # Keep a list of the fields that were in the original hdf5 file
    orig_field_list = [field.decode() for field in h5_handle['unknown names'].value[:,0]]

    if not isinstance(write_fields, list):
        write_fields = [write_fields]

    comm = communication_system.communicators[-1]
    # Only do the IO in the master process
    if comm.rank == 0:
        exist_fields = []
        written_fields = []
        if os.path.isfile(sfname):
            with h5py.File(sfname, 'r') as h5file:
                # First check if the fields already exist
                if 'unknown names' in h5file.keys():
                    exist_fields = [f.decode('utf8') for f in h5file['unknown names'].value]
                for field in write_fields.copy():
                    #TODO: Check if the data are the same
                    if field in exist_fields and field in h5file.keys():
                        write_fields.remove(field)
                    elif field in h5file.keys():
                    # Some fields are not recorded by already in the dataset
                        write_fields.remove(field)
                        exist_fields.append(field)
                        sanitize_fieldnames = True
                mylog.info('Closing File: %s', h5file)
            mylog.info('Fields already exist: %s', sorted(exist_fields))
        else:
        # File does not exist
            pass


    # On all processes
    write_fields = comm.mpi_bcast(write_fields, root=0)

    if write_fields:
        mylog.info('Fields to be generated: %s', write_fields)
        if comm.rank == 0:
            wdata = {}
        for field in write_fields:
            mylog.debug('Preparing field: %s', field)
            # Here we do the actual calculation (in yt) and save the grid data
            data = prep_field_data(ds, field)
            # On master process only
            if comm.rank == 0:
                wdata[field] = data

        # On master process only
        if comm.rank == 0:
            with h5py.File(sfname, 'a') as h5file:
                mylog.info('Writing to %s', h5file)
                mylog.info('Fields to be written: %s', write_fields)
                for field in write_fields:
                    # Writing data to HDF5 file
                    mylog.info('Writing field: %s', field)
                    h5file.create_dataset(field, wdata[field].shape, wdata[field].dtype, wdata[field])
                    written_fields.append(field)
                del wdata

                # Go through all the items in the FLASH hdf5 file
                for name, v in h5_handle.items():
                    if name in itertools.chain(orig_field_list, h5file.keys()):
                        # Do not write the fields that were present in the FLASH hdf5 file
                        # or other metadata that had been copied in the destination hdf5 file
                        pass
                    elif name == 'unknown names':
                        pass
                    else:
                        # Keep other simulation information
                        h5file.create_dataset(v.name, v.shape, v.dtype, v.value)

    if write_fields or sanitize_fieldnames:
        if comm.rank == 0:
            with h5py.File(sfname, 'a') as h5file:
                # Add the new field names that have already been written
                all_fields = list(set(written_fields + exist_fields))
                # Remove the field that does not exist in the dataset
                for field in all_fields:
                    if field not in h5file.keys():
                        all_fields.remove(field)
                # We need to encode the field name to binary format
                bnames = [f.encode('utf8') for f in all_fields]
                # Create a new dataset for the field names
                if 'unknown names' in h5file.keys():
                    del h5file['unknown names']
                h5file.create_dataset('unknown names', data=bnames)
                mylog.info('Field List: %s', sorted(bnames))
