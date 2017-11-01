import os
import numpy as np
import h5py
import yt
from yt.fields.field_detector import FieldDetector
from yt.funcs import mylog, only_on_root
from yt.utilities.file_handler import HDF5FileHandler
from yt.utilities.parallel_tools.parallel_analysis_interface import \
    parallel_objects, \
    communication_system
from particles.particle_filters import *



class StokesFieldName(object):
    """
    Define field names for Stokes I, Q, and U for easy access.
    """
    def __init__(self, ptype, nu, field_type='deposit', fieldname_prefix='nn_emissivity'):
        # particle type
        self.ptype = ptype
        # frequency tuple
        self.nu = nu
        # frequency in string
        self.nu_str = '%.1f%s' % nu
        self.I = (field_type, fieldname_prefix+'_i_%s_%s' % (self.ptype, self.nu_str))
        self.Q = (field_type, fieldname_prefix+'_q_%s_%s' % (self.ptype, self.nu_str))
        self.U = (field_type, fieldname_prefix+'_u_%s_%s' % (self.ptype, self.nu_str))
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

    return 1.0 - icm_volume_fraction


def add_synchrotron_pol_emissivity(ds, ptype='jnsp', nu=(1.4, 'GHz'), method='nearest', proj_axis='x'):
    me = yt.utilities.physical_constants.mass_electron #9.109E-28
    c  = yt.utilities.physical_constants.speed_of_light #2.998E10
    e  = yt.utilities.physical_constants.elementary_charge #4.803E-10 esu

    gamma_min = yt.YTQuantity(10, 'dimensionless')
    # Index for electron power law distribution
    p = 2.0
    pol_ratio = (p+1.)/(p+7./3.)
    # Fitted constants for the approximated power-law + exponential spectra
    # Integral of 2*F(x) -> tot_const*(nu**-2)*exp(-nu/nuc)
    tot_const = 4.1648


    nu = yt.YTQuantity(*nu)
    nu_str = str(nu).replace(' ', '')

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
    elif proj_axis is list: los = proj_axis
    else: raise IOError
    los = np.array(los)
    xvec = np.array(xvec)
    yvec = np.array(yvec)
    los = los/np.sqrt(np.sum(los*los))

    fnames = []
    if ('gas', 'jet_volume_fraction') not in ds.derived_field_list:
        ds.add_field(('gas', 'jet_volume_fraction'), function=_jet_volume_fraction,
                     display_name="Jet Volume Fraction", sampling_type='cell')

    def _synchrotron_spec(field, data):
        # ptype needs to be 'io' (only in this function)
        ptype = 'io'
        # To convert from FLASH "none" unit to cgs unit, times the B field from FLASH by sqrt(4*pi)
        #B = np.sqrt(data[(ptype, 'particle_magx')]**2
        #           +data[(ptype, 'particle_magy')]**2
        #           +data[(ptype, 'particle_magz')]**2)*np.sqrt(4.0*np.pi)
        #B = data.apply_units(B, 'gauss')
        if isinstance(data, FieldDetector):
            return (data[ptype, 'particle_magx'] + \
                    data[ptype, 'particle_magy'] + \
                    data[ptype, 'particle_magz']) \
                   /data[ptype, 'particle_gamc']
        Bvec = np.array([data[(ptype, 'particle_magx')],\
                         data[(ptype, 'particle_magy')],\
                         data[(ptype, 'particle_magz')]])*np.sqrt(4.0*np.pi)
        Bvec = data.apply_units(Bvec, 'gauss')
        #B = np.sqrt(np.sum(Bvec*Bvec, axis=0))

        cross = np.cross(los, Bvec, axisb=0)
        Bsina = np.sqrt(np.sum(cross*cross, axis=-1))
        Bsina = data.apply_units(Bsina, 'gauss')


        gamc = data[(ptype, 'particle_gamc')]

        # Cutoff frequency
        nuc = 3.0*gamc**2*e*Bsina/(4.0*np.pi*me*c)
        #nu = data.get_field_parameter("frequency", default=yt.YTQuantity(1.4, 'GHz'))

        # B**1.5 is taken from the grid data
        norm = 3.0/8.0*e**3.5/(c**2.5*me**1.5*(np.pi)**0.5)
        # P is taken from the grid data
        N0 = 3.0/me/c/c/(np.log(np.abs(gamc/gamma_min)))

        return N0*norm*nu**(-0.5)*np.exp(-nu/nuc)

    fname1 =('io', 'particle_sync_spec_%s' % nu_str)
    ds.add_field(fname1, function=_synchrotron_spec, sampling_type='particle',
                 units='cm**(3/4)*s**(3/2)/g**(3/4)', force_override=True)

    try:
        ds.add_particle_filter(ptype)
    except:
        raise NotImplementedError

    ###########################################################################
    ## Nearest Neighbor method
    ###########################################################################
    fname2 = ds.add_deposited_particle_field(
            (ptype, 'particle_sync_spec_%s' % nu_str), 'nearest')

    def _nn_emissivity_i(field, data):
        '''
        Emissivity using nearest neighbor. Integrate over line of sight to get intensity.
        '''
        Bvec = np.array([data[('gas', 'magnetic_field_x')],\
                         data[('gas', 'magnetic_field_y')],\
                         data[('gas', 'magnetic_field_z')]])
        cross = np.cross(los, Bvec, axisb=0)
        # B * sin(alpha) = (B * |(los x Bvec)|/|los|/|Bvec|)
        # = |(los x Bvec)|
        Bsina = np.sqrt(np.sum(cross*cross, axis=-1))
        Bsina = data.apply_units(Bsina, 'gauss')

        # P * B^1.5 /4pi
        PBsina = data['gas', 'pressure']*Bsina**1.5\
                 /yt.YTQuantity(4.*np.pi, 'sr')
                 #*data['gas', 'magnetic_field_strength']**1.5\
        frac = data['gas', 'jet_volume_fraction']

        return PBsina*frac*tot_const*data['deposit', '%s_nn_sync_spec_%s' % (ptype, nu_str)]


    #print ds.field_info[('jetp', 'particle_emissivity')].units
    #print ds.field_info[f4].units

    fname_nn_emis = ('deposit', 'nn_emissivity_i_%s_%s' % (ptype, nu_str))
    ds.add_field(fname_nn_emis, function=_nn_emissivity_i, sampling_type='cell',
                 display_name='%s NN Emissivity I (%s)' % (nu_str, ptype),
                 units='Jy/cm/arcsec**2', take_log=True,
                 force_override=True)

    def _nn_emissivity_q(field, data):
        Bvec = np.stack([data[('gas', 'magnetic_field_x')],\
                         data[('gas', 'magnetic_field_y')],\
                         data[('gas', 'magnetic_field_z')]], axis=-1)
        Bproj = Bvec - np.expand_dims(np.inner(Bvec, los), -1)*los
        # cos = cos(theta), theta is the angle between projected B and xvec
        cos = np.inner(Bproj, xvec)/np.sqrt(np.sum(Bproj*Bproj, axis=-1))
        cos[np.isnan(cos)] = 0.0
        fname_nn_emis = ('deposit', 'nn_emissivity_i_%s_%s' % (ptype, nu_str))
        # pol_ratio = (perp - para) / (perp + para)
        # The minus accounts for the perpendicular polarization
        return -data[fname_nn_emis]*pol_ratio*(2*cos*cos-1.0)

    fname_nn_emis_h = ('deposit', 'nn_emissivity_q_%s_%s' % (ptype, nu_str))
    ds.add_field(fname_nn_emis_h, function=_nn_emissivity_q, sampling_type='cell',
                 display_name='%s NN Emissivity Q (%s)' % (nu_str, ptype),
                 units='Jy/cm/arcsec**2', take_log=False,
                 force_override=True)

    def _nn_emissivity_u(field, data):
        Bvec = np.stack([data[('gas', 'magnetic_field_x')],\
                         data[('gas', 'magnetic_field_y')],\
                         data[('gas', 'magnetic_field_z')]], axis=-1)
        Bproj = Bvec - np.expand_dims(np.inner(Bvec, los), -1)*los
        cos = np.inner(Bproj, xvec)/np.sqrt(np.sum(Bproj*Bproj, axis=-1))
        sin = np.sqrt(1.0-cos*cos)
        cos[np.isnan(cos)] = 0.0
        sin[np.isnan(sin)] = 0.0
        fname_nn_emis = ('deposit', 'nn_emissivity_i_%s_%s' % (ptype, nu_str))
        return -data[fname_nn_emis]*pol_ratio*2*sin*cos

    fname_nn_emis_v = ('deposit', 'nn_emissivity_u_%s_%s' % (ptype, nu_str))
    ds.add_field(fname_nn_emis_v, function=_nn_emissivity_u, sampling_type='cell',
                 display_name='%s NN Emissivity U (%s)' % (nu_str, ptype),
                 units='Jy/cm/arcsec**2', take_log=False,
                 force_override=True)

    return fname1, fname2, fname_nn_emis, nu_str


def add_synchrotron_dtau_emissivity(ds, ptype='lobe', nu=(1.4, 'GHz'), method='nearest', proj_axis='x', \
                                    extend_cells=None):
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

    stokes = StokesFieldName(ptype, nu)
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

    def _synchrotron_spec(field, data):
        # ptype needs to be 'io' (only in this function)
        ptype = 'io'
        # To convert from FLASH "none" unit to cgs unit, times the B field from FLASH by sqrt(4*pi)
        Bvec = np.array([data[(ptype, 'particle_magx')],\
                         data[(ptype, 'particle_magy')],\
                         data[(ptype, 'particle_magz')]])*np.sqrt(4.0*np.pi)
        Bvec = data.apply_units(Bvec, 'gauss')

        # Calculate sin(a), in which a is the pitch angle of the electrons relative to B field.
        # See _nn_emissivity_i for more comments
        cross = np.cross(los, Bvec, axisb=0)
        Bsina = np.sqrt(np.sum(cross*cross, axis=-1))
        Bsina = data.apply_units(Bsina, 'gauss')
        #B = np.sqrt(np.sum(Bvec*Bvec, axis=0))

        # Return for the FieldDetector; do nothing
        if isinstance(data, FieldDetector):
            return data[ptype, 'particle_dens']/data[ptype, 'particle_den1']**(1./3.)/ \
                    (data[(ptype, 'particle_dtau')])

        # Density when the particle left the jet
        den1 = data[(ptype, 'particle_den1')]
        dtau = data[(ptype, 'particle_dtau')]

        if np.any(dtau < 0.0):
            print('negative tau!')
            print(data)
            print(data[(ptype, 'particle_tau')])
            print(dtau)

        # The new cutoff gamma
        gamc = (data[(ptype, 'particle_dens')] / den1)**(1./3.) / dtau
        ind = np.where(gamc < 0.0)[0]
        if ind.shape[0] > 0:
            print(ind)
            print(gamc)

        #gamc = data[(ptype, 'particle_gamc')]

        # Cutoff frequency
        nuc = 3.0*gamc**2*e*Bsina/(4.0*np.pi*me*c)
        #nu = data.get_field_parameter("frequency", default=yt.YTQuantity(1.4, 'GHz'))

        # B**1.5 is taken from the grid data
        norm = 3.0/8.0*e**3.5/(c**2.5*me**1.5*(np.pi)**0.5)
        # P is taken from the grid data
        N0 = 3.0/me/c/c/(np.log(np.abs(gamc/gamma_min)))/yt.YTQuantity(4.*np.pi, 'sr')

        # Fix where the cutoff gamma < 0
        N0[ind] = 0.0

        return N0*norm*nu**(-0.5)*np.exp(-nu/nuc)

    # particle field name
    pfname = ('io', 'particle_sync_spec_%s' % stokes.nu_str)
    ds.add_field(pfname, function=_synchrotron_spec, sampling_type='particle',
                 units='cm**(3/4)*s**(3/2)/g**(3/4)/sr', force_override=True)
    #try:
    ds.add_particle_filter(ptype)
    #except:
    #    raise NotImplementedError

    ###########################################################################
    ## Nearest Neighbor method
    ###########################################################################
    fname_nn = ds.add_deposited_particle_field(
            (ptype, 'particle_sync_spec_%s' % stokes.nu_str), 'nearest', extend_cells=extend_cells)

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

        # P * (B*sina)^1.5
        PBsina = data['gas', 'pressure']*Bsina**1.5

        frac = data['gas', 'jet_volume_fraction']

        return PBsina*frac*tot_const*data[fname_nn]

    ds.add_field(stokes.I, function=_nn_emissivity_i, sampling_type='cell',
                 display_name=stokes.display_name('I'),
                 units='Jy/cm/arcsec**2', take_log=True,
                 force_override=True)

    def _nn_emissivity_q(field, data):
        Bvec = np.stack([data[('gas', 'magnetic_field_x')],\
                         data[('gas', 'magnetic_field_y')],\
                         data[('gas', 'magnetic_field_z')]], axis=-1)
        Bproj = Bvec - np.expand_dims(np.inner(Bvec, los), -1)*los
        # cos = cos(theta), theta is the angle between projected B and xvec
        cos = np.inner(Bproj, xvec)/np.sqrt(np.sum(Bproj*Bproj, axis=-1))
        cos[np.isnan(cos)] = 0.0
        # pol_ratio = (perp - para) / (perp + para)
        # The minus accounts for the perpendicular polarization
        return -data[stokes.I]*pol_ratio*(2*cos*cos-1.0)

    ds.add_field(stokes.Q, function=_nn_emissivity_q, sampling_type='cell',
                 display_name=stokes.display_name('Q'),
                 units='Jy/cm/arcsec**2', take_log=False,
                 force_override=True)

    def _nn_emissivity_u(field, data):
        Bvec = np.stack([data[('gas', 'magnetic_field_x')],\
                         data[('gas', 'magnetic_field_y')],\
                         data[('gas', 'magnetic_field_z')]], axis=-1)
        Bproj = Bvec - np.expand_dims(np.inner(Bvec, los), -1)*los
        cos = np.inner(Bproj, xvec)/np.sqrt(np.sum(Bproj*Bproj, axis=-1))
        sin = np.sqrt(1.0-cos*cos)
        cos[np.isnan(cos)] = 0.0
        sin[np.isnan(sin)] = 0.0
        return -data[stokes.I]*pol_ratio*2*sin*cos

    ds.add_field(stokes.U, function=_nn_emissivity_u, sampling_type='cell',
                 display_name=stokes.display_name('U'),
                 units='Jy/cm/arcsec**2', take_log=False,
                 force_override=True)

    return pfname, fname_nn, stokes.I, stokes.nu_str


def setup_part_file(ds):
    filename = os.path.join(ds.directory,ds.basename)
    updated_pfname = filename.replace('plt_cnt', 'part')+'_updated'
    if os.path.exists(updated_pfname):
        ds._particle_handle = HDF5FileHandler(updated_pfname)
        ds.particle_filename = filename.replace('plt_cnt', 'part')+'_updated'
        mylog.info('Changed particle files to: %s', ds.particle_filename)
        return True
    else:
        return False


def synchrotron_file_name(ds, ptype, proj_axis, nu):
    nu_str = '%.1f%s' % nu
    postfix = '_synchrotron_%s_%s_' % (ptype, nu_str)
    if type(proj_axis) is str:
        postfix += proj_axis
    elif type(proj_axis) is list:
        postfix += '%.1f_%.1f_%.1f' % tuple(proj_axis)
    return ds.basename + postfix

def prep_field_data(ds, field, offset=1):
    """
    Prepare the grid data. Read the field data grid by grid, remove bad values.
    Return the numpy array with shape [ngrid, nx, ny, nz].
    """
    # data.shape should be (ngrid, nxb, nyb, nzb)
    data = np.zeros([ds.index.num_grids, *ds.index.grid_dimensions[0]], dtype='float32')
    # Go through all the grids in the index
    for g in parallel_objects(ds.index.grids, njobs=0):
        # Print the grid if nan or inf is in it
        if np.nan in g[field].v or np.inf in g[field].v:
            mylog.warning('Encountered non-physical values in %s', g) # g[field].v)
        # Calculate the field values in each grid
        # Use numpy nan_to_num to convert the bad values anyway
        # Transpose the array since the grid data in yt is transposed
        data[g.id-offset] = np.nan_to_num(g[field].v.transpose())
    comm = communication_system.communicators[-1]
    data = comm.mpi_allreduce(data, op="sum")

    return data


def write_synchrotron_hdf5(ds, ptype='lobe', proj_axis='x', nu=(1.4, 'GHz')):
    """
    Calculate the emissivity of Stokes I, Q, and U in each cell. Write them
    to a new HDF5 file and copy metadata from the original HDF5 files.
    The new HDF5 file can then be loaded into yt and make plots.
    """
    # The new file name that we are going to write to
    sfname = synchrotron_file_name(ds, ptype, proj_axis, nu)

    pars = add_synchrotron_dtau_emissivity(ds, ptype=ptype, nu=nu, proj_axis=proj_axis)

    h5_handle = ds._handle

    # Keep a list of the fields that were in the original hdf5 file
    orig_field_list = [field.decode() for field in h5_handle['unknown names'].value[:,0]]

    # Field names that we are going to write to the new hdf5 file
    stokes = StokesFieldName(ptype, nu)
    # Take only the field name, discard "deposit" field type
    write_fields = np.array([f for ftype, f in stokes.IQU])

    # Here we do the actual calculation (in yt) and save the grid data
    data = {}
    for field in write_fields:
        mylog.info('Preparing field: %s', field)
        data[field] = prep_field_data(ds, field)
    comm = communication_system.communicators[-1]
    if comm.rank == 0:
        with h5py.File(os.path.join(ds.directory, sfname), 'w') as h5file:
            mylog.info('Writing to %s', sfname)
            mylog.info('Fields to be written: %s', write_fields)

            # Go through all the items in the original hdf5 file
            for name, v in h5_handle.items():
                if name in orig_field_list:
                    # Do not write the fields that were present in the original hdf5 file
                    pass
                elif name == 'unknown names':
                    # Replace the field names by the new ones
                    bnames = [f.encode('utf8') for f in write_fields]
                    h5file.create_dataset('unknown names', data=bnames)
                else:
                    # Keep other simulation information
                    h5file.create_dataset(v.name, v.shape, v.dtype, v.value)
            for field in write_fields:
                fieldname = field[1] if type(field) is tuple else field
                # Here we do the actual writing
                mylog.info('Writing field: %s', fieldname)
                h5file.create_dataset(fieldname, data[field].shape, data[field].dtype, data[field])
