import os
import numpy as np
import pickle
import yt
from yt.fields.derived_field import ValidateGridType
from yt.fields.field_detector import FieldDetector
from yt.funcs import mylog, only_on_root
import sys
sys.path.append('../Particles/')
from particle_filters import *

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

    fnames = []
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

        B = np.sqrt(np.sum(Bvec*Bvec, axis=0))

        # Do nothing for now. How do we integrate over uniform pitch angle? 
        Bsina = B

        # Return for the FieldDetector; do nothing
        if isinstance(data, FieldDetector):
            return data[ptype, 'particle_dens']/data[ptype, 'particle_den1']**(1./3.)/ \
                    (data[(ptype, 'particle_dtau')])

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
        N0 = 3.0/me/c/c/(np.log(np.abs(gamc/gamma_min)))
        N0[ind] = 0.0

        return N0*norm*nu**(-0.5)*np.exp(-nu/nuc)

    fname1 =('io', 'particle_sync_spec_%s' % nu_str)
    ds.add_field(fname1, function=_synchrotron_spec, sampling_type='particle',
                 units='cm**(3/4)*s**(3/2)/g**(3/4)', force_override=True)
    #try:
    ds.add_particle_filter(ptype)
    #except:
    #    raise NotImplementedError

    ###########################################################################
    ## Nearest Neighbor method
    ###########################################################################
    fname2 = ds.add_deposited_particle_field(
            (ptype, 'particle_sync_spec_%s' % nu_str), 'nearest', extend_cells=extend_cells)

    def _nn_emissivity_i(field, data):
        '''
        Emissivity using nearest neighbor. Integrate over line of sight to get intensity.
        '''
        B = data[('gas', 'magnetic_field_magnitude')]

        # P * B^1.5
        PB = data['gas', 'pressure']*B**1.5\
             /yt.YTQuantity(4.*np.pi, 'sr')
        frac = data['gas', 'jet_volume_fraction']

        # Integral of 1/2*(sin(alpha))^(5/2) from 0 to pi
        sina52 = 0.5*1.43777

        return PB*sina52*frac*tot_const*data['deposit', '%s_nn_sync_spec_%s' % (ptype, nu_str)]

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

    fname_nn_emis_q = ('deposit', 'nn_emissivity_q_%s_%s' % (ptype, nu_str))
    ds.add_field(fname_nn_emis_q, function=_nn_emissivity_q, sampling_type='cell',
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

    fname_nn_emis_u = ('deposit', 'nn_emissivity_u_%s_%s' % (ptype, nu_str))
    ds.add_field(fname_nn_emis_u, function=_nn_emissivity_u, sampling_type='cell',
                 display_name='%s NN Emissivity U (%s)' % (nu_str, ptype),
                 units='Jy/cm/arcsec**2', take_log=False,
                 force_override=True)

    return fname1, fname2, fname_nn_emis, nu_str
