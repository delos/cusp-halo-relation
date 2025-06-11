# cusp-halo-relation
Code for modeling the central prompt cusps of dark matter halos in accordance with the methodology described by [Delos (2025)](https://arxiv.org/abs/2506.03240).

## Requirements

We require Python with `numpy` and `scipy`.

## Usage

Clone or [download](/../../archive/refs/heads/main.zip) the repository. In Python, start with:

```
import sys
sys.path.append("path/to/repository") # replace this with the path to your copy of the repository
import cusp_halo_relation
```

### Evaluating cusp parameters

To quickly evaluate cusp parameters in warm dark matter models, create a `CuspHaloWDM` object:

```
model = cusp_halo_relation.CuspHaloWDM(
    mX=10., # dark matter particle mass in keV
    )

halo_mass = 1e9 # in solar mass
redshift = 2.

A = model.A_at_z(M=halo_mass,z=redshift) # cusp coefficient A for these halos, in solar mass/Mpc^1.5
m = model.A_at_z(M=halo_mass,z=redshift) # cusp mass for these halos, in solar mass
c = model.c_at_z(z=redshift) # halo concentration
```

By default we use [Planck 2018](https://arxiv.org/abs/1807.06209) cosmological parameters, a matter power spectrum precomputed using [CLASS](http://class-code.net/) with those parameters, and a warm dark matter transfer function from [Vogel & Abazajian (2023)](https://arxiv.org/abs/2210.10753). See the docstring with `help(cusp_halo_relation.CuspHaloWDM)` (or in [cusp_halo_concordance.py](/cusp_halo_relation/cusp_halo_concordance.py)) for further options.

For more general dark matter models, create a `CuspHaloStandard` object instead, which allows passing a custom transfer function. See the docstring with `help(cusp_halo_relation.CuspHaloStandard)` (or in [cusp_halo_concordance.py](/cusp_halo_relation/cusp_halo_concordance.py)) for how to do this. For convenience, we also include a class `Cutoff` which includes some calculations to support cold dark matter models. For example, for a 100 GeV WIMP decoupling at 30 MeV:

```
model = cusp_halo_relation.CuspHaloStandard(
    cutoff=cusp_halo_relation.Cutoff(
        'G04', # Green, Hofmann, & Schwarz (2004) transfer function
        m=100e3, # dark matter particle mass in MeV
        Td=30., # dark matter decoupling temperature in MeV
        ),
    include_baryons=False, # baryons do not contribute to halos and cusps
    )
```

Here we use the WIMP transfer function from [Green, Hofmann, & Schwarz (2004)](https://arxiv.org/abs/astro-ph/0309621), and with `include_baryons=False` we assume that only dark matter (and not baryons) cluster and contribute to halos and cusps. It is also possible to specify a custom free-streaming transfer function and take advantage of a supplied calculation of the free-streaming scale that properly accounts for the Standard Model thermal history; see `help(cusp_halo_relation.Cutoff)` (or [cutoffs.py](/cusp_halo_relation/cutoffs.py)).

For a completely arbitrary cosmology, use `CuspHalo`, which takes the matter power spectrum as input. See `help(cusp_halo_relation.CuspHalo)` (or [cusp_halo.py](/cusp_halo_relation/cusp_halo.py)) for further instruction.

### Using cuspy halo density profiles

To evaluate the cusp-NFW density use `cusp_halo_relation.cuspNFW.density(r,rs,rhos,A)`, which returns the density at radius `r` given scale radius `rs`, scale density `rhos`, and cusp coefficient `A`. Other quantities are also available:
- To evaluate the enclosed mass, use `cusp_halo_relation.cuspNFW.mass(r,rs,rhos,A)`.
- To evaluate the gravitational potential, use `cusp_halo_relation.cuspNFW.potential(r,rs,rhos,A,G)`. Here `G` is the gravitational constant. There is also an optional argument `zero_at_inf`; if `False` (the default) the zero point of energy corresponds to the potential is at the center of the halo, while if `True` it is the potential at infinity.
- To evaluate the radial velocity dispersion (assuming isotropic velocities), use `cusp_halo_relation.cuspNFW.veldisp_r(r,rs,rhos,A,G)`.
- To evaluate the distribution function at energy `E` (assuming isotropic velocities), use `cusp_halo_relation.cuspNFW.df(E,rs,rhos,A,G)`. The optional argument `zero_at_inf` is also allowed here.

To obtain the scale radius and scale density in the first place, use `cusp_halo_relation.cuspNFW.scale_from_c(c,M,A,rho_vir)`. This takes the halo mass `M`, concentration `c`, cusp coefficient `A`, and virial density `rho_vir`. For example, continuing the warm dark matter example above, we can evaluate the scale radius and density for the cusp-NFW profile of a halo with the predicted cusp coefficient and halo concentration at a given redshift:

```
model = cusp_halo_relation.CuspHaloWDM(mX=10.) # 10 keV warm dark matter

halo_mass = 1e9 # in solar mass
redshift = 2.

A = model.A_at_z(M=halo_mass,z=redshift) # cusp coefficient A for these halos, in solar mass/Mpc^1.5
c = model.c_at_z(z=redshift) # halo concentration

rho_vir = 200.*model.rhoCrit_at_z(z=redshift)
rs, rhos = cusp_halo_relation.cuspNFW.scale_from_c(c,M,A,rho_vir)
```

See `help(cusp_halo_relation.cuspNFW)` (or [cuspNFW.py](/cusp_halo_relation/cuspNFW.py)) for further possibilities.

## Examples

We include in the [examples](/examples/) subdirectory several examples of how to use the code.
- [wdm_cusps.py](/examples/wdm_cusps.py): generates figures 14 and 15 of [Delos (2025)](https://arxiv.org/abs/2506.03240), which show how the central cusps of halos of fixed mass depend on the dark matter particle mass.
- [wdm_profiles.py](/examples/wdm_profiles.py): generates figure 17 of [Delos (2025)](https://arxiv.org/abs/2506.03240), which shows examples of cusp-NFW density profiles with different parameters.
- [stability.py](/examples/stability.py): generates figure 24 of [Delos (2025)](https://arxiv.org/abs/2506.03240), which shows the distribution functions of halos with cusp-NFW density profiles and isotropic velocity distributions.

## Acknowledgement

If you use this code, please cite the paper [Delos (2025)](https://arxiv.org/abs/2506.03240). Other citations are likely also appropriate, including

* [Vogel & Abazajian (2023)](https://arxiv.org/abs/2210.10753), if you use the `'VA23'` warm dark matter transfer function.
* [Viel et al. (2005)](https://arxiv.org/abs/astro-ph/0501562), if you use the `'V05'` warm dark matter transfer function.
* [Green, Hofmann, & Schwarz (2004)](https://arxiv.org/abs/astro-ph/0309621), if you use the `'G04'` WIMP transfer function.
* [Eisenstein & Hu (1998)](https://arxiv.org/abs/astro-ph/9709112), if you use `transfer='EH'` to make the matter power spectrum.
* [Blas, Lesgourgues, & Tram (2011)](https://arxiv.org/abs/1104.2933), if you use `transfer='table'` to make the matter power spectrum, as this uses a power spectrum precomputed using [CLASS](http://class-code.net/).
* [Laine & Meyer (2015)](https://arxiv.org/abs/1503.04935) and [Borsanyi et al. (2016)](https://arxiv.org/abs/1606.07494), if you use `Cutoff` to do calculations with cold dark matter (decoupling temperatures above ~10 MeV), as we use Standard Model thermal history data drawn from these works.
* [Ludlow et al. (2013)](https://arxiv.org/abs/1302.0288), if you use halo concentrations from this code, since we directly use the results from that work.
