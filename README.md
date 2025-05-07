# cusp-halo-relation
Code for modeling the central prompt cusps of dark matter halos.

## Usage:
In Python, start with:

```
import sys
sys.path.append("path/to/cusp-halo-relation")
import cusp_halo_relation
```

### Evaluating cusp parameters

To quickly evaluate cusp parameters in warm dark matter models, create a `CuspHaloWDM` object:

```
dark_matter_mass = 10. # dark matter particle mass in keV
model = cusp_halo_relation.CuspHaloWDM(mX=dark_matter_mass)

halo_mass = 1e9 # in solar mass
redshift = 2.

A = model.A_at_z(M=halo_mass,z=redshift) # cusp coefficient A for these halos, in solar mass/Mpc^1.5
m = model.A_at_z(M=halo_mass,z=redshift) # cusp mass for these halos, in solar mass
c = model.c_at_z(z=redshift) # halo concentration
```

By default we use [Planck 2018](https://arxiv.org/abs/1807.06209) cosmological parameters, a matter power spectrum precomputed using [CLASS](http://class-code.net/) with those parameters, and a warm dark matter transfer function from [Vogel & Abazajian (2023)](https://arxiv.org/abs/2210.10753). See the docstring with `help(cusp_halo_relation.CuspHaloWDM)` (or in [cusp_halo_concordance.py](/cusp_halo_relation/cusp_halo_concordance.py)) for further options.

For more general dark matter models, create a `CuspHaloStandard` object instead, which allows passing a custom transfer function. See the docstring with `help(cusp_halo_relation.CuspHaloStandard)` (or in [cusp_halo_concordance.py](/cusp_halo_relation/cusp_halo_concordance.py)) for how to do this. For convenience, we also include a class `Cutoff` which includes some calculations to support cold dark matter models. For example, for a 100 GeV WIMP decoupling at 30 MeV:

```
dark_matter_mass = 100e3 # in MeV
decoupling_temperature = 30. # in MeV
cutoff = cusp_halo_relation.Cutoff('G04',m=dark_matter_mass,Td=decoupling_temperature)
model = cusp_halo_relation.CuspHaloStandard(cutoff=cutoff.transfer,include_baryons=False)
```

Here we use the WIMP transfer function from [Green, Hofmann, & Schwarz (2004)](https://arxiv.org/abs/astro-ph/0309621), and with `include_baryons=False` we assume that only dark matter (and not baryons) cluster and contribute to halos and cusps. It is also possible to specify a custom free-streaming transfer function and take advantage of a supplied calculation of the free-streaming scale that properly accounts for the Standard Model thermal history; see `help(cusp_halo_relation.Cutoff)` (or [cutoffs.py](/cusp_halo_relation/cutoffs.py)).

For a completely arbitrary cosmology, use `CuspHalo`, which takes the matter power spectrum as input. See `help(cusp_halo_relation.CuspHalo)` (or [cusp_halo.py](/cusp_halo_relation/cusp_halo.py)) for further instruction.

### Using cuspy halo density profiles

To evaluate the cusp-NFW density use `cusp_halo_relation.cuspNFW.density(r,rs,rhos,A)`, which returns the density at radius `r` given scale radius `rs`, scale density `rhos`, and cusp coefficient `A`. To evaluate the enclosed mass, use `cusp_halo_relation.cuspNFW.mass(r,rs,rhos,A)`.

To obtain the scale radius and scale density in the first place, use `cusp_halo_relation.cuspNFW.scale_from_c(c,M,A,rho_vir)`. This takes the halo mass `M`, concentration `c`, cusp coefficient `A`, and virial density `rho_vir`.

See `help(cusp_halo_relation.cuspNFW)` (or [cuspNFW.py](/cusp_halo_relation/cuspNFW.py)) for further possibilities.

## Requirements

We require Python with `numpy` and `scipy`.

## Acknowledgement

If you use this code, please cite the paper [coming soon]. Other citations are likely also appropriate, including

* [Vogel & Abazajian (2023)](https://arxiv.org/abs/2210.10753), if you use the `'VA23'` warm dark matter transfer function.
* [Viel et al. (2005)](https://arxiv.org/abs/astro-ph/0501562), if you use the `'V05'` warm dark matter transfer function.
* [Green, Hofmann, & Schwarz (2004)](https://arxiv.org/abs/astro-ph/0309621), if you use the `'G04'` WIMP transfer function.
* [Eisenstein & Hu (1998)](https://arxiv.org/abs/astro-ph/9709112), if you use `transfer='EH'` to make the matter power spectrum.
* [Blas, Lesgourgues, & Tram (2011)](https://arxiv.org/abs/1104.2933), if you use `transfer='table'` to make the matter power spectrum, as this uses a power spectrum precomputed using [CLASS](http://class-code.net/).
* [Laine & Meyer (2015)](https://arxiv.org/abs/1503.04935) and [Borsanyi et al. (2016)](https://arxiv.org/abs/1606.07494), if you use `Cutoff` to do calculations with cold dark matter (decoupling temperatures above ~10 MeV), as we use Standard Model thermal history data drawn from these works.
* [Ludlow et al. (2013)](https://arxiv.org/abs/1302.0288), if you use halo concentrations from this code, since we directly use the results from that work.
