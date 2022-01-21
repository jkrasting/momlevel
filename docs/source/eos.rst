Equation of State
=========================================

The **momlevel** package includes low-level NumPy functions as well as xarray-based wrappers to calculate in-situ density, derivatives of temperature and salinity with respect to density, and the thermal expansion / haline contraction coefficients.  The implementation here uses the Wright (1997) equation of state (EOS), which is the default EOS used in MOM6 simulations.  Other packages already exist to calculate these quantities, such as the `Gibbs SeaWater (GSW) toolbox <https://www.teos-10.org/pubs/gsw/html/>`_ that uses the TEOS-10 EOS. While there is little fundamental difference between the two EOSs, attention must be paid to unit details such as conservative versus potential temperature and absolute versus practical salinity.

The Wright EOS implementation provided here is consistent with the EOS used in the model and is configured to use potential temperature referenced to the surface and practical salinity, which are directly output from the model.

Background
----------
The Wright Equation of State (1997) is the default EOS used in MOM6 for climate applications. The EOS provides a formulation for in-situ density based on potential temperature, salinity, and pressure. This implementation is computationally efficient and targeted for use in numerical models.

The **momlevel** package is structured in modular way such that addition EOS implementations can be added to the `momlevel.eos <api/momlevel.eos.html>`_ module.  The low-level NumPy functions have a basic interface of (`T`, `S`, and `p`) and are optimized through the higher-level xarray functions in the package.


Thermal and Haline Expansion Coefficients
-----------------------------------------
Thermal expansion coefficient (:math:`\alpha`) and haline contraction coefficient (beta) are derived from the equation state.  These quantities are typically involved in the calculation of buoyancy and stratification, i.e.:

.. math::
   N^2 = g \left(\alpha \frac{\partial\theta}{\partial z} - \beta\frac{\partial S}{\partial z} \right)

but they are also relevant for sea level rise as discussed below. The naming convention follows that increasing temperature leads to a decrease in density (i.e. expansion) and that increasing salinity leads to an increase in density (i.e. contraction).

Definitions
~~~~~~~~~~~
The coefficients are calculated by taking the derivative of density with respect to either temperature or salinity while holding the other quantity fixed along with pressure.

.. math::
    \alpha = - \frac{1}{\rho} \: \frac{\partial \rho}{\partial \theta}

.. math::
    \beta = \frac{1}{\rho} \: \frac{\partial \rho}{\partial S}


Relationship to sea level
~~~~~~~~~~~~~~~~~~~~~~~~~
The coefficients can be interpreted as a "potential", i.e. how much a given change in temperature or salinity can contribute locally to steric sea level change.  In the absence of a transient scenario where heat or salt is added to the ocean, the rearrangement of existing water masses can contribute to non-zero density-driven sea level changes. This is one way in which changes in ocean circulation can impact sea level.

The coefficients are related to thermosteric and halosteric sea level change:

.. math::
   \left( \frac{\partial\eta}{\partial t}  \right)_{thermosteric} = \int_{-H}^{\eta} \alpha \left( \frac{\partial \theta}{\partial t} \right) dz

.. math::
   \left( \frac{\partial\eta}{\partial t}  \right)_{halosteric} = - \int_{-H}^{\eta} \beta \left( \frac{\partial S}{\partial t} \right) dz

Where :math:`\eta` is the sea level and :math:`-H` is the sea floor

.. note::
  These definitions of the local thermosteric and halosteric sea level change are different from the implementation used in the ``momlevel.steric`` module.

References
----------
Wright, D.G., 1997. An equation of state for use in ocean models: Eckart’s formula revisited. Journal of Atmospheric and Oceanic Technology, 14(3), pp.735-740. `https://doi.org/10.1175/1520-0426(1997)014%3C0735:AEOSFU%3E2.0.CO;2 <https://doi.org/10.1175/1520-0426(1997)014%3C0735:AEOSFU%3E2.0.CO;2>`_


