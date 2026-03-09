Predefined Units
================

The default registry (``u``) comes pre-populated with SI base units, derived units, and
common non-SI units.

Base SI Units
-------------

.. list-table::
   :header-rows: 1
   :widths: 10 20 20 30 10

   * - Symbol
     - Name
     - Dimension
     - Aliases
     - Prefixable?
   * - ``m``
     - meter
     - Length (L)
     - -
     - Yes
   * - ``kg``
     - kilogram
     - Mass (M)
     - -
     - No
   * - ``s``
     - second
     - Time (T)
     - -
     - Yes
   * - ``A``
     - ampere
     - Current (I)
     - -
     - Yes
   * - ``K``
     - kelvin
     - Temperature (Θ)
     - -
     - No
   * - ``mol``
     - mole
     - Amount (N)
     - -
     - Yes
   * - ``cd``
     - candela
     - Luminous (J)
     - -
     - Yes

Named Derived SI Units
----------------------

.. list-table::
   :header-rows: 1
   :widths: 10 20 20 30 10

   * - Symbol
     - Name
     - Dimension
     - Aliases
     - Prefixable?
   * - ``g``
     - gram
     - Mass
     - -
     - Yes
   * - ``Hz``
     - hertz
     - Frequency (1/T)
     - -
     - Yes
   * - ``N``
     - newton
     - Force (M·L/T²)
     - -
     - Yes
   * - ``Pa``
     - pascal
     - Pressure (M/L·T²)
     - -
     - Yes
   * - ``J``
     - joule
     - Energy (M·L²/T²)
     - -
     - Yes
   * - ``W``
     - watt
     - Power (M·L²/T³)
     - -
     - Yes
   * - ``C``
     - coulomb
     - Charge (I·T)
     - -
     - Yes
   * - ``V``
     - volt
     - Voltage (M·L²/T³·I)
     - -
     - Yes
   * - ``F``
     - farad
     - Capacitance (I²·T⁴/M·L²)
     - -
     - Yes
   * - ``Ω``
     - ohm
     - Resistance (M·L²/T³·I²)
     - ``ohm``, ``Ohm``, ``OHM``
     - Yes
   * - ``S``
     - siemens
     - Conductance (I²·T³/M·L²)
     - -
     - Yes
   * - ``Wb``
     - weber
     - Magnetic Flux (M·L²/T²·I)
     - -
     - Yes
   * - ``T``
     - tesla
     - Flux Density (M/T²·I)
     - -
     - Yes
   * - ``H``
     - henry
     - Inductance (M·L²/T²·I²)
     - -
     - Yes
   * - ``lm``
     - lumen
     - Luminous Flux (J)
     - -
     - Yes
   * - ``lx``
     - lux
     - Illuminance (J/L²)
     - -
     - Yes
   * - ``Bq``
     - becquerel
     - Radioactivity (1/T)
     - -
     - Yes
   * - ``Gy``
     - gray
     - Absorbed Dose (L²/T²)
     - -
     - Yes
   * - ``Sv``
     - sievert
     - Equivalent Dose (L²/T²)
     - -
     - Yes
   * - ``kat``
     - katal
     - Catalytic Activity (N/T)
     - -
     - Yes

Dimensionless Units
-------------------

.. list-table::
   :header-rows: 1
   :widths: 10 20 20 30 10

   * - Symbol
     - Name
     - Dimension
     - Aliases
     - Prefixable?
   * - ``rad``
     - radian
     - Dimensionless
     - -
     - Yes
   * - ``sr``
     - steradian
     - Dimensionless
     - -
     - Yes

Non-SI Time Units
-----------------

.. list-table::
   :header-rows: 1
   :widths: 10 20 20 30 10

   * - Symbol
     - Name
     - Dimension
     - Aliases
     - Prefixable?
   * - ``min``
     - minute
     - Time
     - ``minute``, ``minutes``
     - No
   * - ``h``
     - hour
     - Time
     - ``hr``, ``hour``, ``hours``
     - No
   * - ``d``
     - day
     - Time
     - ``day``, ``days``
     - No
   * - ``wk``
     - week
     - Time
     - ``week``, ``weeks``
     - No
   * - ``fortnight``
     - fortnight
     - Time
     - ``fortnights``
     - No
   * - ``mo``
     - month
     - Time (Gregorian avg)
     - ``month``, ``months``
     - No
   * - ``yr``
     - year
     - Time (Gregorian avg)
     - ``year``, ``years``, ``annum``
     - No
   * - ``yr_julian``
     - Julian year
     - Time
     - -
     - No
   * - ``decade``
     - decade
     - Time
     - ``dec``, ``decades``
     - No
   * - ``century``
     - century
     - Time
     - ``cent``, ``centuries``
     - No
   * - ``millennium``
     - millennium
     - Time
     - ``millennia``
     - No


Temperature Difference Units
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 10 24 18 30 10

   * - Symbol
     - Name
     - Dimension
     - Aliases
     - Prefixable?
   * - ``K``  (*ΔT*)
     - kelvin difference
     - Temperature (Θ)
     - -
     - No
   * - ``Δ°C``
     - celsius degree difference
     - Temperature (Θ)
     - ``delta_degC``, ``delta_deg_celsius``, ``delta_celsius``
     - No
   * - ``Δ°F``
     - fahrenheit degree difference
     - Temperature (Θ)
     - ``delta_degF``, ``delta_deg_fahrenheit``, ``delta_fahrenheit``
     - No
   * - ``Δ°R``
     - rankine degree difference
     - Temperature (Θ)
     - ``delta_degR``, ``delta_deg_rankine``, ``delta_rankine``
     - No
