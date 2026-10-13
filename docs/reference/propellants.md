# ensim.core.propellants

Propellants in their storage state, with the reactant enthalpies NASA CEA uses.
`PROPELLANTS` maps each key to a `Propellant`; `FUELS` and `OXIDIZERS` list the
keys by role, in the order the desktop application shows them, and
`LEGACY_NAMES` maps the gas-phase names used by EnSim 3.x (`H2`, `O2`, `RP1`, …)
to the current keys.

| Key | Name | Role | Formula | T, K | h, kJ/kg |
|---|---|---|---|---|---|
| LH2 | Liquid hydrogen | fuel | H2 | 20.27 | −4470.5 |
| LCH4 | Liquid methane | fuel | CH4 | 111.64 | −5562.3 |
| RP-1 | RP-1 kerosene | fuel | CH1.9423 | 298.15 | −1768.6 |
| N2H4 | Hydrazine | fuel | N2H4 | 298.15 | +1572.2 |
| MMH | Monomethylhydrazine | fuel | CH6N2 | 298.15 | +1176.4 |
| UDMH | Unsymmetrical dimethylhydrazine | fuel | C2H8N2 | 298.15 | +813.7 |
| ETHANOL | Ethanol | fuel | C2H6O | 298.15 | −6023.9 |
| LOX | Liquid oxygen | oxidizer | O2 | 90.17 | −405.6 |
| NTO | Nitrogen tetroxide | oxidizer | N2O4 | 298.15 | −190.7 |
| HTP | Hydrogen peroxide (100 %) | oxidizer | H2O2 | 298.15 | −5520.6 |
| N2O | Nitrous oxide (vapour) | oxidizer | N2O | 298.15 | +1854.0 |

::: ensim.core.propellants
    options:
      show_root_heading: false
      members:
        - Propellant
        - get_propellant
