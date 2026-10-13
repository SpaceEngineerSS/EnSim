# Command line

The `ensim` command needs no GUI packages.

| Command | Does |
|---|---|
| `ensim` | Opens the desktop application (needs `ensim[gui]`). |
| `ensim engine …` | One engine operating point. |
| `ensim run project.ensim` | Evaluates a saved project; `--flight` also flies it. |
| `ensim check` | Short self-test of the installation. |
| `ensim --version` | Version. |

## engine

```console
$ ensim engine --fuel LCH4 --of 3.4 --pc 100 --eps 35 --throat-area 200
LCH4/LOX  O/F 3.4  Pc 100 bar  eps 35

Chamber
  temperature         3598.0 K
  gamma (frozen)      1.1955
  molecular weight    21.770 g/mol
  c*                  1809.2 m/s

Nozzle
  Isp vacuum        337.7 s
  Isp sea level     272.2 s
  exit Mach         4.234
  exit pressure     22.65 kPa
  Cf vacuum        1.8302
  thrust vacuum    366.04 kN
  mass flow       110.545 kg/s

Composition (mole fraction)
  H2O   0.49306
  CO    0.17817
  CO2   0.13022
  H2    0.08010
  OH    0.06636
  O2    0.02188
  H     0.02075
  O     0.00928
```

| Option | Meaning |
|---|---|
| `--fuel`, `--oxidizer` | Propellant keys; the oxidizer defaults to `LOX`. |
| `--of` | Oxidizer-to-fuel mass ratio. |
| `--pc` | Chamber pressure, bar. |
| `--eps` | Nozzle area ratio. |
| `--pa` | Ambient pressure for the operating point, bar (default 0). |
| `--throat-area` | Throat area, cm². Without it, thrust and mass flow are not reported. |
| `--eta-cstar`, `--eta-cf` | Efficiencies (default 1). |
| `--half-angle` | Conical divergence half-angle, degrees (default 15). |
| `--json [FILE]` | Machine-readable output, to standard output without a file name. |

## run

```console
$ ensim run kerolox_sounding_rocket.ensim --flight --json result.json
```

Reads the engine, rocket, recovery and environment sections of the project.
`--max-time` limits the flight (default 600 s).

## In scripts and CI

Exit status is 0 on success, 1 when the inputs are invalid or a solver does not
converge (with the reason on standard error), and 2 when the desktop
application is asked for without its packages.

```bash
ensim engine --fuel LH2 --of 6 --pc 70 --eps 60 --json | jq '.vacuum.isp'
```
