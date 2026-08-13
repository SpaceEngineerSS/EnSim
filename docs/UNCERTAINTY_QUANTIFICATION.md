# Uncertainty quantification

## Engine UQ

The engine analyzer perturbs chamber pressure, area ratio, chamber temperature,
gamma and molar mass around a nominal ideal-nozzle calculation. Positive inputs
use lognormal multiplicative factors. Gamma is truncated to its mathematical
domain. Every run uses the same supplied seed and preserves sample order.

Failed samples are counted and omitted from statistics; they are never replaced
with the nominal output. Results include attempted, valid and failed counts.

## Flight dispersion

The flight analyzer perturbs thrust, specific impulse, mass, drag and wind. It
uses the configured rocket and a burn time derived from its propellant and mass
flow. Landing covariance uses sample statistics (`N-1`). The confidence ellipse
is obtained from the eigensystem of the horizontal covariance matrix and the
chi-square quantile for two dimensions at the displayed confidence level.

## Interpretation

These calculations propagate aleatory distributions selected by the user. They
do not identify model-form error, infer distributions from data or establish
epistemic credibility. Correlation between uncertain inputs is not automatically
invented. A defensible study should document:

1. the source and units of every input distribution;
2. dependencies and truncation;
3. convergence of output statistics with sample count;
4. numerical failure rate;
5. separate model-form discrepancy and validation evidence.

## References

- NASA, [NASA-STD-7009B: Standard for Models and Simulations](https://standards.nasa.gov/standard/nasa/nasa-std-7009), 2024.
- NASA, [NASA-HDBK-7009: Handbook for Models and Simulations](https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20140002378.pdf), 2013.
