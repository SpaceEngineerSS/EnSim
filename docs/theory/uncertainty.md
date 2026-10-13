# Uncertainty

Two studies repeat a calculation many times with inputs drawn at random: the
engine uncertainty on the Analysis page and the flight dispersion on the Flight
page. Both are only as meaningful as the distributions you give them.

## Distributions

Quantities that must stay positive (thrust, Isp, burn time, drag coefficient,
chamber pressure, throat area) are multiplied by a lognormal factor with mean
one and the stated coefficient of variation $c$:

$$
\ln X\sim\mathcal N\!\left(-\tfrac12\sigma^2,\ \sigma^2\right),\qquad \sigma^2=\ln(1+c^2).
$$

For small $c$ this is indistinguishable from a normal distribution; for large
$c$ it never produces a negative thrust.

Other inputs:

| Input | Distribution |
|---|---|
| Gamma (engine) | Normal, truncated to $1<\gamma<2$ |
| Rail elevation | Normal about the nominal value |
| Fin misalignment | Uniform in $[-\delta,\delta]$ |
| Wind speed | Normal, truncated at zero |
| Wind direction | Normal about the mean, or uniform over the compass |

The inputs are independent. If yours are correlated, thrust and burn time
for a given total impulse for instance, the spread will be overstated.

## Engine uncertainty

Chamber pressure, throat area and gamma are drawn around the last engine
result and pushed through the constant-$\gamma$ nozzle relations with the
chamber temperature and molar mass held fixed. The output is the distribution of
thrust and Isp. With a minimum thrust set, the share of samples above it is
reported as reliability.

Samples where the nozzle cannot be evaluated are counted as failed and left
out of the statistics; they are never replaced with the nominal value. The
result states attempted, valid and failed counts.

## Flight dispersion

Each case flies the full six-degree-of-freedom model with its own thrust, Isp,
burn time, drag coefficient, rail elevation, fin misalignment and wind. From
the cases that land before the time limit:

- **mean impact point** and the sample standard deviation ($N-1$) in east and north;
- **CEP**, the median distance of the impacts from the mean impact point;
- **confidence ellipse** from the eigenvectors of the impact covariance matrix,
  with semi-axes $\sqrt{\lambda_i\,\chi^2_2(P)}$ at confidence $P$, 99.7 % by
  default.

Apogee statistics use every case, landed or not.

A fixed seed reproduces the study exactly, in one process or many.

## How many runs

The mean converges quickly; the tails do not. As a rule of thumb, fifty cases
give a usable mean and standard deviation, a few hundred are needed before the
99.7 % ellipse settles. Run the study twice with different seeds: if the
ellipse changes noticeably, run more.

## What this is not

These studies propagate the scatter you put in. They do not account for the
model being wrong (constant drag coefficient, frozen nozzle flow, the
atmosphere on the day) and they do not infer distributions from data. A study
that will be relied on should record where each distribution came from.

## Sources

- NASA-STD-7009B, *Standard for Models and Simulations*, 2024.
- NASA-HDBK-7009, *Handbook for Models and Simulations*, 2013.
