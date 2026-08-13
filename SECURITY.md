# Security policy

## Reporting a vulnerability

Please use the repository's
[private vulnerability reporting](https://github.com/SpaceEngineerSS/EnSim/security/advisories/new).
Do not disclose an exploitable issue in a public issue before a fix is available.

Include the affected version, platform, impact, minimal reproduction and any
suggested mitigation. No fixed response time is promised, but actionable reports
will be triaged as maintainer availability permits.

## Security boundary

EnSim is a local desktop and Python application. Normal calculations do not send
project or simulation data over the network. Project files are JSON data and are
not intended to contain executable code. Treat files from untrusted sources as
untrusted input and use a virtual environment with current dependencies.

Only the latest release line receives fixes. Older releases may remain available
for reproducibility but should not be assumed supported.
