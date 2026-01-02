# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | ✅ Yes             |
| < 1.0   | ❌ No              |

## Reporting a Vulnerability

We take security issues seriously. If you discover a security vulnerability in EnSim, please report it responsibly.

### How to Report

**Please do NOT report security vulnerabilities through public GitHub issues.**

Instead, please report them via one of these methods:

1. **Email**: Send details to security@ensim.io (preferred)
2. **GitHub Security Advisories**: Use the [Security tab](../../security/advisories/new) to privately report vulnerabilities

### What to Include

Please include the following information in your report:

- **Description**: A clear description of the vulnerability
- **Impact**: What an attacker could potentially achieve
- **Steps to Reproduce**: Detailed steps to reproduce the issue
- **Affected Versions**: Which versions are affected
- **Possible Fix**: If you have suggestions for fixing the vulnerability

### Response Timeline

- **Initial Response**: Within 48 hours of receiving your report
- **Status Update**: Within 7 days with our assessment
- **Resolution**: We aim to release a fix within 30 days for critical issues

### What to Expect

1. **Acknowledgment**: We'll confirm receipt of your report
2. **Assessment**: We'll investigate and assess the severity
3. **Communication**: We'll keep you informed of our progress
4. **Credit**: With your permission, we'll acknowledge your contribution in our release notes

## Security Best Practices for Users

### Running EnSim Safely

1. **Download from Official Sources**: Only download EnSim from the official GitHub repository
2. **Verify Releases**: Check release signatures when available
3. **Keep Updated**: Always use the latest stable version
4. **Virtual Environments**: Run EnSim in a Python virtual environment

### Data Security

1. **Project Files**: `.ensim` project files are JSON-based and may contain your simulation parameters
2. **No Network**: EnSim does not send data over the network by default
3. **Local Storage**: All data is stored locally on your machine

## Security Features

### Current Implementation

- **No Remote Code Execution**: EnSim does not execute remote code
- **No Network Requests**: Core functionality works fully offline
- **Input Validation**: User inputs are validated before processing
- **Dependency Management**: Dependencies are pinned to specific versions

### Known Limitations

- **Numba JIT Compilation**: Numba compiles Python to machine code; we use `cache=True` for security
- **File Operations**: Project save/load operations use Python's `json` module
- **PyQt6**: UI framework with its own security considerations

## Third-Party Dependencies

EnSim uses the following major dependencies:

| Package | Security Notes |
|---------|---------------|
| NumPy | Widely audited scientific library |
| Numba | JIT compiler; uses cached compilation |
| PyQt6 | Qt framework with active security patches |
| Matplotlib | Plotting library; no network operations |
| PyVista | VTK-based 3D visualization |

We regularly update dependencies and monitor for security advisories.

## Acknowledgments

We thank the security researchers who have helped improve EnSim's security:

*No security issues have been reported yet.*

---

*This security policy is adapted from industry best practices and may be updated as needed.*

