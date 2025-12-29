# Contributing to EnSim

Thank you for your interest in contributing to EnSim! This document provides guidelines for contributing.

## Code of Conduct

Be respectful and constructive in all interactions.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ensim.git
   cd ensim
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

## Development Workflow

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates

### Code Style

- Follow PEP 8
- Use type hints for function signatures
- Maximum line length: 100 characters
- Use docstrings (Google style)

### Testing

Run tests before submitting:
```bash
pytest tests/ -v
```

All tests must pass. Add tests for new features.

### Commit Messages

Use clear, descriptive commit messages:
```
Add Mach number solver for area ratio

- Implement Newton-Raphson iteration
- Add Numba JIT optimization
- Include unit tests
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Run tests: `pytest tests/`
4. Update documentation if needed
5. Submit PR with clear description

## Scientific Contributions

For physics-related changes:
- Cite sources in code comments
- Add validation tests comparing to NASA CEA
- Update `docs/VALIDATION.md`

## Areas for Contribution

- Additional propellant combinations
- Shifting equilibrium implementation
- Performance optimizations
- UI/UX improvements
- Documentation and examples

## Questions?

Open an issue for discussion.
