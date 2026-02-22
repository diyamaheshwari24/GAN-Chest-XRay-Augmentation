# Contributing to GAN-Chest-XRay-Augmentation

Thank you for your interest in contributing to this project! 🎉

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](../../issues)
2. If not, create a new issue with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (OS, Python version, GPU)

### Suggesting Enhancements

1. Open an issue with the `enhancement` label
2. Describe the feature and its use case
3. Provide examples if possible

### Pull Requests

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Run tests (if applicable)
5. Commit with descriptive messages
6. Push to your fork
7. Open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR-USERNAME/GAN-Chest-XRay-Augmentation.git
cd GAN-Chest-XRay-Augmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt

# Download dataset (see data/README.md)
```

## Code Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Add docstrings to functions and classes
- Keep functions focused and small

## Areas for Contribution

- [ ] Add FID score calculation for GAN evaluation
- [ ] Implement Progressive GAN for higher resolution
- [ ] Add support for other medical imaging datasets
- [ ] Improve classifier performance
- [ ] Add web interface demo
- [ ] Write unit tests
- [ ] Improve documentation

## Questions?

Feel free to open an issue or reach out!

Thank you for contributing! 🙏
