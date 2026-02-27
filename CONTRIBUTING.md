# Contributing to AI Traffic Monitor 🚦

Thank you for your interest in contributing! This document provides guidelines for contributing to this project.

## 🚀 Getting Started

1. **Fork** the repository on GitHub
2. **Clone** your fork locally
   ```bash
   git clone https://github.com/YOUR_USERNAME/TrafficMonitor.git
   ```
3. **Create a virtual environment** and install dependencies
   ```bash
   python3 -m venv traffic_env
   source traffic_env/bin/activate
   pip install -r requirements.txt
   ```

## 🌿 Branch Naming Convention

| Type | Format | Example |
|------|--------|---------|
| New Feature | `feature/description` | `feature/speed-estimation` |
| Bug Fix | `fix/description` | `fix/video-loading-error` |
| Documentation | `docs/description` | `docs/api-reference` |
| Performance | `perf/description` | `perf/faster-inference` |

## 📝 Commit Message Format

```
type: short description

Examples:
feat: add speed estimation using optical flow
fix: resolve video stream crash on MacOS
docs: update README installation steps
perf: optimize frame processing pipeline
```

## 🔧 How to Submit a Pull Request

1. Create your feature branch
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes and test them
3. Commit your changes
   ```bash
   git commit -m "feat: describe what you added"
   ```
4. Push to your branch
   ```bash
   git push origin feature/your-feature-name
   ```
5. Open a Pull Request on GitHub with:
   - Clear description of what you changed
   - Screenshots/videos if it's a visual change
   - Reference any related issues

## 🐛 Reporting Bugs

When reporting bugs, please include:
- Your OS (MacOS/Linux/Windows)
- Python version (`python3 --version`)
- Full error message from terminal
- Steps to reproduce the bug

## 💡 Feature Requests

Open an issue with the `enhancement` label and describe:
- What problem does this feature solve?
- How should it work?
- Any implementation ideas?

## 🎯 Priority Areas for Contribution

- [ ] Speed estimation (km/h calculation)
- [ ] Indian vehicle types (rickshaw, auto)
- [ ] Lane-specific analysis
- [ ] Emergency vehicle detection
- [ ] Performance optimization for Raspberry Pi

## 📄 Code Style

- Follow PEP 8 Python style guide
- Add docstrings to all functions and classes
- Keep functions small and focused (< 50 lines ideally)
- Add comments for complex logic

---

Thank you for making this project better! 🙏
