# 🚀 Quick Setup Guide for New Developers

## 🎯 TL;DR - Fast Setup

```bash
# 1. Check Python version (MUST be 3.10.x or 3.11.x)
python setup_check.py

# 2. If wrong version, fix it:
pyenv install 3.10.6 && pyenv local 3.10.6

# 3. Create virtual environment and install
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# 4. Verify everything works
./activate_env.sh
```

## ⚠️ Critical Requirements

### Python Version
- **MUST USE**: Python 3.10.x or 3.11.x
- **CANNOT USE**: Python 3.12+ (breaks PyCaret)
- **Recommended**: Python 3.10.6

### Why These Restrictions?
PyCaret (our core ML library) doesn't support Python 3.12+. This is enforced in:
- `pyproject.toml`: `requires-python = ">=3.10,<3.12"`
- `requirements.txt`: Version warning comments
- `setup_check.py`: Automated version validation

## 🛠️ Detailed Setup

### 1. Install pyenv (if needed)
```bash
# macOS
brew install pyenv

# Linux
curl https://pyenv.run | bash
```

### 2. Install correct Python
```bash
# Install Python 3.10.6
pyenv install 3.10.6

# Set as local version for this project
cd /path/to/TheProjectionWizard
pyenv local 3.10.6
```

### 3. Verify Python version
```bash
# Run our validation script
python setup_check.py

# Should show: ✅ Python version is COMPATIBLE!
```

### 4. Create virtual environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### 5. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Test everything works
```bash
# Use our convenience script
./activate_env.sh

# Should show all green checkmarks ✅
```

## 🧪 Quick Test

After setup, verify PyCaret works:
```bash
python -c "import pycaret; from pycaret.datasets import get_data; print('✅ PyCaret working!')"
```

## 🆘 Troubleshooting

### "PyCaret import failed"
- Run `python setup_check.py` to verify Python version
- If using Python 3.12+, downgrade to 3.10.6
- Recreate virtual environment with correct Python version

### "Command not found: pyenv"
- Install pyenv first (see step 1 above)
- Restart terminal after installation
- Add pyenv to PATH if needed

### "Python version not changing"
- Make sure you're in the project directory
- Run `pyenv local 3.10.6` in project root
- Check `.python-version` file exists
- Restart terminal and try again

## 📁 Important Files

- `.python-version` - Specifies Python 3.10.6 for pyenv
- `setup_check.py` - Validates Python compatibility
- `activate_env.sh` - Convenient environment activation
- `pyproject.toml` - Enforces Python version requirements
- `requirements.txt` - Contains version warnings

---

Need help? The `setup_check.py` script provides detailed error messages and fix instructions! 