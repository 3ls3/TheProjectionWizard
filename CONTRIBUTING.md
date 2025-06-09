# Contributing to The Projection Wizard

Welcome to the team! This guide will help you contribute effectively while working in parallel with other developers.

## 🚀 Quick Start

### **1. Set Up Your Development Environment**
```bash
# Clone and set up the project
git clone [repository-url]
cd TheProjectionWizard

# Create your feature branch immediately
git checkout -b feature/[your-work-area]
# Examples:
# git checkout -b feature/api-implementation
# git checkout -b feature/pipeline-improvements
# git checkout -b feature/testing-suite

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify setup
python scripts/python/health_check.py
```

### **2. Understand Your Work Area**
- **API Developer**: Focus on `api/`, `tests/unit/api/`
- **Pipeline Developer**: Focus on `pipeline/step_X/`, `scripts/python/`
- **Testing Developer**: Focus on `tests/`, `data/test_runs/`

### **3. Read the Interface Documentation**
- Study `docs/INTERFACES.md` for integration points
- Check `COORDINATION.md` for team workflow
- Understand shared file protocols


### **Before Making Changes**
```bash
# Always start with latest main
git checkout main
git pull origin main

# Rebase your feature branch
git checkout feature/your-branch
git rebase main

# If conflicts in shared files (common/, requirements.txt):
# STOP and coordinate with team before resolving
```

### **Development Loop**
```bash
# 1. Make your changes
# 2. Test your changes
python -m pytest tests/
python scripts/python/health_check.py

# 3. Check for shared file conflicts
git status
# If you see common/constants.py, requirements.txt, etc. - coordinate!

# 4. Commit with clear messages
git add .
git commit -m "feat: add pipeline control endpoints

- Add POST /api/v1/runs/upload endpoint
- Add GET /api/v1/runs/{run_id}/status endpoint  
- Update requirements.txt with FastAPI dependencies"

# 5. Push regularly
git push origin feature/your-branch
```

## ⚠️ Shared File Protocols

### **High-Risk Files (Coordinate Before Changing)**
- `common/constants.py`
- `common/schemas.py`  
- `requirements.txt`
- `pyproject.toml`
- `README.md`

### **Protocol for Shared Files**
1. **Announce in Slack**: "Planning to modify requirements.txt to add FastAPI"
2. **Wait for OK**: Get confirmation from team
3. **Make minimal changes**: Only add what you need
4. **Document changes**: Add comments explaining your additions
5. **Test immediately**: Ensure no breaking changes
6. **Notify completion**: "Updated requirements.txt - ready for team to pull"

### **Example: Adding Dependencies**
```python
# requirements.txt
# Core ML and Data Processing
pandas>=2.0.0
numpy>=1.24.0
# ... existing dependencies ...

# =============================================================================
# API Development Dependencies (Tim - December 2024)
# =============================================================================
fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6  # For file uploads

# =============================================================================
# Testing Dependencies (Teammate - December 2024)  
# =============================================================================
pytest-asyncio>=0.21.0
httpx>=0.25.0  # For API testing
```

## 🧪 Testing Guidelines

### **Test Your Own Work First**
```bash
# Unit tests for your changes
python -m pytest tests/unit/ -v

# Integration tests
python -m pytest tests/integration/ -v

# Health check (overall system)
python scripts/python/health_check.py

# If working on API
python -m pytest tests/unit/api/ -v

# If working on pipeline
python -m pytest tests/unit/stage_tests/ -v
```

### **Test Patterns to Follow**
```python
# For API endpoints
def test_my_endpoint():
    response = client.get("/api/v1/my-endpoint")
    assert response.status_code == 200
    assert "expected_field" in response.json()

# For pipeline stages  
def test_my_stage():
    test_run_id = setup_test_environment()
    success = my_stage_function(test_run_id)
    assert success
    assert_expected_outputs_exist(test_run_id)

# For utilities
def test_my_utility():
    result = my_utility_function(test_input)
    assert result == expected_output
    assert isinstance(result, expected_type)
```

## 🔍 Code Quality Standards

### **Code Formatting (Required)**
```bash
# Run before every commit
black .
flake8 .
mypy .

# Or use pre-commit hooks (recommended)
pip install pre-commit
pre-commit install
```

### **Documentation Standards**
```python
def my_function(param1: str, param2: int) -> bool:
    """
    Brief description of what the function does.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When invalid input provided
    """
    pass
```

### **Logging Standards**
```python
from common import logger

# Get stage-specific logger
log = logger.get_stage_logger(run_id, "my_stage")

# Use consistent logging
log.info("Starting process X")
log.warning("Potential issue detected")
log.error(f"Process failed: {error}")

# For structured logging (monitoring)
structured_log = logger.get_stage_structured_logger(run_id, "my_stage")
logger.log_structured_event(
    structured_log,
    "event_type",
    {"key": "value"},
    "Human readable message"
)
```

## 🔄 Pull Request Guidelines

### **Before Creating PR**
- [ ] All tests pass locally
- [ ] Code is formatted (black, flake8, mypy)
- [ ] Shared file changes are coordinated
- [ ] Documentation is updated if needed
- [ ] Health check passes

### **PR Title Format**
```
type(scope): brief description

Examples:
feat(api): add pipeline control endpoints
fix(pipeline): resolve memory leak in prep stage
test(validation): add comprehensive unit tests
docs(readme): update installation instructions
```

### **PR Description Template**
```markdown
## What This PR Does
Brief description of changes

## Type of Change
- [ ] New feature
- [ ] Bug fix
- [ ] Documentation update
- [ ] Test improvement
- [ ] Refactoring

## Shared Files Changed
- [ ] common/constants.py - [describe changes]
- [ ] requirements.txt - [describe additions]
- [ ] None

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Health check passes
- [ ] Added new tests for new functionality

## Coordination
- [ ] Team notified of shared file changes
- [ ] No breaking changes to interfaces
- [ ] Documentation updated

## Screenshots/Examples
[If applicable]
```

### **Review Process**
- **Required**: 1 approval from another team member
- **API PRs**: Pipeline developer should review integration points
- **Pipeline PRs**: API developer should review interface changes
- **Testing PRs**: Both developers should review test coverage

## 🚨 Emergency Procedures

### **I Broke the Build**
1. **Don't panic** - it happens to everyone
2. **Immediately notify team** in Slack
3. **Identify the problem** - run health check
4. **Quick fix or revert**:
   ```bash
   # Option 1: Quick fix
   git add .
   git commit -m "hotfix: resolve build break"
   git push origin feature/your-branch
   
   # Option 2: Revert problematic commit
   git revert HEAD
   git push origin feature/your-branch
   ```
5. **Communicate resolution** to team

### **Merge Conflicts in Shared Files**
1. **Don't resolve alone** - these need coordination
2. **Message team immediately**: "Merge conflict in common/constants.py"
3. **Schedule quick call** to resolve together
4. **Document resolution** for future reference

### **Blocked by Dependencies**
1. **Communicate early**: "Blocked waiting for X feature"
2. **Work on alternatives**: Use mock implementations
3. **Pair program**: If dependency is critical
4. **Update team**: When dependency is resolved

## 🎯 Success Metrics

### **Individual Goals**
- Zero breaking commits to main
- All PRs include tests
- Code quality checks pass
- Regular communication with team

### **Team Goals**
- Smooth integration between all components
- No duplicate work
- Shared understanding of interfaces
- Continuous progress on all fronts

## 📚 Resources

### **Key Documentation**
- `COORDINATION.md` - Team workflow
- `docs/INTERFACES.md` - Technical interfaces
- `README.md` - Project overview
- `api/README.md` - API specific documentation

### **Important Commands**
```bash
# Health check
python scripts/python/health_check.py

# Run all tests
python -m pytest tests/ -v

# Code quality
black . && flake8 . && mypy .

# Start development servers
streamlit run app/main.py          # UI
uvicorn api.main:app --reload      # API
python scripts/python/run_pipeline_cli.py --csv data/fixtures/sample_classification.csv  # CLI
```

### **Getting Help**
- **Slack**: #projection-wizard for daily coordination
- **GitHub**: Use PR comments for technical discussions
- **Emergency**: Direct message + Slack mention
- **Weekly sync**: Fridays 3 PM for deeper technical discussions

---

**Remember**: Communication is key to successful parallel development. When in doubt, ask the team!

**Last Updated**: [Current Date]
**Next Review**: [Weekly with team updates] 