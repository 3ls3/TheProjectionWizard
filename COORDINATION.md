# Team Coordination Guide

This document outlines the coordination strategy for parallel development on The Projection Wizard.

## 🚀 Current Team Work Allocation

### **Tim**: API Development
**Primary Focus**: FastAPI backend implementation
- **Safe Areas**: `api/`, `tests/unit/api/`
- **Coordination Needed**: `requirements.txt`, `common/schemas.py`
- **Current Sprint**: Pipeline control endpoints, file upload/download APIs

### **Pipeline Developer**: Core Pipeline Improvements  
**Primary Focus**: Performance optimizations and new features
- **Safe Areas**: `pipeline/step_X/`, `scripts/python/`
- **Coordination Needed**: `common/constants.py`, `common/storage.py`
- **Current Sprint**: Data preparation optimizations, validation enhancements

### **Testing Developer**: Comprehensive Test Suite
**Primary Focus**: Unit and integration testing
- **Safe Areas**: `tests/`, `data/test_runs/`  
- **Coordination Needed**: `tests/unit/stage_tests/`, test fixtures
- **Current Sprint**: Unit tests for all pipeline stages, API integration tests

## 📋 Daily Coordination Protocol

### **Morning Sync (5 minutes daily)**
**Time**: 9:00 AM  
**Format**: Slack #projection-wizard channel

**Template**:
```
Yesterday: [What did you complete?]
Today: [What are you working on?]
Blockers: [Any conflicts or dependencies?]
Shared Files: [Any changes to common/, requirements.txt, etc.?]
```

### **Weekly Technical Sync (30 minutes)**
**Time**: Fridays 3:00 PM
**Format**: Video call

**Agenda**:
1. Demo completed features (10 min)
2. Review upcoming changes to shared files (10 min)  
3. Plan next week's work and dependencies (10 min)

## ⚠️ High-Risk Shared Files

### **COORDINATE BEFORE CHANGING**:
- `common/constants.py` - Everyone uses this
- `common/schemas.py` - API and pipeline both modify
- `requirements.txt` - All dependencies
- `pyproject.toml` - Project configuration  
- `README.md` - Documentation

### **Protocol for Shared Files**:
1. **Announce in Slack**: "Planning to modify [filename] for [reason]"
2. **Wait for confirmation**: Get OK from others
3. **Add clear comments**: Document your changes
4. **Test immediately**: Ensure no breaking changes
5. **Notify completion**: "Updated [filename] - please pull latest"

## 🔄 Git Workflow

### **Branch Strategy**
```
main                           # Protected - only merge via PR
├── feature/api-implementation # Tim's branch
├── feature/pipeline-improvements  # Pipeline developer's branch
└── feature/testing-suite     # Testing developer's branch
```

### **Before Pushing Changes**
```bash
# 1. Always check for updates first
git checkout main
git pull origin main

# 2. Rebase your feature branch
git checkout feature/your-branch
git rebase main

# 3. Check for conflicts in shared files
git status
# If you see conflicts in common/, requirements.txt - STOP and coordinate

# 4. Run tests to ensure compatibility
python -m pytest tests/
python scripts/python/health_check.py

# 5. Push and create PR
git push origin feature/your-branch
```

### **Merge Request Protocol**
- **Require 1 review** from another team member
- **Run full test suite** before approving
- **Check shared file changes** carefully
- **Update this document** if workflow changes

## 📁 File Ownership Guidelines

### **Tim (API)**
- **Owner**: `api/`, `tests/unit/api/`
- **Contributor**: `common/schemas.py` (API models)
- **Reviewer**: Pipeline integration points

### **Pipeline Developer**  
- **Owner**: `pipeline/step_X/`, `scripts/python/`
- **Contributor**: `common/constants.py`, `common/storage.py`
- **Reviewer**: Core functionality changes

### **Testing Developer**
- **Owner**: `tests/`, `data/test_runs/`
- **Contributor**: Test fixtures and utilities
- **Reviewer**: All PR changes for test coverage

## 🚨 Emergency Procedures

### **Merge Conflict Resolution**
1. **Don't panic** - conflicts are normal
2. **Communicate immediately** in Slack
3. **Schedule quick sync** to resolve together
4. **Document resolution** for future reference

### **Breaking Changes**
If someone accidentally breaks the build:
1. **Notify immediately** in Slack
2. **Create hotfix branch** from last good commit
3. **Fix quickly** or revert the problematic change
4. **Review process** to prevent future occurrences

### **Blocked Dependencies**
If you're blocked waiting for someone else's work:
1. **Create interface contracts** - agree on function signatures
2. **Use mock implementations** temporarily  
3. **Work on parallel features** that don't depend on the blocking work
4. **Pair program** if the dependency is critical

## 📊 Success Metrics

### **Weekly Goals**
- [ ] Zero breaking commits to main
- [ ] All shared file changes coordinated
- [ ] Feature branches stay current with main
- [ ] Test suite passes on all branches

### **Sprint Goals (2 weeks)**
- [ ] Each developer completes 2-3 features
- [ ] Integration between all components works
- [ ] Documentation stays current
- [ ] Performance benchmarks improve

## 📝 Communication Channels

- **Daily coordination**: Slack #projection-wizard
- **Technical discussions**: GitHub PR comments
- **Emergency issues**: Direct message + Slack mention
- **Weekly planning**: Video call

## 🔧 Tools and Automation

### **Recommended IDE Settings**
```json
// .vscode/settings.json (add to your personal settings)
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true
}
```

### **Pre-commit Hooks** (Optional but Recommended)
```bash
# Install pre-commit
pip install pre-commit

# Set up hooks (run this once)
pre-commit install

# Hooks will automatically run on commit
```

---

**Last Updated**: [Date]
**Next Review**: [Date + 1 week]

*This document is living and should be updated as our workflow evolves.* 