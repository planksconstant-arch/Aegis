# Contributing to Local IDE RL Agent

Thank you for your interest in contributing! This guide covers everything you need to get started.

---

## Setting Up a Development Environment

### Prerequisites
- Python 3.11 or 3.12
- Git

### Install

```bash
git clone https://github.com/your-org/local-ide-agent
cd local-ide-agent
pip install -e ".[dev]"
```

This installs the package in editable mode with test dependencies (`pytest`, `pytest-cov`).

### Verify the install

```bash
local-ide-agent train --episodes 3
```

You should see episode logs printed to the terminal.

---

## Running Tests

```bash
# All tests
pytest

# With coverage report
pytest --cov=src --cov-report=term-missing

# Single file
pytest tests/test_policy.py -v
```

All tests must pass before a PR is merged.

---

## Project Layout

```
.
├── src/local_ide_agent/
│   ├── agent/          # Core agent runtime + trajectory buffer
│   ├── cli/            # Terminal dashboard
│   ├── connectors/     # IDE filesystem + event log connector
│   ├── deployment/     # Background agent manager
│   ├── lab/            # Counterfactual experiment lab
│   ├── memory/         # SQLite store (feedback, replay, success rates)
│   ├── rl/             # Neural core: policy, curiosity, n-step, eval, replay
│   ├── shadow/         # Safe shadow-workspace cloning
│   ├── training/       # Environment, curriculum, training loop
│   ├── bridge.py       # Local HTTP bridge (IDE → agent)
│   ├── config.py       # Pydantic settings with YAML loading
│   ├── main.py         # CLI entrypoint
│   └── schemas.py      # Shared data models
├── tests/              # pytest test suite
├── pyproject.toml
├── settings.example.yaml
├── CHANGELOG.md
└── LICENSE
```

---

## Design Constraints

Before contributing, please read the core design decisions:

### Pure Numpy — No Deep Learning Frameworks
All neural network primitives (MLP, Adam, LayerNorm, Huber loss, backward passes) are implemented in `rl/nn.py` using only NumPy. **Do not add PyTorch, TensorFlow, or JAX as required dependencies.** They are allowed in the optional `[models]` extra only.

### SQLite — No External Database
The memory store uses only the Python stdlib `sqlite3` module. No PostgreSQL, Redis, or MongoDB.

### No External API Calls in Core Training
The training loop must work offline. Network calls are only allowed in connectors behind an interface, never in `rl/`, `training/`, or `agent/`.

### Pydantic v2 for All Data Models
All structured data uses `pydantic.BaseModel`. Do not use plain dataclasses for public API surfaces.

---

## Making Changes

### Coding Style
- Follow PEP 8
- Use type annotations on all public functions
- Docstrings on all classes and non-trivial methods
- Keep modules small and focused — prefer a new file over a 500-line god module

### Branching
- `main` — stable, always passes tests
- `dev` — integration branch for new features
- Feature branches: `feat/your-feature-name`
- Bug fixes: `fix/issue-description`

### Commit Messages
Follow [Conventional Commits](https://www.conventionalcommits.org/):
```
feat: add Neovim connector
fix: prevent RND cache corruption in multi-step episodes
docs: update README with dashboard instructions
test: add curriculum promotion/demotion tests
```

### Pull Request Checklist
- [ ] Tests added for new behaviour
- [ ] All existing tests pass (`pytest`)
- [ ] `CHANGELOG.md` updated under `[Unreleased]`
- [ ] Type annotations present on new functions
- [ ] No new required dependencies unless discussed in an issue first

---

## Adding a New Action

1. Add a new `StrategyAction` to `rl/actions.py`
2. Add its reward profile to `training/environment.py` `_REWARD_TABLE`
3. Add soft-masking logic (if appropriate) in `rl/policy.py` `_compute_logit_penalties()`
4. Add a test in `tests/test_policy.py`

---

## Adding a New IDE Connector

1. Implement the interface from `agent/policy.py` `Policy`
2. Add a new file in `connectors/` (e.g. `connectors/neovim.py`)
3. Register it in `config.py` `EncoderBackendSettings`
4. Document it in `README.md`

---

## Reporting Bugs

Open a GitHub issue with:
- Python version (`python --version`)
- OS
- Steps to reproduce
- Full traceback
- Output of `local-ide-agent run` (if the issue is agent-related)

---

## Questions?

Open a Discussion on GitHub or a draft PR — we're happy to help.
