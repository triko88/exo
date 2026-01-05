# Agent Guidelines for EXO

This document provides essential information for AI coding agents working in the EXO codebase.

## Build, Lint, Test Commands

### Prerequisites
- [uv](https://github.com/astral-sh/uv) for Python dependency management
- [Nix](https://nixos.org/) for reproducible development environment
- Python 3.13+

### Common Commands
```bash
# Sync dependencies
just sync                # Standard sync
just sync-clean          # Force reinstall with no cache

# Format code (uses Nix)
just fmt                 # Auto-format all code
nix fmt                  # Alternative

# Lint
just lint                # Run ruff with auto-fix
uv run ruff check --fix  # Direct ruff command

# Type check
just check               # Run basedpyright type checker
uv run basedpyright --project pyproject.toml

# Run all tests
just test                # Run pytest on src/
uv run pytest src        # Direct pytest command

# Run single test file
uv run pytest src/exo/master/tests/test_topology.py

# Run single test function
uv run pytest src/exo/master/tests/test_topology.py::test_add_node

# Run tests with markers
uv run pytest src -m slow         # Run only slow tests
uv run pytest src -m "not slow"   # Skip slow tests (default)

# Dashboard
just build-dashboard     # Build Svelte dashboard

# Rust bindings
just rust-rebuild        # Rebuild rust bindings

# Package
just package             # Create PyInstaller package

# Clean
just clean               # Remove build artifacts
```

### Running EXO from Source
```bash
git clone https://github.com/exo-explore/exo.git
cd exo/dashboard && npm install && npm run build && cd ..
uv run exo
```

## Code Style Guidelines

### Core Principles
1. **Prioritize code clarity**: well-named types, clear function signatures, robust abstractions
2. **Strict typing**: Never bypass the type-checker; maintain exhaustive typing
3. **Pure functions**: Referentially transparent - same inputs yield same outputs, no hidden state
4. **Eliminate unnecessary branches**: Use type-level discipline to avoid runtime `try/catch` or `if` statements
5. **Remove redundant comments**: Only comment complex code segments; code should be self-documenting

### Python Version
- **Requires Python 3.13+**
- Use all modern Python features from 3.13, 3.12, 3.11

### Imports
- Group imports: standard library, third-party, local
- Use absolute imports from project root: `from exo.shared.types.common import NodeId`
- Standard library imports come first, then third-party, then local
- Use multi-line imports for readability when importing multiple items

Example:
```python
import contextlib
from typing import Iterable

import rustworkx as rx
from pydantic import BaseModel, ConfigDict

from exo.shared.types.common import NodeId
from exo.shared.types.profiling import ConnectionProfile
```

### Naming Conventions
- **No three-letter acronyms** or non-standard contractions
- Choose descriptive, self-explanatory names
- Function signatures alone should convey purpose to a layman
- Class names: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`

### Types and Type Safety
- **Strict type checking**: basedpyright in strict mode with `failOnWarnings = true`
- Use `Literal[...]` for enum-like sets by default
- Use `typing.NewType` for primitives with distinct meaning (zero runtime cost)
- For serializable objects with structural similarity, add a `type: str` field
- Never use `Any` - always provide explicit types
- Use advanced typing features from Python 3.13+

### Pydantic Usage
- **Read and respect Pydantic docs religiously**
- Centralize `ConfigDict` with `frozen=True` and `strict=True`
- Reuse common ConfigDict across models

Example:
```python
from pydantic import BaseModel, ConfigDict

class MyModel(BaseModel):
    field: str
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)
```

- For hierarchies, use discriminated unions: `typing.Annotated[Base, Field(discriminator='variant')]`
- Publish single `TypeAdapter[Base]` for all variants

### UUID/ID Generation
- Inherit from Pydantic's `UUID4` for custom ID types
- Generate fresh IDs with `uuid.uuid4()`
- Idempotency keys: hash persisted state + function-specific salt to avoid collisions after crashes

### Functions and Classes
- **Functions are pure** unless they must mutate fixed state
- Same inputs → same outputs (referentially transparent)
- No hidden state, no unintended I/O
- Side effects go in injectable "effect handlers"; keep core logic pure
- Use classes only to safely interface with fixed, mutable state
- If logic doesn't mutate fixed state, use standalone functions instead of classes

### Error Handling
- **Catch exceptions only where you can handle or transform them meaningfully**
- Document in docstrings WHERE and WHY each exception should be handled
- Don't try-catch without reason
- State the exception handling rationale in docstrings

### Use of `@final` and Immutability
- Mark classes, methods, variables as `@final` or otherwise immutable wherever applicable
- Prefer frozen/immutable data structures

### Dependencies
- **Do not introduce new dependencies without approval**
- Only request libraries common in production environments

### Testing
- Tests live in `tests/` directories alongside source code
- Test files: `test_*.py` or `*_test.py`
- Use pytest with async support (`pytest-asyncio`)
- Test structure: Arrange-Act-Assert pattern with comments
- Use fixtures for common test data
- Mark slow tests: `@pytest.mark.slow`
- Environment variable `EXO_TESTS=1` is automatically set during tests

Example:
```python
import pytest
from exo.shared.topology import Topology

@pytest.fixture
def topology() -> Topology:
    return Topology()

def test_add_node(topology: Topology):
    # arrange
    node_id = NodeId()
    
    # act
    topology.add_node(NodeInfo(node_id=node_id, node_profile=profile))
    
    # assert
    data = topology.get_node_profile(node_id)
    assert data == profile
```

## Repository Workflow

### When You Spot Rule Violations
**DO NOT fix violations in code you weren't asked to work on**. Instead:
1. Inform the user directly
2. File a GitHub issue when applicable

### Commit Message Style
Use imperative mood with change type prefix (max 50 chars):
- `documentation:` - Documentation changes
- `feature:` - New feature
- `refactor:` - Code change that neither fixes bug nor adds feature
- `bugfix:` - Bug fix
- `chore:` - Routine tasks, maintenance, tooling
- `test:` - Adding or correcting tests

Example: `feature: Add support for distributed inference`

### Pull Requests
- Keep changes focused - one feature or fix per PR
- Avoid combining unrelated changes
- Test before and after to demonstrate improvement
- Add automated tests where possible

## Tools and Configuration

### Ruff (Linter)
- Config in `pyproject.toml`
- Enabled rules: I (isort), N (naming), B (bugbear), A (builtins), PIE, SIM
- Excludes: `shared/protobufs/**`, `*mlx_typings/**`, `rust/**`

### Basedpyright (Type Checker)
- Mode: strict with `failOnWarnings = true`
- All type errors are blocking
- Stub path: `.mlx_typings`
- Python version: 3.13
- Reports: Any, Unknown types, Missing stubs all set to "error"

### Pytest
- Async mode: auto
- Python path: project root
- Slow tests marked and deselected by default: `-m 'not slow'`
- Environment: `EXO_TESTS=1` automatically set

## Project Structure
- `src/exo/` - Main source code
  - `master/` - Master node logic
  - `worker/` - Worker node logic
  - `shared/` - Shared types and utilities
  - `utils/` - General utilities
  - `routing/` - Routing logic
- `rust/` - Rust code (networking, bindings)
- `dashboard/` - Svelte frontend
- `docs/` - Documentation
- `.github/` - CI/CD workflows
- Tests live alongside source in `tests/` subdirectories

## Multi-Language Notes
- **Prefer Rust for new code** unless there's a good reason otherwise
- Leverage type systems: Rust's types, Python type hints, TypeScript types
- When writing new code, consider if it should be in Rust for performance
