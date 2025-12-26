# Coding Conventions

**Analysis Date:** 2025-12-26

## Naming Patterns

**Files:**
- Python modules: `snake_case.py` (e.g., `base_agent.py`, `quant_analyst.py`, `binance_futures.py`)
- Test files: `test_*.py` (e.g., `test_providers.py`)
- Config files: `.env.example`, `requirements.txt`

**Functions:**
- Pattern: `snake_case`
- Async: Same naming (e.g., `async def execute()`, `async def fetch_ohlcv()`)
- Queries: No prefix (e.g., `get_agent_name()`, `is_profitable()`)
- Actions: Verb prefix (e.g., `fetch_`, `create_`, `calculate_`)

**Variables:**
- Pattern: `snake_case` (e.g., `market_data`, `bull_vote`, `recent_closes`)
- Constants: `SCREAMING_SNAKE_CASE` (e.g., `MAX_POSITION_SIZE_USD`, `PROVIDERS_TO_TEST`)
- Booleans: `is_*` prefix (e.g., `is_uptrend`, `is_bullish`)

**Types:**
- Classes: `PascalCase` (e.g., `TradingConfig`, `BaseAgent`, `DataSyncAgent`)
- Dataclasses: `PascalCase` (e.g., `OHLCV`, `Ticker`, `Position`)
- Enums: `PascalCase` for class, values vary (e.g., `VoteAction.BUY`, `NotificationLevel.INFO`)
- Type aliases: `PascalCase`

## Code Style

**Formatting:**
- Indentation: 4 spaces (Python standard)
- Line length: 80-100 characters
- Quotes: Double quotes `"` for strings and docstrings
- Semicolons: Not used (Python)

**Linting:**
- No explicit linter configuration detected
- Follows PEP 8 style guide (inferred from code)
- No `.flake8`, `.pylintrc`, or `pyproject.toml` found

## Import Organization

**Order:**
1. Standard library (e.g., `logging`, `os`, `dataclasses`)
2. Third-party packages (e.g., `ccxt`, `pandas`, `fastapi`)
3. Local imports (e.g., `from trading.models import ...`)
4. Type imports last (e.g., `from typing import Optional`)

**Grouping:**
- Blank lines between groups
- No explicit sorting within groups

**Path Aliases:**
- None used (relative imports: `from trading.agents import BaseAgent`)

## Error Handling

**Patterns:**
- Broad exception catching: `except Exception as e:` (50+ occurrences - technical debt)
- Logging before re-raising
- No custom exception hierarchy

**Error Types:**
- Mostly generic `Exception` catches
- Should use specific exceptions (ValueError, KeyError, etc.) - improvement needed

**Async:**
- try/catch in async functions
- No timeout handling on most async operations (technical debt)

## Logging

**Framework:**
- Python logging module
- File output: `logs/trading.log`
- Configurable via `LOG_LEVEL` env var

**Patterns:**
- `logger.info()`, `logger.warning()`, `logger.error()`
- Mixed with `print()` statements (79 occurrences - should be migrated to logging)

**Levels:**
- DEBUG, INFO, WARNING, ERROR
- No CRITICAL level usage observed

## Comments

**When to Comment:**
- Complex business logic
- TODO markers for incomplete features
- Docstrings for all public functions/classes

**JSDoc/TSDoc:**
- Not applicable (Python codebase)

**Python Docstrings:**
- Style: Google-style docstrings (PEP 257)
- Format: Multi-line with Args, Returns, Raises, Example sections
- Examples:
  ```python
  """One-line summary.

  Args:
      param1: Description
      param2: Description

  Returns:
      Return type and description

  Raises:
      ExceptionType: When this happens

  Example:
      >>> code_example()
      Output
  """
  ```

**TODO Comments:**
- Format: `# TODO: Description`
- Common pattern: Mark incomplete agent implementations
- Examples: `trading/agents/predict.py:18`, `trading/agents/quant_analyst.py:18`

## Function Design

**Size:**
- No strict limit observed
- Some functions exceed 50 lines

**Parameters:**
- Context dictionary pattern: `context: dict[str, Any]`
- Dataclass parameters for complex inputs
- No strict parameter count limit

**Return Values:**
- Explicit return types with type hints
- Dictionary returns common (context pattern)
- Dataclass returns for structured data

## Module Design

**Exports:**
- Named exports preferred
- No default exports (not Python pattern)
- `__all__` not commonly used

**Package Structure:**
- `__init__.py` files for packages
- Flat import hierarchy

## Type Hints

**Style:**
- Modern Python 3.10+ syntax
- `dict[str, Any]` instead of `Dict[str, Any]`
- `list[OHLCV]` instead of `List[OHLCV]`
- `Optional[str]` for nullable types
- `tuple[bool, Optional[str]]` for tuples

**Coverage:**
- Extensive type hints throughout
- Some `dict[str, Any]` usage (could be more specific)

---

*Convention analysis: 2025-12-26*
*Update when patterns change*
