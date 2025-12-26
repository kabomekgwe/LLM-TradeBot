# Testing Patterns

**Analysis Date:** 2025-12-26

## Test Framework

**Runner:**
- pytest 7.4.0+
- Config: No explicit config file (uses defaults)

**Assertion Library:**
- pytest built-in expect
- Matchers: Standard Python assertions, `pytest.raises()`, `pytest.approx()`

**Run Commands:**
```bash
pytest                                    # Run all tests
pytest -v                                 # Verbose output
pytest trading/tests/test_providers.py   # Single file
pytest --cov=trading --cov-report=html   # Coverage report
```

## Test File Organization

**Location:**
- Single test directory: `trading/tests/`
- Pattern: Not co-located with source (centralized test directory)

**Naming:**
- Unit tests: `test_*.py` (e.g., `test_providers.py`)
- No integration or E2E test files detected

**Structure:**
```
trading/
  tests/
    __init__.py
    test_providers.py           # Only existing test file
```

## Test Structure

**Suite Organization:**
```python
import pytest

class TestProviderInterface:
    """Test provider interface compliance."""

    @pytest.mark.parametrize("provider_name", PROVIDERS_TO_TEST)
    def test_provider_creation(self, provider_name):
        # Arrange
        # Act
        # Assert
        pass

class TestDataModels:
    """Test data model validation."""

    def test_ohlcv_validation(self):
        # Arrange
        # Act
        # Assert
        pass
```

**Patterns:**
- Test classes group related functionality
- AAA pattern (Arrange, Act, Assert)
- Parametrized tests for multi-provider testing
- Most tests skipped due to live API requirements

## Mocking

**Framework:**
- No mocking framework detected in requirements
- Tests skip live API calls instead of mocking

**Patterns:**
- `pytest.skip("Requires live API connection or mocking")`
- No mock fixtures observed

**What to Mock:**
- External APIs (not currently mocked - tests skipped instead)
- File system (not currently tested)
- Network calls (not currently mocked)

**What NOT to Mock:**
- Data validation logic
- Pure functions
- Model calculations

## Fixtures and Factories

**Test Data:**
- Inline test data in test methods
- `PROVIDERS_TO_TEST` constant for parametrization
- No separate fixture files

**Location:**
- Test data defined in test file
- No `conftest.py` detected

## Coverage

**Requirements:**
- No enforced coverage target
- pytest-cov available for reporting

**Configuration:**
- No coverage config detected

**View Coverage:**
```bash
pytest --cov=trading --cov-report=html
open htmlcov/index.html
```

**Current State:**
- Very low coverage (only one test file)
- Critical gaps: Agents, ML, Memory, Notifications, Web

## Test Types

**Unit Tests:**
- Scope: Data model validation
- Examples: `test_ohlcv_validation`, `test_position_pnl_calculation`
- Status: Limited coverage

**Integration Tests:**
- Scope: Provider interface compliance
- Examples: `test_provider_creation`, `test_fetch_ohlcv_returns_unified_format`
- Status: Most skipped (require live API)

**E2E Tests:**
- Not detected

## Common Patterns

**Async Testing:**
```python
@pytest.mark.asyncio
async def test_async_operation(self):
    result = await async_function()
    assert result is not None
```

**Parametrized Testing:**
```python
@pytest.mark.parametrize("provider_name", PROVIDERS_TO_TEST)
def test_multiple_providers(self, provider_name):
    provider = create_provider(provider_name)
    assert provider is not None
```

**Error Testing:**
```python
def test_validation_error(self):
    with pytest.raises(ValueError, match="Invalid price"):
        OHLCV(timestamp=..., open=0, ...)  # Invalid data
```

**Skip Pattern:**
```python
def test_live_api(self):
    pytest.skip("Requires live API connection or mocking")
```

## Test Coverage Gaps (Critical)

**Untested Areas:**
- ❌ Agent logic (Bull, Bear, DecisionCore, RiskAudit, Execution) - 0 tests
- ❌ Order management and lifecycle - 0 tests
- ❌ Risk calculations and circuit breakers - 0 tests
- ❌ State persistence and recovery - 0 tests
- ❌ CLI commands - 0 integration tests
- ❌ Notification system - 0 tests
- ❌ ML models and predictions - 0 tests
- ❌ Backtest engine - 0 tests
- ❌ Web dashboard - 0 tests
- ❌ Trade journal and pattern detection - 0 tests

**Tested Areas:**
- ✅ Data model validation (partial)
- ✅ Provider interface (basic, mostly skipped)

**Priority:**
- High: Agent logic, risk management, state management
- Medium: Notifications, ML models, backtest
- Low: Web dashboard (UI testing)

---

*Testing analysis: 2025-12-26*
*Update when test patterns change*
