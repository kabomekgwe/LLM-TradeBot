# Phase 3 Discovery: Pytest Async Testing Patterns & ML Testing

**Research Date:** 2025-12-26
**Phase:** 03-comprehensive-testing
**Scope:** pytest-asyncio patterns, CCXT mocking, ML model testing, market data fixtures

## Research Summary

This discovery confirms current (2025) best practices for testing async Python trading systems with pytest, including patterns for pytest-asyncio, CCXT exchange mocking, LightGBM model testing, and market data fixture design.

## Pytest-Asyncio Patterns (2025)

### Current Version & Changes
- **Latest version**: pytest-asyncio 1.3.0 (released Nov 10, 2025)
- **Major change**: Version 1.0.0 (May 25, 2025) removed the `event_loop` fixture to streamline API
- **Python support**: 3.10-3.14

### Best Practices for Async Fixtures

**1. Use @pytest_asyncio.fixture decorator:**
```python
import pytest_asyncio

@pytest_asyncio.fixture
async def async_client():
    async with httpx.AsyncClient() as client:
        yield client  # Cleanup happens automatically
```

**2. Async context managers pattern:**
- Use async context managers with `yield` for automatic cleanup
- Critical for HTTP clients, database connections, exchange clients

**3. Auto mode recommended:**
```python
# pytest.ini or pyproject.toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
asyncio_default_fixture_loop_scope = "function"
```

**4. Fixture dependency management:**
- Avoid deadlocks by ordering fixture dependencies carefully
- Ensure event loop remains unblocked during setup
- Minimize shared state between fixtures

**5. Loop scope configuration:**
```python
@pytest.mark.asyncio(loop_scope="session")
async def test_with_session_loop():
    # Test runs in session-scoped loop
    pass
```

## CCXT Exchange Mocking

### Current State
- **No official CCXT mock**: CCXT doesn't provide built-in mock/fake exchange
- **Community solutions**: `dsbaars/ccxt.mock` - "quick and dirty" but functional
- **Recommended approach**: Create custom fixtures with AsyncMock

### Mocking Pattern for CCXT

**1. AsyncMock for async methods:**
```python
from unittest.mock import AsyncMock
import pytest_asyncio

@pytest_asyncio.fixture
async def mock_exchange():
    exchange = AsyncMock()

    # Mock async methods
    exchange.fetch_ohlcv = AsyncMock(return_value=[
        [1609459200000, 29000, 29500, 28800, 29200, 1250.5],  # OHLCV format
        [1609462800000, 29200, 29600, 29100, 29400, 1180.2],
    ])

    exchange.fetch_ticker = AsyncMock(return_value={
        'symbol': 'BTC/USDT',
        'last': 29400,
        'bid': 29398,
        'ask': 29402
    })

    exchange.fetch_balance = AsyncMock(return_value={
        'USDT': {'free': 10000, 'used': 0, 'total': 10000},
        'BTC': {'free': 0.5, 'used': 0, 'total': 0.5}
    })

    exchange.create_order = AsyncMock(return_value={
        'id': 'test-order-123',
        'status': 'closed',
        'filled': 0.01
    })

    # Mock async context manager methods
    exchange.__aenter__ = AsyncMock(return_value=exchange)
    exchange.__aexit__ = AsyncMock(return_value=None)

    return exchange
```

**2. Testnet/Demo accounts:**
- Binance, Bybit, OKX support `{'test': True}` parameter
- Recommended for integration tests (not unit tests)

**3. Recorded fixtures:**
- Record real API responses during manual testing
- Replay in tests for consistency
- Useful for edge cases and error scenarios

## Market Data Fixture Patterns

### Factory Fixture Pattern (Recommended)

**Benefits:**
- Reduces test code duplication by 30-60%
- Allows multiple calls with different parameters per test
- Improves readability

**Implementation:**
```python
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class OHLCVCandle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float

@pytest.fixture
def ohlcv_factory():
    """Factory fixture for creating OHLCV data."""
    def _create_ohlcv(
        symbol: str = "BTC/USDT",
        start_price: float = 30000,
        num_candles: int = 100,
        trend: str = "sideways",  # 'uptrend', 'downtrend', 'sideways'
        volatility: float = 0.02
    ):
        candles = []
        current_price = start_price
        base_time = int(datetime.now().timestamp() * 1000)

        for i in range(num_candles):
            timestamp = base_time - (num_candles - i) * 300000  # 5-min candles

            # Apply trend
            if trend == "uptrend":
                current_price *= 1.001  # 0.1% increase per candle
            elif trend == "downtrend":
                current_price *= 0.999  # 0.1% decrease per candle

            # Add randomness
            import random
            open_price = current_price
            high = open_price * (1 + random.uniform(0, volatility))
            low = open_price * (1 - random.uniform(0, volatility))
            close = random.uniform(low, high)
            volume = random.uniform(100, 500)

            candles.append(OHLCVCandle(
                timestamp=timestamp,
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume
            ))

            current_price = close

        return candles

    return _create_ohlcv
```

**Usage:**
```python
def test_bull_agent_uptrend(ohlcv_factory):
    # Create uptrend data
    data = ohlcv_factory(trend="uptrend", num_candles=50)

    # Test agent recognizes uptrend
    result = await bull_agent.execute({'ohlcv_data': {'5m': data}})
    assert result['vote'] > 0
```

### Parametrized Fixtures

**Benefits:**
- Reduces assertion logic by 40%
- Reusable test scenarios
- Better coverage with less code

**Implementation:**
```python
@pytest.fixture(params=[
    {"scenario": "bull_market", "trend": "uptrend", "expected_vote": 1.0},
    {"scenario": "bear_market", "trend": "downtrend", "expected_vote": -1.0},
    {"scenario": "sideways", "trend": "sideways", "expected_vote": 0.0},
])
def market_scenario(request, ohlcv_factory):
    scenario = request.param
    return {
        'data': ohlcv_factory(trend=scenario['trend']),
        'expected': scenario['expected_vote'],
        'name': scenario['scenario']
    }
```

### Fixture Scopes

```python
@pytest.fixture(scope="session")
def trained_lightgbm_model():
    """Expensive: train once per session."""
    # Train model with synthetic data
    return model

@pytest.fixture(scope="function")
def fresh_agent():
    """Cheap: create new instance per test."""
    return PredictAgent(config)
```

## LightGBM Model Testing Patterns

### Test Categories

**1. Model Training Tests:**
```python
def test_model_training():
    """Test model trains successfully with valid data."""
    X_train, y_train = generate_synthetic_features()

    train_data = lgb.Dataset(X_train, label=y_train)
    params = {'objective': 'binary', 'metric': 'binary_logloss'}

    model = lgb.train(params, train_data, num_boost_round=10)

    assert model is not None
    assert model.num_trees() == 10
```

**2. Prediction Tests:**
```python
@pytest.mark.parametrize("features,expected_range", [
    ([75, 0.5, 0.3, 0.2, 30100, 30000, 29900, 0.01], (0.0, 1.0)),  # High RSI
    ([25, -0.5, -0.3, -0.2, 29900, 30000, 30100, -0.01], (0.0, 1.0)),  # Low RSI
])
def test_model_predictions(trained_model, features, expected_range):
    """Test model returns valid probabilities."""
    prediction = trained_model.predict([features])[0]

    assert expected_range[0] <= prediction <= expected_range[1]
    assert isinstance(prediction, (float, np.floating))
```

**3. Mock Trained Model (for agent tests):**
```python
@pytest.fixture
def mock_lightgbm_model():
    """Mock LightGBM model for testing agents without training."""
    mock_model = Mock()
    mock_model.predict = Mock(return_value=np.array([0.75]))  # 75% up probability
    mock_model.best_iteration = 100
    return mock_model
```

**4. Feature Engineering Tests:**
```python
def test_feature_extraction_from_indicators():
    """Test features match training schema."""
    indicators = {
        'rsi': {'value': 65},
        'macd': {'macd': 0.5, 'signal': 0.3, 'histogram': 0.2},
        'bollinger': {'upper': 30100, 'middle': 30000, 'lower': 29900}
    }

    features = extract_features(indicators)

    assert len(features) == 8  # Must match training feature count
    assert features[0] == 65  # RSI
    assert features[1] == 0.5  # MACD line
```

### Behavioral Testing (Black Box)

```python
def test_predict_agent_confidence_scaling():
    """Test confidence scales correctly from probability."""
    # prob=0.5 → confidence=0.0 (neutral)
    # prob=1.0 → confidence=1.0 (max)
    # prob=0.0 → confidence=1.0 (max opposite direction)

    test_cases = [
        (0.5, 0.0),  # Neutral
        (0.75, 0.5),  # Moderate bullish
        (1.0, 1.0),  # Strong bullish
        (0.25, 0.5),  # Moderate bearish
    ]

    for prob, expected_conf in test_cases:
        result = convert_probability_to_confidence(prob)
        assert abs(result - expected_conf) < 0.01
```

## Integration Testing Patterns

### Full Pipeline Test

```python
@pytest.mark.asyncio
async def test_full_agent_pipeline(mock_exchange, ohlcv_factory, mock_lightgbm_model):
    """Test complete decision flow from data fetch to final vote."""
    # Setup
    ohlcv_data = ohlcv_factory(trend="uptrend", num_candles=100)
    mock_exchange.fetch_ohlcv.return_value = ohlcv_data

    # Execute pipeline
    data_sync_result = await data_sync_agent.execute(mock_exchange)
    quant_result = await quant_analyst_agent.execute(data_sync_result)
    predict_result = await predict_agent.execute(quant_result)
    bull_result = await bull_agent.execute(quant_result)
    bear_result = await bear_agent.execute(quant_result)

    decision = decision_core.vote([bull_result, bear_result, predict_result])

    # Verify
    assert decision['action'] in ['buy', 'sell', 'hold']
    assert 0.0 <= decision['confidence'] <= 1.0
```

## Implementation Recommendations

### Test File Structure

```
trading/tests/
├── conftest.py              # Shared fixtures (OHLCV factory, mock exchange)
├── test_agents.py           # Agent logic tests (all 8 agents)
├── test_risk.py             # Risk management tests (100% coverage)
├── test_state.py            # State persistence tests (100% coverage)
├── test_ml.py               # ML model training/prediction tests
├── test_integration.py      # Full pipeline integration
└── fixtures/
    ├── market_data.py       # OHLCV factory, scenarios
    ├── mock_exchange.py     # CCXT mock fixtures
    └── mock_models.py       # LightGBM mock fixtures
```

### Fixture Organization (conftest.py)

```python
# conftest.py
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock

# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)

@pytest.fixture(scope="session")
def anyio_backend():
    return "asyncio"

# Import factory fixtures
from trading.tests.fixtures.market_data import ohlcv_factory
from trading.tests.fixtures.mock_exchange import mock_exchange
from trading.tests.fixtures.mock_models import mock_lightgbm_model

# Re-export for discoverability
__all__ = ['ohlcv_factory', 'mock_exchange', 'mock_lightgbm_model']
```

### Test Markers

```python
# pytest.ini
[pytest]
markers =
    unit: Unit tests (fast, no external dependencies)
    integration: Integration tests (slower, may use mocks)
    slow: Slow tests (model training, large datasets)
    ml: Machine learning tests
    async: Async tests requiring event loop
```

**Usage:**
```python
@pytest.mark.unit
def test_momentum_calculation():
    pass

@pytest.mark.slow
@pytest.mark.ml
def test_lightgbm_training_full():
    pass
```

## Success Criteria

✅ Discovery Complete When:
- [x] pytest-asyncio 1.3.0 patterns confirmed
- [x] CCXT mocking strategy defined (AsyncMock + custom fixtures)
- [x] Market data fixture patterns researched (factory pattern recommended)
- [x] LightGBM testing patterns documented (unit + behavioral)
- [x] Integration testing approach defined
- [x] Test file structure planned

**Ready to proceed with planning.**

---

## Sources

- [Essential pytest asyncio Tips for Modern Async Testing](https://articles.mergify.com/pytest-asyncio-2/)
- [pytest-asyncio PyPI](https://pypi.org/project/pytest-asyncio/)
- [A Practical Guide To Async Testing With Pytest-Asyncio](https://pytest-with-eric.com/pytest-advanced/pytest-asyncio/)
- [Mastering Async Context Manager Mocking in Python Tests](https://dzone.com/articles/mastering-async-context-manager-mocking-in-python)
- [GitHub - dsbaars/ccxt.mock](https://github.com/dsbaars/ccxt.mock)
- [Five Advanced Pytest Fixture Patterns](https://www.inspiredpython.com/article/five-advanced-pytest-fixture-patterns)
- [Testing Machine Learning Projects With Pytest](https://medium.com/@haythemtellili/testing-machine-learning-projects-with-pytest-8c0ae77d392d)
- [PyTest for Machine Learning Tutorial](https://towardsdatascience.com/pytest-for-machine-learning-a-simple-example-based-tutorial-a3df3c58cf8/)
- [LightGBM Official Test Suite](https://github.com/microsoft/LightGBM/blob/master/tests/python_package_test/test_basic.py)
