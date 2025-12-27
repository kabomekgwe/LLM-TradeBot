# Phase 4 Discovery: Structured Logging & Exception Patterns

**Research Date:** 2025-12-27
**Phase:** 04-decision-transparency-error-handling
**Scope:** Structured logging, custom exception hierarchies, async timeouts, decision tracing

## Research Summary

This discovery confirms current (2025) best practices for Python structured logging, custom exception design, async timeout handling, and decision transparency. All patterns are well-established with clear industry standards.

## Structured Logging Patterns (2025)

### Current Best Practices

**1. Use JSON Format for Production:**
- JSON structured logging turns messy text into clean, organized data
- Ideal for production systems or microservices where logs must be parsed by machines
- Standard format for log aggregation platforms (Datadog, New Relic, etc.)

**2. Popular Libraries:**

**python-json-logger** (recommended for simple cases):
```python
from pythonjsonlogger import jsonlogger

logger = logging.getLogger()
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter()
logHandler.setFormatter(formatter)
logger.addHandler(logHandler)

# Usage
logger.info("User login", extra={"user_id": "123", "ip": "192.168.1.1"})
```

**structlog** (recommended for advanced use):
```python
import structlog

structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
)

log = structlog.get_logger()
log.info("trade_executed", symbol="BTC/USDT", side="buy", quantity=0.01)
```

**3. Consistent Timestamps:**
- ISO 8601 format recommended for easy sorting and tracing
- Example: `2025-12-27T10:30:45.123456Z`

**4. Contextual Information (Most Important):**
- Use `extra` dictionary for application-specific context
- Correlation ID / Request ID for tracing across services
- Example context fields:
  - `agent_name`: Which agent is logging
  - `symbol`: Trading pair being analyzed
  - `decision_id`: UUID linking related logs
  - `confidence`: Decision confidence score

**5. Log to stdout in Containers:**
- Modern cloud/container environments expect stdout logging
- Let orchestration layer handle log collection (Docker, Kubernetes)

**6. Configuration Best Practices:**
- Use `logging.getLogger(__name__)` for module-level loggers
- Configure levels: DEBUG for dev, INFO+ for production
- Implement log rotation for file outputs
- Use QueueHandler for thread safety
- NEVER log sensitive data (API keys, passwords)

### Migration Strategy

**Replace print() with structured logging:**
```python
# ❌ Current (79 instances)
print(f"Bull agent vote: {vote}, confidence: {confidence}")

# ✅ Target
logger.info(
    "agent_vote",
    extra={
        "agent": "bull",
        "vote": vote,
        "confidence": confidence,
        "symbol": symbol,
        "decision_id": decision_id
    }
)
```

**Log Levels Guide:**
- DEBUG: Detailed diagnostic info (indicator values, intermediate calculations)
- INFO: Normal operations (agent votes, decisions, order execution)
- WARNING: Unexpected but recoverable (rate limits, retries)
- ERROR: Failures requiring attention (API errors, rejected orders)
- CRITICAL: System-threatening failures (circuit breaker trip, state corruption)

## Custom Exception Hierarchies (2025)

### Best Practices

**1. Inherit from Exception, Not BaseException:**
- BaseException includes low-level classes (SystemExit, KeyboardInterrupt)
- Custom exceptions should derive from Exception

**2. Choose the Right Parent Class:**
- Exception for general cases
- ValueError for input validation errors
- RuntimeError for operational failures
- Create domain-specific base exception for library/project

**3. Common Root Exception Pattern:**
```python
# Base exception for entire project
class TradingBotError(Exception):
    """Base exception for all trading bot errors."""
    pass

# Domain-specific categories
class ConfigurationError(TradingBotError):
    """Configuration-related errors."""
    pass

class APIError(TradingBotError):
    """External API communication errors."""
    pass

class RiskViolationError(TradingBotError):
    """Risk management constraint violations."""
    pass

class AgentError(TradingBotError):
    """Agent execution errors."""
    pass

# Specific exceptions
class InsufficientBalanceError(RiskViolationError):
    """Insufficient balance to execute trade."""
    pass

class PositionLimitExceededError(RiskViolationError):
    """Position would exceed max position size."""
    pass

class ExchangeConnectionError(APIError):
    """Cannot connect to exchange API."""
    pass

class InvalidIndicatorDataError(AgentError):
    """Indicator data is invalid or incomplete."""
    pass
```

**4. Naming and Documentation:**
- End with "Error" or "Exception"
- Add docstrings documenting what triggers the exception
- Include context in exception messages

**5. Organization:**
- Place all custom exceptions in `trading/exceptions.py`
- Import where needed: `from trading.exceptions import RiskViolationError`

**6. Modern Python Features (3.11+):**
- Exception Groups for handling multiple exceptions simultaneously
- Useful for concurrent operations (multiple agent failures)

### Exception Hierarchy for Trading Bot

```
TradingBotError (root)
├── ConfigurationError
│   ├── MissingCredentialError
│   └── InvalidConfigValueError
├── APIError
│   ├── ExchangeConnectionError
│   ├── RateLimitExceededError
│   └── OrderRejectedError
├── RiskViolationError
│   ├── InsufficientBalanceError
│   ├── PositionLimitExceededError
│   ├── DailyDrawdownExceededError
│   └── CircuitBreakerTrippedError
├── AgentError
│   ├── InvalidIndicatorDataError
│   ├── ModelPredictionError
│   └── AgentTimeoutError
└── StateError
    ├── StateCorruptedError
    └── StateSaveFailedError
```

### Replacing Generic Exception Catches

```python
# ❌ Current (50+ instances)
try:
    result = await agent.execute(context)
except Exception as e:
    logger.error(f"Agent failed: {e}")
    return None

# ✅ Target
try:
    result = await agent.execute(context)
except InvalidIndicatorDataError as e:
    logger.warning("agent_skipped", extra={"reason": "invalid_data", "error": str(e)})
    return {"vote": 0.0, "confidence": 0.0, "reason": "insufficient_data"}
except AgentTimeoutError as e:
    logger.error("agent_timeout", extra={"agent": agent.name, "timeout_seconds": 30})
    raise  # Re-raise timeout errors
except TradingBotError as e:
    logger.error("agent_error", extra={"error_type": type(e).__name__, "message": str(e)})
    raise  # Re-raise known errors
except Exception as e:
    # Only catch truly unexpected errors
    logger.critical("agent_unexpected_error", extra={"error": str(e)}, exc_info=True)
    raise
```

## Async Timeout Patterns (2025)

### Primary Methods

**1. asyncio.wait_for() (Python 3.7+):**
```python
try:
    result = await asyncio.wait_for(
        exchange.fetch_ohlcv(symbol, timeframe),
        timeout=30.0  # 30 seconds
    )
except asyncio.TimeoutError:
    logger.error("fetch_timeout", extra={"symbol": symbol, "timeframe": timeframe})
    raise AgentTimeoutError(f"Failed to fetch {symbol} {timeframe} data within 30s")
```

**2. asyncio.timeout() Context Manager (Python 3.11+):**
```python
async with asyncio.timeout(30.0):
    result = await exchange.fetch_ohlcv(symbol, timeframe)
# TimeoutError raised automatically if exceeds 30s
```

### Best Practices

**When to Use Timeouts:**
- Any external system interaction (exchange APIs, sentiment APIs)
- File I/O operations
- Database queries
- Long-running computations

**Recommended Timeouts:**
- Quick API calls (ticker, balance): 5-10s
- OHLCV data fetches: 15-30s
- Order placement: 10-15s
- ML model predictions: 5-10s
- Agent execution: 30-60s

**Decorator Pattern for Reusability:**
```python
def with_timeout(seconds: float):
    """Decorator to add timeout to async functions."""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                async with asyncio.timeout(seconds):
                    return await func(*args, **kwargs)
            except TimeoutError:
                raise AgentTimeoutError(
                    f"{func.__name__} exceeded {seconds}s timeout"
                )
        return wrapper
    return decorator

# Usage
@with_timeout(30.0)
async def fetch_market_data(symbol: str, timeframe: str):
    return await exchange.fetch_ohlcv(symbol, timeframe)
```

**Using asyncio.shield():**
- Prevents cancellation of critical operations
- Use for cleanup, state saves, notification sends
```python
try:
    async with asyncio.timeout(30.0):
        result = await risky_operation()
except TimeoutError:
    # Shield state save from cancellation
    await asyncio.shield(state.save())
    raise
```

### Timeout Strategy for Trading Bot

| Operation | Timeout | Retry | Reason |
|-----------|---------|-------|--------|
| `fetch_ohlcv()` | 30s | 2x | Historical data can be slow |
| `fetch_ticker()` | 10s | 1x | Real-time data should be fast |
| `create_order()` | 15s | 0x | Don't retry order placement |
| `fetch_balance()` | 10s | 1x | Account data usually fast |
| Agent `execute()` | 60s | 0x | Agent should complete quickly |
| ML `predict()` | 10s | 0x | Model inference fast |
| State `save()` | 5s | 0x | File write should be instant |

## Decision Tracing & Transparency (2025)

### AI Audit Trail Principles

**Key Concept:** An AI audit trail is a detailed record of inputs, outputs, model behavior, and decision logic at every step of a workflow.

**Why it matters:**
- Regulatory compliance (especially for financial applications)
- Debugging unexpected decisions
- Learning from past decisions
- Transparency for users

### Decision Context Pattern

**Create decision_id for correlation:**
```python
import uuid

decision_id = str(uuid.uuid4())

# All logs for this decision include decision_id
logger.info("decision_start", extra={"decision_id": decision_id, "symbol": symbol})

# Agent votes
logger.info("agent_vote", extra={
    "decision_id": decision_id,
    "agent": "bull",
    "vote": 1.0,
    "confidence": 0.75,
    "factors": ["RSI oversold", "MACD bullish"]
})

# Final decision
logger.info("decision_final", extra={
    "decision_id": decision_id,
    "action": "buy",
    "confidence": 0.82,
    "votes": {"bull": 1.0, "bear": 0.0, "predict": 0.8}
})

# Execution result
logger.info("order_executed", extra={
    "decision_id": decision_id,
    "order_id": "ABC123",
    "filled": 0.01,
    "price": 29450.0
})
```

### Decision Record Structure

```python
@dataclass
class DecisionRecord:
    """Complete audit trail for a trading decision."""
    decision_id: str
    timestamp: datetime
    symbol: str

    # Inputs
    market_data: dict  # OHLCV, ticker, indicators
    agent_votes: dict[str, dict]  # Each agent's vote + reasoning

    # Decision
    final_action: str  # buy, sell, hold
    final_confidence: float
    weighted_vote: float
    decision_reason: str

    # Risk Audit
    risk_checks: dict[str, bool]
    risk_vetoed: bool
    veto_reason: Optional[str]

    # Execution
    order_id: Optional[str]
    execution_price: Optional[float]
    execution_quantity: Optional[float]
    execution_status: str

    # Outcome (populated later)
    pnl: Optional[float]
    success: Optional[bool]
```

**Store in TradeJournal with full context:**
- Enable post-mortem analysis
- Identify patterns in successful/failed decisions
- Train ML models on decision quality

### Transparency in Agent Reasoning

**Current problem:** Agents return votes but reasoning is opaque
**Solution:** Include factor breakdown in agent output

```python
# Bull agent output
{
    "vote": 1.0,
    "confidence": 0.75,
    "direction": "bullish",
    "factors": [
        {"name": "RSI oversold", "value": 28.5, "weight": 0.4, "signal": "bullish"},
        {"name": "MACD crossover", "histogram": 0.15, "weight": 0.3, "signal": "bullish"},
        {"name": "Price at lower BB", "position": "lower", "weight": 0.3, "signal": "bullish"}
    ],
    "reason": "RSI oversold (28.5); MACD bullish crossover (hist=0.15); Price near lower BB"
}
```

## Implementation Recommendations

### File Structure

```
trading/
├── exceptions.py          # All custom exceptions
├── logging_config.py      # Structured logging setup
├── utils/
│   ├── timeout.py         # Timeout decorators and utilities
│   └── decision_trace.py  # Decision tracing utilities
└── agents/
    └── base_agent.py      # Add timeout + logging to base
```

### Migration Order

**Phase 1: Foundation**
1. Create `trading/exceptions.py` with exception hierarchy
2. Create `trading/logging_config.py` with JSON logger setup
3. Update `trading/agents/base_agent.py` to use structured logging

**Phase 2: Replace print() Statements**
1. Identify all 79 print() locations
2. Replace with appropriate log level (INFO, WARNING, ERROR)
3. Add contextual information (agent name, symbol, decision_id)

**Phase 3: Replace Generic Exceptions**
1. Identify all 50+ `except Exception` blocks
2. Determine specific exception types for each case
3. Replace with targeted exception handling
4. Add proper error recovery or re-raising

**Phase 4: Add Timeouts**
1. Add timeouts to all external API calls (CCXT operations)
2. Add timeouts to agent execute() methods
3. Add timeouts to ML predictions
4. Test timeout behavior with slow/hanging operations

**Phase 5: Decision Tracing**
1. Generate decision_id for each trading loop
2. Add decision_id to all related logs
3. Create DecisionRecord structure
4. Enhance TradeJournal to store full decision context

## Success Criteria

✅ Discovery Complete When:
- [x] Structured logging patterns researched (python-json-logger, structlog)
- [x] Custom exception hierarchy designed (TradingBotError root, 4 categories)
- [x] Async timeout patterns documented (asyncio.wait_for, asyncio.timeout)
- [x] Decision tracing approach defined (decision_id correlation, DecisionRecord)
- [x] Migration strategy outlined (5 phases)

**Ready to proceed with planning.**

---

## Sources

**Structured Logging:**
- [Python Logging Format Tutorial with Examples](https://middleware.io/blog/python-logging-format/)
- [Structured Logging in Python: The Key to Observability | Hrekov](https://www.hrekov.com/blog/python-structured-logging)
- [Application Logging in Python: Recipes for Observability · Dash0](https://www.dash0.com/guides/logging-in-python)
- [Python Logging Best Practices: Complete Guide 2025](https://www.carmatec.com/blog/python-logging-best-practices-complete-guide/)
- [Python Logging Best Practices | SigNoz](https://signoz.io/guides/python-logging-best-practices/)
- [A Comprehensive Guide to Python Logging with Structlog | Better Stack](https://betterstack.com/community/guides/logging/structlog/)
- [Guide to structured logging in Python](https://newrelic.com/blog/log/python-structured-logging)

**Custom Exceptions:**
- [6 Best practices for Python exception handling](https://www.qodo.ai/blog/6-best-practices-for-python-exception-handling/)
- [Mastering Python Exception Hierarchy in 2025 | Toxigon](https://toxigon.com/python-exception-hierarchy)
- [Writing a Python Custom Exception — CodeSolid](https://codesolid.com/writing-a-python-custom-exception/)
- [Mastering Custom Python Exceptions: Best Practices](https://pythonprograming.com/blog/mastering-custom-python-exceptions-best-practices-use-cases-and-expert-tips)
- [How to Properly Declare Custom Exceptions in Modern Python](https://www.pythontutorials.net/blog/proper-way-to-declare-custom-exceptions-in-modern-python/)

**Async Timeouts:**
- [Coroutine Timeout Injection in Python: A Decorator Approach](https://medium.com/@RampantLions/coroutine-timeout-injection-in-python-a-decorator-approach-with-asyncio-wait-for-171dc0a3f5be)
- [Coroutines and Tasks — Python 3.14.2 documentation](https://docs.python.org/3/library/asyncio-task.html)
- [asyncio.timeout() To Wait and Cancel Tasks - Super Fast Python](https://superfastpython.com/asyncio-timeout/)
- [A Complete Guide to Timeouts in Python | Better Stack](https://betterstack.com/community/guides/scaling-python/python-timeouts/)
- [Asyncio Timeout Best Practices - Super Fast Python](https://superfastpython.com/asyncio-timeout-best-practices/)
- [Asyncio in Python — The Essential Guide for 2025](https://medium.com/@shweta.trrev/asyncio-in-python-the-essential-guide-for-2025-a006074ee2d1)

**Decision Tracing:**
- [The AI Audit Trail: How to Ensure Compliance and Transparency with LLM Observability](https://medium.com/@kuldeep.paul08/the-ai-audit-trail-how-to-ensure-compliance-and-transparency-with-llm-observability-74fd5f1968ef)
- [The Rise of AI Audit Trails: Ensuring Traceability in Decision-Making | Aptus Data Labs](https://www.aptusdatalabs.com/thought-leadership/the-rise-of-ai-audit-trails-ensuring-traceability-in-decision-making)
- [PEP 578 – Python Runtime Audit Hooks](https://peps.python.org/pep-0578/)
- [GitHub - Amsterdam/python-audit-log](https://github.com/Amsterdam/python-audit-log)
