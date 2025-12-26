# Codebase Concerns

**Analysis Date:** 2025-12-26

## Tech Debt

**Incomplete Agent Implementations:**
- Issue: Critical agents return placeholder data instead of real analysis
- Files:
  - `trading/agents/predict.py:18,57` - Returns neutral confidence 0.0, LightGBM model not integrated
  - `trading/agents/quant_analyst.py:18,59` - Returns hardcoded neutral signals, TA-Lib indicators not implemented
  - `trading/agents/bull.py:47` - Simplified 2% momentum detection, no technical indicators
  - `trading/agents/bear.py:47` - Mirrors Bull agent with same simplistic logic
- Why: MVP phase implementation, full features deferred
- Impact: 8-agent system makes decisions without proper analysis - high risk of bad trades
- Fix approach: Implement TA-Lib indicators in QuantAnalyst, integrate LightGBM in Predict agent

**Duplicate Bull/Bear Logic:**
- Issue: Nearly identical trend detection logic in both agents
- Files:
  - `trading/agents/bull.py:48-74` - Price trend detection
  - `trading/agents/bear.py:48-74` - Same logic with inverted conditions
- Why: Quick implementation without abstraction
- Impact: Changes must be made twice, increases bug risk
- Fix approach: Extract shared `calculate_momentum()` utility function

**Broad Exception Handling:**
- Issue: Generic `except Exception as e:` used throughout (50+ occurrences)
- Files: `trading/cli.py:62,108,126,156,177,217,268`, `trading/manager.py:91,235,271,297`, most other files
- Why: Lack of specific exception types, defensive programming
- Impact: Masks specific errors, makes debugging harder, prevents proper recovery
- Fix approach: Replace with specific exception types (ValueError, KeyError, APIError, etc.)

**CLI Command Duplication:**
- Issue: Identical try-except-json_output pattern repeated 5+ times
- Files: `trading/cli.py:27-63` (status), `66-109` (positions), `112-127` (run), `160-178` (cancel), `181-220` (close)
- Why: Copy-paste coding
- Impact: Maintenance burden when error handling needs updates
- Fix approach: Extract decorator or wrapper function for command execution

## Known Bugs

**No Atomic State Writes:**
- Symptoms: State file could be corrupted on crash
- Trigger: Process crashes during `state.py:61-62` write operation
- Files: `trading/state.py:59-62` (save method)
- Workaround: None (data loss possible)
- Root cause: Direct JSON dump without temp file + rename pattern
- Fix: Use temp file write + atomic rename pattern

**Missing Position Validation:**
- Symptoms: Could create orders larger than max position size
- Trigger: High confidence leads to large position
- Files: `trading/agents/execution.py:88-97` - No check if position exceeds max before creating order
- Workaround: Manual monitoring
- Root cause: Validation happens in RiskAudit but after position sizing
- Fix: Add pre-flight check in Execution agent before creating orders

**Unhandled Async Timeouts:**
- Symptoms: System hangs on network issues
- Trigger: Exchange API doesn't respond
- Files: `trading/agents/data_sync.py:59` - `asyncio.gather()` with no timeout
- Workaround: Manual restart
- Root cause: No timeout handling on async operations
- Fix: Add `asyncio.wait_for()` with reasonable timeouts (5-30s)

## Security Considerations

**Missing .gitignore:**
- Risk: API keys in `.env` could be committed to git
- Files: Project root - no `.gitignore` file exists
- Current mitigation: None (relying on manual vigilance)
- Recommendations: Create `.gitignore` with `.env`, `*.pem`, `*.key`, `__pycache__`, `.venv/`, `logs/`, `data/`

**Plain Text API Keys in Memory:**
- Risk: API keys stored as plain strings, exposed in logs/errors
- Files:
  - `trading/config.py:22-23` - API credentials as dataclass fields
  - `trading/config.py:51-52` - Keys passed to ccxt exchange constructor
- Current mitigation: `__repr__` masking (insufficient)
- Recommendations: Use credential management library (keyring, AWS Secrets Manager, etc.)

**Insufficient Secret Masking:**
- Risk: API secrets partially visible in repr
- Files: `trading/config.py:199` - Only masks first 4 chars of secret
- Current mitigation: Partial masking
- Recommendations: Full masking or complete exclusion from repr

## Performance Bottlenecks

**No Caching of Market Data:**
- Problem: Fetches same data multiple times in single loop
- Files: `trading/agents/data_sync.py` - No caching mechanism
- Measurement: Not profiled
- Cause: Each agent refetches data independently
- Improvement path: Add context-level caching for single loop iteration

**Potential N+1 Pattern:**
- Problem: Multiple sequential API calls without batching
- Files: `trading/agents/data_sync.py:48-55` - Uses asyncio.gather but no error recovery
- Measurement: Not profiled
- Cause: No request batching or caching
- Improvement path: Implement request deduplication and caching layer

**No Pagination on Trade History:**
- Problem: Loads all trades into memory
- Files: `trading/cli.py:135` - Glob search without limit
- Measurement: Unknown scaling limit
- Cause: File-based storage without chunking
- Improvement path: Add pagination to history queries

## Fragile Areas

**State Persistence:**
- Why fragile: JSON write not atomic, no schema versioning
- Common failures: Corrupted state on crash, no migration path for schema changes
- Files: `trading/state.py:59-62`, `85-92`
- Safe modification: Add temp file + rename, version schema
- Test coverage: No tests for state management

**Risk Audit Veto Logic:**
- Why fragile: Complex business rules with no validation
- Common failures: Unknown (untested)
- Files: `trading/agents/risk_audit.py`
- Safe modification: Add comprehensive tests before changing
- Test coverage: 0 tests

**Agent Context Passing:**
- Why fragile: Dictionary mutation across agents, no schema
- Common failures: KeyError if expected keys missing
- Files: All agents in `trading/agents/`
- Safe modification: Define TypedDict for context schema
- Test coverage: No integration tests for agent pipeline

## Scaling Limits

**File-Based State:**
- Current capacity: Unknown (not profiled with large trade counts)
- Limit: File system I/O becomes bottleneck
- Symptoms at limit: Slow state saves, high disk I/O
- Scaling path: Migrate to SQLite or PostgreSQL

**Synchronous State Writes:**
- Current capacity: Not measured
- Limit: Blocks trading loop on every state update
- Symptoms at limit: Increased latency per trade
- Scaling path: Async state writes with write-behind cache

## Dependencies at Risk

**TA-Lib Installation:**
- Risk: Requires system-level dependencies, hard to install
- Impact: Users can't run technical indicators
- Migration plan: Consider pure Python alternative (pandas-ta)

**Graphiti Core:**
- Risk: Optional dependency, unclear maintenance status
- Impact: Memory feature unavailable if broken
- Migration plan: Already optional, low risk

## Missing Critical Features

**No Order Cancellation on Failure:**
- Problem: Failed orders not automatically cancelled
- Current workaround: Manual cancellation via CLI
- Blocks: Automated recovery from errors
- Implementation complexity: Low (add to error handlers)

**No Rate Limit Handling:**
- Problem: Could hit exchange rate limits
- Current workaround: CCXT enableRateLimit (may be insufficient)
- Blocks: High-frequency trading
- Implementation complexity: Medium (backoff + retry logic)

**No Transaction History Export:**
- Problem: Can't export trades for tax reporting
- Current workaround: Manual file inspection
- Blocks: Compliance, accounting
- Implementation complexity: Low (add CSV export)

## Test Coverage Gaps

**Agent Logic:**
- What's not tested: All 8 agents have 0 tests
- Risk: Critical bugs in trading decisions go undetected
- Priority: **CRITICAL**
- Difficulty to test: Medium (requires market data fixtures)

**Risk Management:**
- What's not tested: Circuit breakers, position limits, veto logic
- Risk: Safety mechanisms could fail silently
- Priority: **CRITICAL**
- Difficulty to test: Low (unit testable)

**State Persistence:**
- What's not tested: Save/load, recovery, corruption handling
- Risk: Data loss on crash
- Priority: **HIGH**
- Difficulty to test: Low (file system mocking)

**ML Models:**
- What's not tested: Model training, predictions, ensemble
- Risk: Incorrect predictions go unnoticed
- Priority: **MEDIUM**
- Difficulty to test: Medium (requires training data fixtures)

**Integration Testing:**
- What's not tested: Full agent pipeline, end-to-end flows
- Risk: Agent interactions break
- Priority: **HIGH**
- Difficulty to test: High (complex fixtures, async orchestration)

---

*Concerns audit: 2025-12-26*
*Update as issues are fixed or new ones discovered*
