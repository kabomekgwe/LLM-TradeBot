# Phase 5: Enhanced Feature Engineering - Research

**Researched:** 2025-12-27
**Domain:** Financial feature engineering (TA-Lib indicators, sentiment APIs, market microstructure, time-series features)
**Confidence:** HIGH

<research_summary>
## Summary

Researched the Python ecosystem for building comprehensive feature engineering pipelines for algorithmic trading systems. The standard approach combines TA-Lib (150+ technical indicators) or pandas-ta (150+ indicators with better Python integration) for technical analysis, CCXT for market microstructure data (order book depth, bid-ask spread), sentiment APIs (Alternative.me, News APIs, Twitter), and pandas_market_calendars for time-based features.

Key finding: Don't hand-roll technical indicators, sentiment aggregation, or volatility regime detection. TA-Lib/pandas-ta provide battle-tested implementations avoiding calculation errors. Use existing libraries for market calendar integration and regime detection (HMM, GARCH) to prevent look-ahead bias and data leakage.

**Primary recommendation:** Use pandas-ta (easier installation than TA-Lib) + CCXT (order book/spread) + Alternative.me (fear/greed) + pandas_market_calendars (sessions/holidays) + Hidden Markov Models (volatility regimes). Focus on preventing look-ahead bias through strict timestamp-based feature construction.
</research_summary>

<standard_stack>
## Standard Stack

The established libraries/tools for financial feature engineering:

### Core Technical Indicators
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| pandas-ta | latest (0.3.x) | 150+ technical indicators | Pure Python, easier installation than TA-Lib, pandas integration, actively maintained |
| TA-Lib | 0.6.8 | 150+ technical indicators (alternative) | Industry standard, C implementation (faster), 2-4x faster than SWIG, established patterns |
| talib | latest | Alternative TA implementation | Lower-level, requires C libraries, harder setup |

### Market Data & Microstructure
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| ccxt | 4.x | Order book depth, bid-ask spread, OHLCV | Already in project, unified exchange API |
| crobat | latest | Order flow imbalance (academic) | Cryptocurrency order book analysis, research applications |

### Sentiment Analysis
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| tweepy | 4.x | Twitter API v2 access | Social media sentiment (requires Twitter API approval) |
| fear-and-greed-crypto | latest | Alternative.me Fear & Greed Index | Crypto sentiment (free, no API key required) |
| textblob | latest | Basic sentiment analysis | Quick sentiment scoring for text data |
| transformers | latest | Advanced NLP models | Pre-trained financial sentiment models |

### Time-Based Features
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pandas_market_calendars | 4.x | 50+ exchange calendars, holidays, sessions | Trading session detection, holiday handling |
| exchange_calendars | latest | Alternative calendar library | 50+ global exchanges, break support (Asian lunch breaks) |

### Volatility Regime Detection
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| statsmodels | 0.14.x | Markov regime-switching models | Volatility regime detection (low/high volatility states) |
| hmmlearn | latest | Hidden Markov Models | Market regime classification |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| TA-Lib | pandas-ta | pandas-ta easier to install (pure Python), TA-Lib faster (C implementation) |
| pandas_market_calendars | exchange_calendars | exchange_calendars more actively maintained, pandas_market_calendars simpler API |
| Twitter API | Reddit/StockTwits APIs | Reddit easier access, Twitter requires API approval |
| textblob | transformers | transformers more accurate but slower, textblob fast but simple |

**Installation:**
```bash
# Core stack (recommended)
pip install pandas-ta ccxt fear-and-greed-crypto pandas_market_calendars statsmodels

# Alternative with TA-Lib (requires C libraries)
# brew install ta-lib  # macOS
pip install TA-Lib

# Sentiment (optional - requires API keys)
pip install tweepy textblob transformers

# Regime detection
pip install hmmlearn
```
</standard_stack>

<architecture_patterns>
## Architecture Patterns

### Recommended Project Structure
```
trading/
├── features/
│   ├── technical.py          # TA-Lib/pandas-ta indicators
│   ├── sentiment.py           # Twitter, news, fear/greed aggregation
│   ├── microstructure.py      # Order book, bid-ask spread, trade flow
│   ├── temporal.py            # Time-based features, sessions, holidays
│   └── regime.py              # Volatility regime detection
├── ml/
│   ├── feature_extraction.py  # Orchestrates all feature modules
│   └── preprocessing.py       # Scaling, missing data handling
└── utils/
    └── time_utils.py          # Timestamp alignment, look-ahead prevention
```

### Pattern 1: Feature Engineering with pandas-ta
**What:** Use pandas-ta for technical indicators with pandas DataFrames
**When to use:** Need comprehensive technical indicators without TA-Lib C library setup
**Example:**
```python
# Source: pandas-ta documentation
import pandas_ta as ta

# Calculate indicators on OHLCV DataFrame
df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# RSI (already have, included for completeness)
df['rsi'] = ta.rsi(df['close'], length=14)

# ATR (volatility)
df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)

# Stochastic Oscillator
stoch = ta.stoch(df['high'], df['low'], df['close'])
df['stoch_k'] = stoch['STOCHk_14_3_3']
df['stoch_d'] = stoch['STOCHd_14_3_3']

# Aroon (trend strength)
aroon = ta.aroon(df['high'], df['low'])
df['aroon_up'] = aroon['AROONU_14']
df['aroon_down'] = aroon['AROOND_14']

# Money Flow Index (volume-weighted momentum)
df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
```

### Pattern 2: Market Microstructure Features from CCXT
**What:** Extract order book depth, bid-ask spread, and trade imbalance
**When to use:** Need level 2 market data for ML features
**Example:**
```python
# Source: CCXT examples and community patterns
import ccxt

exchange = ccxt.binance({'enableRateLimit': True})

# Fetch order book with depth
order_book = exchange.fetch_order_book('BTC/USDT', limit=20)  # Top 20 levels

# Calculate bid-ask spread
best_bid = order_book['bids'][0][0]  # Price of best bid
best_ask = order_book['asks'][0][0]  # Price of best ask
spread = best_ask - best_bid
spread_pct = (spread / best_ask) * 100

# Order book imbalance (bid volume vs ask volume)
bid_volume = sum([level[1] for level in order_book['bids'][:10]])  # Top 10 levels
ask_volume = sum([level[1] for level in order_book['asks'][:10]])
imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)

# Mid-price
mid_price = (best_bid + best_ask) / 2

features = {
    'bid_ask_spread': spread,
    'spread_pct': spread_pct,
    'order_book_imbalance': imbalance,
    'mid_price': mid_price,
}
```

### Pattern 3: Sentiment Aggregation (Fear & Greed Index)
**What:** Fetch crypto market sentiment from Alternative.me API
**When to use:** Need sentiment indicator without API approval delays
**Example:**
```python
# Source: fear-and-greed-crypto PyPI package
from fear_and_greed import get_latest, get_historical

# Get current fear & greed index (0-100)
latest = get_latest()
fgi_value = latest['value']  # 0 = Extreme Fear, 100 = Extreme Greed
fgi_class = latest['value_classification']  # 'Fear', 'Greed', 'Neutral', etc.

# Get historical data (last 30 days)
historical = get_historical(limit=30)
fgi_series = pd.DataFrame(historical)
fgi_series['timestamp'] = pd.to_datetime(fgi_series['timestamp'], unit='s')
```

### Pattern 4: Time-Based Features with pandas_market_calendars
**What:** Detect trading sessions, holidays, and time-of-day features
**When to use:** Need to account for market hours, session effects, holidays
**Example:**
```python
# Source: pandas_market_calendars documentation
import pandas_market_calendars as mcal

# Get NYSE calendar
nyse = mcal.get_calendar('NYSE')

# Get valid trading days
schedule = nyse.schedule(start_date='2024-01-01', end_date='2025-01-01')

# Check if a date is a trading day
is_trading_day = date in schedule.index

# Detect trading session (Asian, London, New York)
def get_session(hour_utc):
    """Map UTC hour to trading session"""
    if 0 <= hour_utc < 8:
        return 'Asia'  # Tokyo open
    elif 8 <= hour_utc < 16:
        return 'London'  # London open
    else:
        return 'New_York'  # NY open

# Create time-based features
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6])
df['session'] = df['hour'].apply(get_session)
df['is_holiday'] = ~df['timestamp'].dt.date.isin(schedule.index)
```

### Pattern 5: Volatility Regime Detection with HMM
**What:** Detect market volatility regimes (low/high volatility) using Hidden Markov Models
**When to use:** Adapt strategy to market conditions
**Example:**
```python
# Source: statsmodels and community patterns
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

# Calculate returns
returns = df['close'].pct_change().dropna()

# Fit 2-regime model (low volatility, high volatility)
model = MarkovRegression(
    returns,
    k_regimes=2,  # 2 regimes
    switching_variance=True,  # Variance switches between regimes
)
results = model.fit()

# Get regime probabilities for each timestamp
df['regime_prob_0'] = results.smoothed_marginal_probabilities[0]  # Low volatility
df['regime_prob_1'] = results.smoothed_marginal_probabilities[1]  # High volatility
df['current_regime'] = results.smoothed_marginal_probabilities.idxmax(axis=1)
```

### Anti-Patterns to Avoid
- **Hand-rolling technical indicators:** TA-Lib/pandas-ta have edge cases handled (e.g., ATR unstable period)
- **Using future data in features:** Strict timestamp alignment required to prevent look-ahead bias
- **Ignoring missing data:** Forward-fill price data causes issues, handle explicitly
- **Not scaling features:** Different indicator ranges (RSI 0-100, MACD unbounded) hurt ML models
- **Using raw sentiment text:** Aggregate and normalize sentiment scores into numeric features
</architecture_patterns>

<dont_hand_roll>
## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Technical indicators (RSI, ATR, etc.) | Custom indicator calculations | pandas-ta or TA-Lib | Edge cases (unstable periods, NaN handling, smoothing), calculation errors, performance |
| Volatility regime detection | Manual threshold rules | HMM (hmmlearn) or GARCH models | Statistical rigor, transition probabilities, regime persistence modeling |
| Trading calendar | Manual holiday lists | pandas_market_calendars | 50+ exchanges, updated holidays, session breaks, regulatory changes |
| Sentiment scoring | Regex/keyword matching | TextBlob or Transformers | Negation handling, context, sarcasm, pre-trained on financial text |
| Order flow imbalance | Manual bid/ask volume sums | CCXT + proven formulas | Weighted depth, time decay, normalization strategies |
| Feature scaling | Manual min-max | sklearn StandardScaler/RobustScaler | Outlier handling, inverse transforms, fit/transform pattern |
| Time-series split | Random train/test | sklearn TimeSeriesSplit | Prevents future data leakage, expanding/sliding windows |

**Key insight:** Financial feature engineering has 30+ years of established research. TA-Lib implements proper indicator calculations with unstable periods handled. pandas_market_calendars tracks regulatory calendar changes. Transformers provide pre-trained models on financial text. Custom implementations miss edge cases that cause silent bugs in production (look-ahead bias, calculation errors, missing data handling).
</dont_hand_roll>

<common_pitfalls>
## Common Pitfalls

### Pitfall 1: Look-Ahead Bias in Feature Construction
**What goes wrong:** Features use future data that wouldn't be available at prediction time
**Why it happens:** Pandas operations like `.shift()` are confusing, or using non-causal calculations
**How to avoid:**
- Use strict timestamp-based joins (merge_asof with direction='backward')
- All indicators based on past data only (RSI uses last 14 candles, not next 14)
- Validate with manual spot-checks: "Would I have this data at time T?"
**Warning signs:** Unrealistically high backtest performance, model fails in live trading

### Pitfall 2: Data Leakage from Scaling
**What goes wrong:** Fitting scaler on entire dataset including test data
**Why it happens:** Using `fit_transform()` on full dataset before train/test split
**How to avoid:**
- Fit scaler ONLY on training data
- Transform both train and test using training scaler
- Use TimeSeriesSplit for proper cross-validation
**Warning signs:** Model performs well in backtest but fails forward, sudden performance drop

### Pitfall 3: Indicator Unstable Periods
**What goes wrong:** First N candles have invalid indicator values (NaN or incorrect)
**Why it happens:** RSI needs 14 candles, MACD needs 26, ATR has warm-up period
**How to avoid:**
- Check TA-Lib function documentation for unstable period
- Drop first N rows after indicator calculation
- Use `df.dropna()` carefully (can hide the issue)
**Warning signs:** Model performs better on older data, strange patterns in first N predictions

### Pitfall 4: Survivorship Bias in Sentiment Data
**What goes wrong:** Only analyzing sentiment for currently traded assets, missing delisted/failed projects
**Why it happens:** APIs only return data for active symbols
**How to avoid:**
- Track historical symbol lists including delisted assets
- Document when assets were added/removed
- Use point-in-time asset lists for backtests
**Warning signs:** Backtest shows no losses from failed projects, unrealistic success rate

### Pitfall 5: Session/Timezone Misalignment
**What goes wrong:** Features calculated in wrong timezone, missing session transitions
**Why it happens:** Mixing UTC, exchange local time, and system time
**How to avoid:**
- Standardize on UTC for all timestamps
- Use pandas_market_calendars for session detection
- Explicitly convert timezones when needed
**Warning signs:** Features show strange patterns around midnight, session detection fails
</common_pitfalls>

<code_examples>
## Code Examples

Verified patterns from official sources:

### Preventing Look-Ahead Bias in Feature Joins
```python
# Source: Best practices from financial ML literature
import pandas as pd

# WRONG - this can cause look-ahead bias
df_features = df_prices.merge(df_indicators, on='timestamp')

# CORRECT - merge_asof ensures we only use past data
df_features = pd.merge_asof(
    df_prices.sort_values('timestamp'),
    df_indicators.sort_values('timestamp'),
    on='timestamp',
    direction='backward'  # Only use indicators from past or present
)
```

### Proper Train/Test Split for Time-Series
```python
# Source: sklearn TimeSeriesSplit documentation
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_index, test_index in tscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Fit scaler ONLY on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # Use training scaler

    # Train model
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
```

### Complete Feature Engineering Pipeline
```python
# Source: Combining best practices from all sources
import pandas as pd
import pandas_ta as ta
import pandas_market_calendars as mcal
from fear_and_greed import get_historical

class FeatureEngineer:
    def __init__(self, exchange_ccxt, calendar='NYSE'):
        self.exchange = exchange_ccxt
        self.calendar = mcal.get_calendar(calendar)

    async def extract_features(self, symbol, start_date, end_date):
        """Extract all feature categories"""

        # 1. Get OHLCV data
        ohlcv = await self.exchange.fetch_ohlcv(symbol, '5m')
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # 2. Technical indicators (pandas-ta)
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)

        # 3. Market microstructure
        order_book = await self.exchange.fetch_order_book(symbol, limit=20)
        df['spread'] = order_book['asks'][0][0] - order_book['bids'][0][0]
        df['mid_price'] = (order_book['bids'][0][0] + order_book['asks'][0][0]) / 2

        # 4. Sentiment (fear & greed)
        fgi_data = get_historical(limit=30)
        fgi_df = pd.DataFrame(fgi_data)
        fgi_df['timestamp'] = pd.to_datetime(fgi_df['timestamp'], unit='s')
        df = pd.merge_asof(df, fgi_df[['timestamp', 'value']], on='timestamp', direction='backward')
        df.rename(columns={'value': 'fear_greed_index'}, inplace=True)

        # 5. Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])

        # 6. Remove unstable periods and NaN
        df = df.dropna()  # Drop rows with missing indicators

        return df
```
</code_examples>

<sota_updates>
## State of the Art (2024-2025)

What's changed recently:

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| TA-Lib only | pandas-ta | 2020+ | Pure Python, easier installation, actively maintained, pandas native |
| Twitter API v1.1 | Twitter API v2 | 2021 | Requires elevated access, Tweepy 4.x required, better rate limits |
| quantopian/trading_calendars | pandas_market_calendars | 2019+ | Quantopian shut down, pandas_market_calendars is maintained fork |
| Manual regime detection | HMM/GARCH models | 2015+ | Statistical rigor, statsmodels support, automated regime classification |
| Alternative.me only | Multiple fear/greed sources | 2023+ | CFGI.io for crypto, CNN for stocks, diversified sentiment |
| TextBlob | Transformers (FinBERT) | 2023+ | Pre-trained on financial text, better accuracy, context understanding |

**New tools/patterns to consider:**
- **pandas-ta experimental streaming API:** Compute latest indicator value incrementally (useful for live trading)
- **CFGI.io:** Real-time fear & greed for 50+ crypto tokens (updates every 15 minutes)
- **FinBERT (Transformers):** Pre-trained BERT model on financial news sentiment
- **exchange_calendars:** More actively maintained than pandas_market_calendars, supports exchange breaks

**Deprecated/outdated:**
- **quantopian/trading_calendars:** Company shut down, use pandas_market_calendars instead
- **Twitter API v1.1:** Deprecated, use v2 with Tweepy 4.x
- **VADER for financial text:** Generic social media model, use FinBERT for finance-specific sentiment
- **Manual holiday lists:** Use calendar libraries to avoid missing regulatory changes
</sota_updates>

<open_questions>
## Open Questions

Things that couldn't be fully resolved:

1. **Twitter API v2 Elevated Access**
   - What we know: Requires application approval, can take days/weeks, rate limits vary by tier
   - What's unclear: Approval criteria, timeline, rejection likelihood for trading bots
   - Recommendation: Start with fear/greed index (no API key), add Twitter later if approved

2. **pandas-ta vs TA-Lib Performance at Scale**
   - What we know: TA-Lib faster (C implementation), pandas-ta easier (pure Python)
   - What's unclear: Performance difference on 1M+ candles, memory usage comparison
   - Recommendation: Start with pandas-ta, benchmark if performance issues arise, switch to TA-Lib if needed

3. **Order Book Depth Optimal Levels**
   - What we know: CCXT supports 20+ levels, more depth = more context
   - What's unclear: Diminishing returns beyond N levels, exchange-specific depth limits
   - Recommendation: Start with 10 levels (captures most liquidity), experiment with 20 if needed

4. **Sentiment Signal Lag**
   - What we know: Fear/greed index updates daily, Twitter real-time but noisy
   - What's unclear: Optimal aggregation window for sentiment (1h, 4h, 24h)
   - Recommendation: Test multiple windows (1h, 4h, 24h), use walk-forward validation to find optimal
</open_questions>

<sources>
## Sources

### Primary (HIGH confidence)
- [TA-Lib Python Official Documentation](https://ta-lib.github.io/ta-lib-python/) - Indicator list, function groups
- [TA-Lib Python Functions List](https://ta-lib.github.io/ta-lib-python/funcs.html) - All 150+ indicators documented
- [pandas-ta PyPI](https://pypi.org/project/pandas-ta/) - Installation, indicator list
- [pandas-ta Official Documentation](https://www.pandas-ta.dev/) - Usage patterns, examples
- [CCXT Documentation](https://docs.ccxt.com/) - Unified exchange API
- [pandas_market_calendars PyPI](https://pypi.org/project/pandas_market_calendars/) - Exchange calendars
- [fear-and-greed-crypto PyPI](https://pypi.org/project/fear-and-greed-crypto/) - Alternative.me API wrapper

### Secondary (MEDIUM confidence)
- [pandas-ta vs TA-Lib Comparison](https://www.slingacademy.com/article/comparing-ta-lib-to-pandas-ta-which-one-to-choose/) - Verified installation differences, performance comparison
- [CCXT Order Book Tutorial](https://www.slingacademy.com/article/fetching-market-data-with-ccxt-tickers-order-books-and-ohlcv/) - Verified with CCXT examples
- [Market Regime Detection with HMM](https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/) - Verified with statsmodels documentation
- [Look-Ahead Bias Prevention](https://www.marketcalls.in/machine-learning/understanding-look-ahead-bias-and-how-to-avoid-it-in-trading-strategies.html) - Verified with financial ML best practices

### Tertiary (LOW confidence - needs validation during implementation)
- Community sentiment aggregation patterns - Test multiple approaches
- Optimal sentiment window sizing - Requires backtesting to validate
- Order book depth levels - Exchange-specific, needs empirical testing
</sources>

<metadata>
## Metadata

**Research scope:**
- Core technology: pandas-ta (technical indicators), CCXT (market microstructure)
- Ecosystem: Sentiment APIs (Alternative.me, Twitter v2, News APIs), calendar libraries, regime detection (HMM, GARCH)
- Patterns: Feature engineering pipelines, look-ahead bias prevention, time-series split, scaling
- Pitfalls: Look-ahead bias, data leakage, unstable periods, survivorship bias, timezone issues

**Confidence breakdown:**
- Standard stack: HIGH - pandas-ta, CCXT, pandas_market_calendars verified from official sources
- Architecture: HIGH - Patterns verified from TA-Lib documentation, CCXT examples, statsmodels docs
- Pitfalls: HIGH - Look-ahead bias and data leakage well-documented in financial ML literature
- Code examples: MEDIUM - Verified structure but need integration testing with existing LLM-TradeBot architecture

**Research date:** 2025-12-27
**Valid until:** 2026-01-27 (30 days - indicator libraries stable, sentiment APIs change infrequently)

**Next steps:** This research informs Phase 5 planning. When planning:
1. Choose pandas-ta over TA-Lib (easier installation unless performance issues)
2. Start with Alternative.me fear/greed (no API key required)
3. Use pandas_market_calendars for session detection
4. Implement strict look-ahead bias prevention in feature pipeline
5. Focus on preventing data leakage through proper train/test splitting
</metadata>

---

*Phase: 05-enhanced-feature-engineering*
*Research completed: 2025-12-27*
*Ready for planning: yes*
