/**
 * LLM-TradeBot Dashboard - Frontend JavaScript
 *
 * Handles:
 * - WebSocket connection and real-time updates
 * - API data fetching
 * - UI updates and rendering
 * - Activity feed management
 */

class Dashboard {
    constructor() {
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 2000;
        this.activityLog = [];
        this.maxActivityItems = 50;

        this.init();
    }

    /**
     * Initialize dashboard
     */
    async init() {
        this.connectWebSocket();
        await this.loadInitialData();
        this.startPeriodicUpdates();
    }

    /**
     * Connect to WebSocket server
     */
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.reconnectAttempts = 0;
            this.updateConnectionStatus(true);
        };

        this.ws.onmessage = (event) => {
            this.handleWebSocketMessage(event.data);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.updateConnectionStatus(false);
            this.attemptReconnect();
        };
    }

    /**
     * Handle incoming WebSocket messages
     */
    handleWebSocketMessage(data) {
        try {
            const message = JSON.parse(data);
            const { type, data: payload } = message;

            switch (type) {
                case 'position_update':
                    this.updatePosition(payload);
                    break;
                case 'trade_executed':
                    this.handleTradeExecuted(payload);
                    break;
                case 'metrics_update':
                    this.updateMetrics(payload);
                    break;
                case 'agent_decision':
                    this.addActivity('Agent Decision', payload.decision, 'trade');
                    break;
                case 'alert':
                    this.addActivity('Alert', payload.message, 'alert');
                    break;
                case 'circuit_breaker':
                    this.addActivity('Circuit Breaker', payload.reason, 'alert');
                    break;
                default:
                    console.log('Unknown message type:', type);
            }
        } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
        }
    }

    /**
     * Attempt to reconnect WebSocket
     */
    attemptReconnect() {
        if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.error('Max reconnection attempts reached');
            return;
        }

        this.reconnectAttempts++;
        const delay = this.reconnectDelay * this.reconnectAttempts;

        console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

        setTimeout(() => {
            this.connectWebSocket();
        }, delay);
    }

    /**
     * Update connection status indicator
     */
    updateConnectionStatus(connected) {
        const indicator = document.getElementById('status-indicator');
        const statusText = document.getElementById('connection-status');

        if (connected) {
            indicator.className = 'status-indicator status-connected';
            statusText.textContent = 'Connected';
        } else {
            indicator.className = 'status-indicator status-disconnected';
            statusText.textContent = 'Disconnected';
        }
    }

    /**
     * Load initial data from API
     */
    async loadInitialData() {
        try {
            await Promise.all([
                this.loadMetrics(),
                this.loadPositions(),
                this.loadTrades()
            ]);
        } catch (error) {
            console.error('Failed to load initial data:', error);
        }
    }

    /**
     * Load performance metrics
     */
    async loadMetrics() {
        try {
            const response = await fetch('/api/metrics');
            const metrics = await response.json();
            this.updateMetrics(metrics);
        } catch (error) {
            console.error('Failed to load metrics:', error);
        }
    }

    /**
     * Load current positions
     */
    async loadPositions() {
        try {
            const response = await fetch('/api/positions');
            const positions = await response.json();
            this.renderPositions(positions);
        } catch (error) {
            console.error('Failed to load positions:', error);
        }
    }

    /**
     * Load trade history
     */
    async loadTrades() {
        try {
            const response = await fetch('/api/trades?limit=20');
            const trades = await response.json();
            this.renderTrades(trades);
        } catch (error) {
            console.error('Failed to load trades:', error);
        }
    }

    /**
     * Update metrics display
     */
    updateMetrics(metrics) {
        // Total P&L
        const pnl = metrics.total_pnl || 0;
        document.getElementById('total-pnl').textContent = this.formatCurrency(pnl);
        document.getElementById('total-pnl').className = `card-value ${pnl >= 0 ? 'positive' : 'negative'}`;

        // Win Rate
        const winRate = metrics.win_rate || 0;
        document.getElementById('win-rate').textContent = `${winRate.toFixed(1)}%`;
        document.getElementById('total-trades').textContent = `${metrics.total_trades || 0} trades`;

        // Sharpe Ratio
        const sharpe = metrics.sharpe_ratio || 0;
        document.getElementById('sharpe-ratio').textContent = sharpe.toFixed(2);
        document.getElementById('sharpe-ratio').className = `card-value ${sharpe >= 1 ? 'positive' : 'neutral'}`;

        // Max Drawdown
        const drawdown = metrics.max_drawdown || 0;
        document.getElementById('max-drawdown').textContent = `${Math.abs(drawdown).toFixed(2)}%`;
        document.getElementById('max-drawdown').className = `card-value ${drawdown > -10 ? 'neutral' : 'negative'}`;
    }

    /**
     * Render positions table
     */
    renderPositions(positions) {
        const container = document.getElementById('positions-table');

        if (positions.length === 0) {
            container.innerHTML = '<div class="empty-state"><p>No open positions</p></div>';
            return;
        }

        const html = `
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Size</th>
                        <th>Entry</th>
                        <th>Current</th>
                        <th>P&L</th>
                    </tr>
                </thead>
                <tbody>
                    ${positions.map(pos => `
                        <tr>
                            <td>${pos.symbol}</td>
                            <td><span class="badge badge-${pos.side.toLowerCase()}">${pos.side.toUpperCase()}</span></td>
                            <td>${pos.size.toFixed(4)}</td>
                            <td>${this.formatCurrency(pos.entry_price)}</td>
                            <td>${this.formatCurrency(pos.current_price)}</td>
                            <td class="${pos.pnl >= 0 ? 'positive' : 'negative'}">${this.formatCurrency(pos.pnl)}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;

        container.innerHTML = html;
    }

    /**
     * Render trades table
     */
    renderTrades(trades) {
        const container = document.getElementById('trades-table');

        if (trades.length === 0) {
            container.innerHTML = '<div class="empty-state"><p>No trades yet</p></div>';
            return;
        }

        const html = `
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Amount</th>
                        <th>Entry</th>
                        <th>Exit</th>
                        <th>P&L</th>
                    </tr>
                </thead>
                <tbody>
                    ${trades.map(trade => `
                        <tr>
                            <td>${this.formatTime(trade.closed_at || trade.opened_at)}</td>
                            <td>${trade.symbol}</td>
                            <td><span class="badge badge-${trade.side.toLowerCase()}">${trade.side.toUpperCase()}</span></td>
                            <td>${trade.amount.toFixed(4)}</td>
                            <td>${this.formatCurrency(trade.entry_price)}</td>
                            <td>${trade.exit_price ? this.formatCurrency(trade.exit_price) : '-'}</td>
                            <td class="${trade.pnl >= 0 ? 'positive' : 'negative'}">${this.formatCurrency(trade.pnl || 0)}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;

        container.innerHTML = html;
    }

    /**
     * Handle trade executed event
     */
    handleTradeExecuted(trade) {
        this.addActivity('Trade Executed', `${trade.side.toUpperCase()} ${trade.amount} ${trade.symbol} @ ${this.formatCurrency(trade.price)}`, 'trade');
        this.loadPositions();
        this.loadTrades();
        this.loadMetrics();
    }

    /**
     * Update position (real-time)
     */
    updatePosition(position) {
        this.loadPositions();
    }

    /**
     * Add activity to feed
     */
    addActivity(title, message, type = 'trade') {
        const feed = document.getElementById('activity-feed');
        const timestamp = new Date().toLocaleTimeString();

        const item = `
            <div class="activity-item ${type}">
                <div class="activity-time">${timestamp}</div>
                <div class="activity-content"><strong>${title}:</strong> ${message}</div>
            </div>
        `;

        if (feed.querySelector('.empty-state')) {
            feed.innerHTML = '';
        }

        feed.insertAdjacentHTML('afterbegin', item);

        // Keep only last N items
        this.activityLog.unshift({ title, message, type, timestamp });
        if (this.activityLog.length > this.maxActivityItems) {
            this.activityLog.pop();
            const items = feed.querySelectorAll('.activity-item');
            if (items.length > this.maxActivityItems) {
                items[items.length - 1].remove();
            }
        }
    }

    /**
     * Start periodic data updates
     */
    startPeriodicUpdates() {
        setInterval(() => {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send('ping');
            }
        }, 30000); // Ping every 30 seconds

        // Refresh data every 60 seconds
        setInterval(() => {
            this.loadMetrics();
            this.loadPositions();
        }, 60000);
    }

    /**
     * Format currency
     */
    formatCurrency(value) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }).format(value);
    }

    /**
     * Format timestamp
     */
    formatTime(timestamp) {
        return new Date(timestamp).toLocaleString();
    }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new Dashboard();
});
