#!/bin/bash
# LLM-TradeBot Setup Script

set -e

echo "🤖 LLM-TradeBot Setup"
echo "===================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version found"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
echo "✓ Dependencies installed"
echo ""

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp config/.env.example .env
    echo "✓ .env file created"
    echo ""
    echo "⚠️  IMPORTANT: Edit .env file to configure your trading provider"
else
    echo "✓ .env file already exists"
fi
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p logs data/trading
echo "✓ Directories created"
echo ""

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file: nano .env"
echo "2. Configure TRADING_PROVIDER (start with 'paper' for safety)"
echo "3. Run: python -m trading.cli status"
echo ""
echo "For paper trading (no API keys needed):"
echo "  python -m trading.cli run --symbol BTC/USDT"
echo ""
echo "Happy trading! 📈"
