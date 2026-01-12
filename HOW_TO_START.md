# 🚀 Chartor Trading Bot - Quick Start Guide

Complete guide to starting and running your institutional-grade crypto trading bot on WEEX exchange.

---

## 📋 Prerequisites

Before starting, ensure you have:

- ✅ **Ubuntu/Linux Server** (or Windows with Python 3.10+)
- ✅ **WEEX Account** with API credentials
- ✅ **Minimum Capital**: $500+ USDT (recommended $1000+)
- ✅ **Python 3.10+** installed
- ✅ **Node.js 18+** (for dashboard)

---

## 🔧 Installation

### 1. Clone and Setup Environment

```bash
# Navigate to project
cd ~/Chartor-Market

# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Credentials

Create `.env` file in project root:

```bash
nano .env
```

Add your WEEX API credentials:

```env
# WEEX Exchange API
WEEX_API_KEY=your_api_key_here
WEEX_SECRET_KEY=your_secret_key_here
WEEX_PASSPHRASE=your_passphrase_here

# Optional: AI Services
GEMINI_API_KEY=your_gemini_key_here
CRYPTOPANIC_API_KEY=your_cryptopanic_key_here
```

**Important:** Keep your `.env` file secure - never share or commit it!

### 3. Get WEEX API Credentials

1. Login to **https://www.weex.com/**
2. Go to **Account** → **API Management**
3. Create new API key with permissions:
   - ✅ **Read** - View account data
   - ✅ **Trade** - Place and cancel orders
   - ✅ **Futures** - Access futures trading
4. Save API Key, Secret Key, and Passphrase to `.env`
5. Whitelist your server IP (recommended)

---

## 🎯 Starting the Bot

### Option 1: Institutional Trading Mode (Recommended)

Runs advanced multi-asset momentum strategy with AI validation.

```bash
# Activate environment
source .venv/bin/activate

# Start institutional trading
python run_institutional_trading.py
```

**Features:**

- Scans 8 crypto pairs every 30 seconds
- AI-powered trade validation (Gemini + ML + Sentiment)
- Automatic position management
- Risk limits: 1.25% per trade, 40% max exposure
- Leverage: 20x

**Confirmation Required:**

```
⚠️ Starting LIVE INSTITUTIONAL TRADING
Account Balance: $XXX.XX USDT
Risk per trade: 1.25% ($XX.XX)
Max exposure: 40% ($XXX.XX)

Type 'START' to begin live trading:
```

Type `START` and press Enter.

### Option 2: Sentinel Mode (Simple)

Runs single-asset AI-powered trading with manual control.

```bash
python sentinel_service.py
```

**Features:**

- Monitors one symbol at a time
- AI analysis every 30 seconds
- Manual trade approval (unless auto-trading enabled)
- Lower complexity

---

## 📊 Starting the Dashboard

In a **separate terminal** (keep bot running):

```bash
cd kairos-trading-hub-main

# Install dependencies (first time only)
npm install

# Start dashboard
npm run dev
```

Access dashboard at: **http://localhost:5173**

**Dashboard Features:**

- Real-time position tracking
- Live PnL monitoring
- Market data watchlist
- AI analysis logs
- Trade history
- Risk metrics

---

## 🔍 Monitoring the Bot

### Check Bot Status

```bash
# View last 50 lines of log
tail -50 api_server_*.log

# Follow live logs
tail -f api_server_*.log
```

### Check Positions

```bash
# Check current positions
python emergency_position_check.py
```

### View Balance

```bash
# Quick balance check
python -c "
from core.weex_api import WeexClient
client = WeexClient()
balance = client.get_balance()
for item in balance:
    if item.get('coinName') == 'USDT':
        print(f\"Available: \${item.get('available')}\")
        print(f\"Equity: \${item.get('equity')}\")
        print(f\"Unrealized PnL: \${item.get('unrealizePnl')}\")
"
```

---

## 🛑 Stopping the Bot

### Graceful Shutdown

In the terminal running the bot, press:

```
Ctrl + C
```

The bot will:

1. Stop scanning for new trades
2. Keep monitoring open positions
3. Execute stop losses if hit
4. Save state to database

**⚠️ IMPORTANT:** Open positions remain on WEEX after shutdown!

### Emergency Position Close

If bot crashes or you need to close everything:

```bash
# Close all positions immediately
python close_all_positions_api.py
```

Or manually via WEEX website:

1. Go to **https://www.weex.com/**
2. **Futures** → **Positions**
3. Click **"Close All"**

---

## 🚨 Emergency Procedures

### Bot Not Starting

```bash
# Check Python version (need 3.10+)
python --version

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check .env file exists
ls -la .env
```

### Positions Losing Money

**Stop losses not working?**

```bash
# Check if Position Manager is running
curl http://localhost:8000/api/position-monitor-status

# Close positions manually
python close_all_positions_api.py
```

### API Errors (521, 400, etc.)

**WEEX API down?**

- Wait 5-10 minutes (temporary outage)
- Close positions via WEEX website
- Check WEEX status: https://www.weex.com/

### Out of Margin

Error: `"The order margin available amount not enough"`

**Solution:**

- Close losing positions first
- Reduce leverage (currently 20x)
- Add more USDT to account

---

## ⚙️ Configuration

### Adjust Risk Settings

Edit `risk/risk_manager.py`:

```python
RISK_PER_TRADE = 0.0125  # 1.25% per trade
MAX_EXPOSURE = 0.40      # 40% max portfolio exposure
MAX_DAILY_LOSS = 0.03    # 3% max daily loss
MAX_LEVERAGE = 20        # 20x leverage
```

### Change Trading Symbols

Edit `trading_orchestrator.py`:

```python
ENABLED_SYMBOLS = [
    "cmt_btcusdt", "cmt_ethusdt", "cmt_solusdt",
    "cmt_dogeusdt", "cmt_xrpusdt", "cmt_adausdt",
    "cmt_bnbusdt", "cmt_ltcusdt"
]
```

### Adjust Scan Interval

Edit `trading_orchestrator.py`:

```python
CYCLE_INTERVAL = 30  # Scan every 30 seconds
```

---

## 📈 Understanding Your First Trade

**Example Trade Execution:**

```
2026-01-12 14:40:40 - InstitutionalTrading - INFO
✅ Best opportunity: cmt_adausdt
   Signal: SHORT | Strength: 76.2 | Regime: NEUTRAL

🎯 Opening SHORT position on cmt_adausdt
   Entry: $0.3845
   Stop Loss: $0.39 (3.9% risk)
   Take Profit: $0.38 (3.9% gain)
   Size: 1060 contracts
   Margin: $20.35 (7% of capital)
   Risk:Reward = 1:2
```

**What this means:**

- Opened SHORT position (betting price goes DOWN)
- Entry price: $0.3845
- If price hits $0.39 → Stop loss triggered (lose ~$8.25)
- If price hits $0.38 → Take profit triggered (gain ~$16.50)
- Using $20.35 margin (7% of your capital)
- Position will auto-close at stop/take profit

---

## 📚 Additional Resources

### Log Files

- `api_server_YYYYMMDD_HHMMSS.log` - Main bot logs
- Position data stored in SQLite: `trading.db`

### Key Scripts

- `run_institutional_trading.py` - Start institutional bot
- `sentinel_service.py` - Start sentinel bot
- `emergency_position_check.py` - Check open positions
- `close_all_positions_api.py` - Emergency close all

### Dashboard

- Frontend: `kairos-trading-hub-main/src/`
- API: `api_server.py` (runs on port 8000)

---

## ✅ Pre-Flight Checklist

Before starting live trading:

- [ ] `.env` file configured with valid API keys
- [ ] Tested API connection: `python -c "from core.weex_api import WeexClient; print(WeexClient().get_balance())"`
- [ ] Minimum $500 USDT in account
- [ ] Understand SHORT vs LONG positions
- [ ] Know how to emergency close positions
- [ ] Dashboard accessible at http://localhost:5173
- [ ] Risk settings reviewed and acceptable
- [ ] Ready to monitor positions actively

---

## 🎓 Trading Concepts

### SHORT Position

- **Profit when price goes DOWN**
- Example: Sell BTC at $100k, buy back at $95k = +$5k profit

### LONG Position

- **Profit when price goes UP**
- Example: Buy BTC at $95k, sell at $100k = +$5k profit

### Leverage (20x)

- Control $2000 worth of crypto with $100 margin
- **Increases both gains AND losses**
- Use carefully!

### Stop Loss

- Automatic exit price to limit losses
- Example: Entry $100, Stop $95 = max 5% loss

### Take Profit

- Automatic exit price to lock in gains
- Example: Entry $100, Target $110 = 10% profit

---

## 🆘 Support

**Issues or Questions?**

1. Check logs: `tail -f api_server_*.log`
2. Check positions: `python emergency_position_check.py`
3. Emergency close: `python close_all_positions_api.py`
4. WEEX website: https://www.weex.com/

**Remember:** This is LIVE TRADING with REAL MONEY. Start small and monitor closely!

---

**Happy Trading! 🚀📈**
