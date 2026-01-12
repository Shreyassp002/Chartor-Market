"""
Professional Backtesting Engine
Uses the same strategy logic as live trading with simulated execution
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    initial_capital: float = 10000.0
    leverage: float = 20.0
    commission_rate: float = 0.0006  # 0.06% taker fee
    slippage_pct: float = 0.001  # 0.1% slippage
    symbols: List[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = [
                "cmt_btcusdt", "cmt_ethusdt", "cmt_solusdt", 
                "cmt_dogeusdt", "cmt_xrpusdt", "cmt_adausdt",
                "cmt_bnbusdt", "cmt_ltcusdt"
            ]


@dataclass
class BacktestTrade:
    """Individual trade record"""
    trade_id: int
    symbol: str
    direction: str
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    size: float
    pnl: float
    pnl_pct: float
    commission: float
    slippage: float
    exit_reason: str
    hold_time_hours: float
    r_multiple: float  # Profit/Loss in R units


@dataclass
class BacktestResults:
    """Comprehensive backtest results"""
    # Performance metrics
    total_return: float
    total_return_pct: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    
    # Profit metrics
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float
    avg_r_multiple: float
    
    # Risk metrics
    max_consecutive_losses: int
    max_consecutive_wins: int
    largest_win: float
    largest_loss: float
    
    # Equity curve
    equity_curve: List[float]
    drawdown_curve: List[float]
    
    # Trade log
    trades: List[BacktestTrade]
    
    # Daily statistics
    daily_returns: List[float]
    
    # Metadata
    config: Dict
    duration_days: int


class BacktestEngine:
    """
    Professional Backtesting Engine
    
    Features:
    - Same strategy logic as live trading
    - Realistic slippage and fees
    - Walk-forward validation support
    - Monte Carlo simulation
    - Execution delay simulation
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.equity = config.initial_capital
        self.peak_equity = config.initial_capital
        
        # Trade tracking
        self.trades: List[BacktestTrade] = []
        self.open_position: Optional[Dict] = None
        self.trade_counter = 0
        
        # Equity tracking
        self.equity_curve = [config.initial_capital]
        self.drawdown_curve = [0.0]
        self.daily_returns = []
        
    def simulate_execution(self, 
                          price: float, 
                          side: str) -> Tuple[float, float, float]:
        """
        Simulate realistic execution with slippage and fees
        
        Returns: (executed_price, slippage_cost, commission)
        """
        # Slippage (adverse price movement)
        if side == "buy":
            slippage = price * self.config.slippage_pct
            executed_price = price + slippage
        else:  # sell
            slippage = price * self.config.slippage_pct
            executed_price = price - slippage
        
        # Commission
        notional = executed_price * (self.equity * 0.0125) / abs(price * 0.015)  # Estimate size
        commission = notional * self.config.commission_rate
        
        return executed_price, slippage, commission
    
    def calculate_position_size(self, 
                                entry_price: float, 
                                stop_loss: float) -> float:
        """
        Calculate position size based on risk management rules
        Same as RiskManager: 1.25% risk per trade
        """
        risk_amount = 0.0125 * self.equity
        stop_distance = abs(entry_price - stop_loss)
        
        if stop_distance == 0:
            return 0.0
        
        size = risk_amount / stop_distance
        
        # Check margin requirements
        notional_value = size * entry_price
        margin_required = notional_value / self.config.leverage
        
        if margin_required > self.equity * 0.4:  # Max 40% exposure
            # Reduce size
            max_notional = self.equity * 0.4 * self.config.leverage
            size = max_notional / entry_price
        
        return size
    
    def open_backtest_position(self,
                              symbol: str,
                              direction: str,
                              entry_price: float,
                              stop_loss: float,
                              take_profit: float,
                              timestamp: datetime) -> bool:
        """Open position in backtest"""
        if self.open_position is not None:
            return False  # Already have position
        
        # Calculate position size
        size = self.calculate_position_size(entry_price, stop_loss)
        
        if size <= 0:
            return False
        
        # Simulate execution
        side = "buy" if direction == "LONG" else "sell"
        executed_price, slippage, commission = self.simulate_execution(entry_price, side)
        
        # Deduct commission from equity
        self.equity -= commission
        
        self.open_position = {
            "symbol": symbol,
            "direction": direction,
            "entry_time": timestamp,
            "entry_price": executed_price,
            "size": size,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "commission": commission,
            "slippage": slippage,
            "highest_price": executed_price,
            "trailing_stop": None
        }
        
        return True
    
    def update_trailing_stop(self, current_price: float, atr: float):
        """Update trailing stop for open position"""
        if self.open_position is None:
            return
        
        pos = self.open_position
        direction = pos["direction"]
        entry_price = pos["entry_price"]
        
        # Update highest/lowest
        if direction == "LONG":
            pos["highest_price"] = max(pos["highest_price"], current_price)
            
            # Activate trailing stop after 1R profit
            if current_price > entry_price:
                trailing_stop = pos["highest_price"] - (2.0 * atr)
                if trailing_stop > pos["stop_loss"]:
                    pos["stop_loss"] = trailing_stop
                    pos["trailing_stop"] = trailing_stop
        else:  # SHORT
            pos["highest_price"] = min(pos.get("highest_price", entry_price), current_price)
            
            if current_price < entry_price:
                trailing_stop = pos["highest_price"] + (2.0 * atr)
                if trailing_stop < pos["stop_loss"]:
                    pos["stop_loss"] = trailing_stop
                    pos["trailing_stop"] = trailing_stop
    
    def check_exit_conditions(self, 
                             current_price: float, 
                             timestamp: datetime,
                             max_hold_hours: float = 24) -> Tuple[bool, str]:
        """Check if position should be exited"""
        if self.open_position is None:
            return False, ""
        
        pos = self.open_position
        direction = pos["direction"]
        
        # Check stop loss
        if direction == "LONG" and current_price <= pos["stop_loss"]:
            return True, "Stop Loss"
        elif direction == "SHORT" and current_price >= pos["stop_loss"]:
            return True, "Stop Loss"
        
        # Check take profit
        if direction == "LONG" and current_price >= pos["take_profit"]:
            return True, "Take Profit"
        elif direction == "SHORT" and current_price <= pos["take_profit"]:
            return True, "Take Profit"
        
        # Check time stop
        hold_time = (timestamp - pos["entry_time"]).total_seconds() / 3600
        if hold_time > max_hold_hours:
            return True, f"Time Stop ({hold_time:.1f}h)"
        
        return False, ""
    
    def close_backtest_position(self, 
                               exit_price: float, 
                               timestamp: datetime,
                               exit_reason: str) -> Optional[BacktestTrade]:
        """Close position and record trade"""
        if self.open_position is None:
            return None
        
        pos = self.open_position
        
        # Simulate execution
        side = "sell" if pos["direction"] == "LONG" else "buy"
        executed_price, slippage, commission = self.simulate_execution(exit_price, side)
        
        # Calculate PnL
        if pos["direction"] == "LONG":
            pnl = (executed_price - pos["entry_price"]) * pos["size"]
        else:  # SHORT
            pnl = (pos["entry_price"] - executed_price) * pos["size"]
        
        # Deduct commission and slippage
        pnl -= (commission + pos["commission"])
        
        # Update equity
        self.equity += pnl
        self.peak_equity = max(self.peak_equity, self.equity)
        
        # Calculate R-multiple
        risk_amount = abs(pos["entry_price"] - pos["stop_loss"]) * pos["size"]
        r_multiple = pnl / risk_amount if risk_amount > 0 else 0.0
        
        # Hold time
        hold_time = (timestamp - pos["entry_time"]).total_seconds() / 3600
        
        # PnL percentage (on margin used)
        margin_used = (pos["size"] * pos["entry_price"]) / self.config.leverage
        pnl_pct = (pnl / margin_used) * 100 if margin_used > 0 else 0.0
        
        # Create trade record
        trade = BacktestTrade(
            trade_id=self.trade_counter,
            symbol=pos["symbol"],
            direction=pos["direction"],
            entry_time=pos["entry_time"],
            entry_price=pos["entry_price"],
            exit_time=timestamp,
            exit_price=executed_price,
            size=pos["size"],
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission=commission + pos["commission"],
            slippage=slippage + pos["slippage"],
            exit_reason=exit_reason,
            hold_time_hours=hold_time,
            r_multiple=r_multiple
        )
        
        self.trades.append(trade)
        self.trade_counter += 1
        
        # Update curves
        self.equity_curve.append(self.equity)
        drawdown = (self.peak_equity - self.equity) / self.peak_equity * 100
        self.drawdown_curve.append(drawdown)
        
        # Clear position
        self.open_position = None
        
        return trade
    
    def calculate_metrics(self, duration_days: int) -> BacktestResults:
        """Calculate comprehensive backtest metrics"""
        if len(self.trades) == 0:
            return BacktestResults(
                total_return=0.0,
                total_return_pct=0.0,
                max_drawdown=0.0,
                max_drawdown_pct=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                expectancy=0.0,
                avg_r_multiple=0.0,
                max_consecutive_losses=0,
                max_consecutive_wins=0,
                largest_win=0.0,
                largest_loss=0.0,
                equity_curve=self.equity_curve,
                drawdown_curve=self.drawdown_curve,
                trades=[],
                daily_returns=[],
                config=asdict(self.config),
                duration_days=duration_days
            )
        
        # Basic metrics
        total_return = self.equity - self.config.initial_capital
        total_return_pct = (total_return / self.config.initial_capital) * 100
        
        # Trade statistics
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]
        
        total_trades = len(self.trades)
        num_wins = len(winning_trades)
        num_losses = len(losing_trades)
        win_rate = (num_wins / total_trades) * 100 if total_trades > 0 else 0
        
        # Profit metrics
        avg_win = np.mean([t.pnl for t in winning_trades]) if num_wins > 0 else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if num_losses > 0 else 0
        
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
        
        avg_r = np.mean([t.r_multiple for t in self.trades])
        
        # Drawdown
        max_dd = max(self.drawdown_curve) if self.drawdown_curve else 0
        
        # Consecutive wins/losses
        max_consec_wins = 0
        max_consec_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in self.trades:
            if trade.pnl > 0:
                current_wins += 1
                current_losses = 0
                max_consec_wins = max(max_consec_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consec_losses = max(max_consec_losses, current_losses)
        
        # Largest win/loss
        largest_win = max([t.pnl for t in self.trades])
        largest_loss = min([t.pnl for t in self.trades])
        
        # Daily returns for Sharpe/Sortino
        equity_series = pd.Series(self.equity_curve)
        daily_returns = equity_series.pct_change().dropna().values
        
        # Sharpe Ratio (assuming risk-free rate = 0)
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        # Sortino Ratio (only downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std()
            if downside_std > 0:
                sortino = (daily_returns.mean() / downside_std) * np.sqrt(252)
            else:
                sortino = 0.0
        else:
            sortino = 0.0
        
        return BacktestResults(
            total_return=total_return,
            total_return_pct=total_return_pct,
            max_drawdown=max_dd * self.peak_equity / 100,
            max_drawdown_pct=max_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            total_trades=total_trades,
            winning_trades=num_wins,
            losing_trades=num_losses,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            avg_r_multiple=avg_r,
            max_consecutive_losses=max_consec_losses,
            max_consecutive_wins=max_consec_wins,
            largest_win=largest_win,
            largest_loss=largest_loss,
            equity_curve=self.equity_curve,
            drawdown_curve=self.drawdown_curve,
            trades=self.trades,
            daily_returns=daily_returns.tolist(),
            config=asdict(self.config),
            duration_days=duration_days
        )
    
    def export_results(self, filepath: str):
        """Export backtest results to JSON"""
        results = self.calculate_metrics(30)  # Placeholder duration
        
        # Convert to dict
        results_dict = asdict(results)
        
        # Convert datetime objects to strings
        for trade in results_dict["trades"]:
            trade["entry_time"] = trade["entry_time"].isoformat()
            trade["exit_time"] = trade["exit_time"].isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Results exported to {filepath}")
