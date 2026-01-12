"""
Unified Position Lifecycle Manager
Single source of truth for all positions across Sentinel and Institutional systems
Handles: Open → Monitor → Stop Loss → Take Profit → Trailing Stop → Close
"""
import time
import threading
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

from core.weex_api import WeexClient
from core.db_manager import update_or_create_position, close_position as db_close_position, save_trade, get_db_connection


class PositionStatus(Enum):
    """Position lifecycle states"""
    OPEN = "OPEN"
    MONITORING = "MONITORING"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"


@dataclass
class Position:
    """Unified position model"""
    symbol: str
    side: str  
    direction: str  
    size: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    trailing_stop: Optional[float]
    leverage: int
    margin_used: float
    unrealized_pnl: float
    unrealized_pnl_pct: float
    opened_at: datetime
    order_id: Optional[str]
    status: PositionStatus
    source: str  
    highest_price: float  
    lowest_price: float
    atr: float
    metadata: Dict
    # Partial exit tracking
    initial_size: float = 0.0  # Original position size
    partial_exits: List[Dict] = field(default_factory=list)  # Track each partial exit
    profit_levels_hit: List[str] = field(default_factory=list)  # Track which profit levels were hit
    
    def __post_init__(self):
        if self.initial_size == 0.0:
            self.initial_size = self.size


class UnifiedPositionManager:
    """
    Unified Position Lifecycle Manager
    
    Single source of truth for ALL positions (Sentinel + Institutional)
    
    Features:
    - Automatic stop-loss execution
    - Automatic take-profit execution
    - Trailing stop management
    - Real-time PnL tracking
    - Database synchronization
    - 5-second monitoring loop
    """
    
    MONITOR_INTERVAL = 5.0  
    TRAILING_STOP_ACTIVATION_R = 1.0  
    TRAILING_STOP_ATR_MULTIPLIER = 2.0
    
    # Partial exit configuration - scale out at profit milestones
    PARTIAL_EXIT_ENABLED = True
    PARTIAL_EXIT_LEVELS = [
        {"r_multiple": 1.0, "exit_pct": 0.25, "label": "1R"},  # Take 25% off at 1R
        {"r_multiple": 2.0, "exit_pct": 0.25, "label": "2R"},  # Take 25% off at 2R  
        {"r_multiple": 3.0, "exit_pct": 0.25, "label": "3R"},  # Take 25% off at 3R
    ]  # Remaining 25% runs with trailing stop
    
    def __init__(self, weex_client: WeexClient, logger: Optional[logging.Logger] = None):
        self.client = weex_client
        self.logger = logger or logging.getLogger(__name__)
        
        # Position storage
        self.positions: Dict[str, Position] = {}
        self.position_lock = threading.Lock()
        
        # Monitoring control
        self.monitor_running = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.total_positions_opened = 0
        self.total_positions_closed = 0
        self.total_sl_hits = 0
        self.total_tp_hits = 0
        self.total_trailing_stops = 0
        
        self.logger.info(" Unified Position Manager initialized")
    
    def open_position(self,
                     symbol: str,
                     side: str,
                     direction: str,
                     size: float,
                     entry_price: float,
                     stop_loss: float,
                     take_profit: float,
                     leverage: int,
                     margin_used: float,
                     atr: float,
                     order_id: Optional[str] = None,
                     source: str = "SENTINEL",
                     metadata: Optional[Dict] = None) -> bool:
        """
        Open a new position with full lifecycle management
        
        Args:
            symbol: Trading pair (e.g., "cmt_btcusdt")
            side: "buy" or "sell" (for WEEX API)
            direction: "LONG" or "SHORT" (for tracking)
            size: Position size in contracts
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            leverage: Leverage used
            margin_used: Margin allocated
            atr: Current ATR value
            order_id: WEEX order ID
            source: "SENTINEL" or "INSTITUTIONAL"
            metadata: Additional data
        
        Returns:
            bool: Success status
        """
        with self.position_lock:
            # Check if position already exists
            if symbol in self.positions:
                self.logger.warning(f"Position already exists for {symbol} - closing old one first")
                self.close_position(symbol, entry_price, "Position replacement")
            
            # Create position
            position = Position(
                symbol=symbol,
                side=side,
                direction=direction,
                size=size,
                entry_price=entry_price,
                current_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                trailing_stop=None,
                leverage=leverage,
                margin_used=margin_used,
                unrealized_pnl=0.0,
                unrealized_pnl_pct=0.0,
                opened_at=datetime.now(),
                order_id=order_id,
                status=PositionStatus.OPEN,
                source=source,
                highest_price=entry_price,
                lowest_price=entry_price,
                atr=atr,
                metadata=metadata or {}
            )
            
            self.positions[symbol] = position
            self.total_positions_opened += 1
            
            # Sync to database
            try:
                update_or_create_position({
                    "symbol": symbol,
                    "side": direction,
                    "size": size,
                    "entry_price": entry_price,
                    "current_price": entry_price,
                    "unrealized_pnl": 0.0,
                    "leverage": leverage,
                    "order_id": order_id
                })
            except Exception as e:
                self.logger.error(f"Failed to sync position to database: {e}")
            
            self.logger.info(f"✅ Position opened: {symbol} {direction} {size} @ ${entry_price:.2f}")
            self.logger.info(f"   SL: ${stop_loss:.2f} | TP: ${take_profit:.2f} | Source: {source}")
            
            # Start monitoring if not already running
            if not self.monitor_running:
                self.start_monitoring()
            
            return True
    
    def update_position_price(self, symbol: str, current_price: float):
        """Update position with current market price"""
        with self.position_lock:
            if symbol not in self.positions:
                return
            
            pos = self.positions[symbol]
            pos.current_price = current_price
            
            # Update highest/lowest for trailing stop
            pos.highest_price = max(pos.highest_price, current_price)
            pos.lowest_price = min(pos.lowest_price, current_price)
            
            # Calculate unrealized PnL
            if pos.direction == "LONG":
                pos.unrealized_pnl = (current_price - pos.entry_price) * pos.size
            else:  # SHORT
                pos.unrealized_pnl = (pos.entry_price - current_price) * pos.size
            
            if pos.margin_used > 0:
                pos.unrealized_pnl_pct = (pos.unrealized_pnl / pos.margin_used) * 100
            
            # Update trailing stop if in profit
            self._update_trailing_stop(pos)
            
            # Sync to database
            try:
                update_or_create_position({
                    "symbol": symbol,
                    "side": pos.direction,
                    "size": pos.size,
                    "entry_price": pos.entry_price,
                    "current_price": current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "leverage": pos.leverage,
                    "order_id": pos.order_id
                })
            except Exception as db_err:
                self.logger.error(f"Failed to sync position to database: {db_err}", exc_info=True)
    
    def _update_trailing_stop(self, pos: Position):
        """Update trailing stop if position is in profit"""
        # Calculate profit in R (risk units)
        risk_distance = abs(pos.entry_price - pos.stop_loss)
        if risk_distance == 0:
            return
        
        if pos.direction == "LONG":
            current_profit = pos.current_price - pos.entry_price  
        else:  # SHORT
            current_profit = pos.entry_price - pos.current_price  
        
        # Only activate trailing stop if ACTUALLY in profit
        if current_profit <= 0:
            return  # Don't trail when in loss
        
        profit_in_r = current_profit / risk_distance
        
        # Activate trailing stop earlier if partial exits were taken (runner position)
        activation_threshold = self.TRAILING_STOP_ACTIVATION_R
        if len(pos.profit_levels_hit) > 0:
            # After taking partials, trail more aggressively on remaining position
            activation_threshold = 0.5  # Start trailing at 0.5R instead of 1R
        
        # Only activate trailing stop after threshold profit
        if profit_in_r < activation_threshold:
            return
        
        # Calculate new trailing stop
        trail_distance = self.TRAILING_STOP_ATR_MULTIPLIER * pos.atr
        
        if pos.direction == "LONG":
            new_trailing_stop = pos.highest_price - trail_distance
            # Only move stop up, never down
            if pos.trailing_stop is None or new_trailing_stop > pos.trailing_stop:
                pos.trailing_stop = new_trailing_stop
                # Update main stop loss if trailing stop is better
                if new_trailing_stop > pos.stop_loss:
                    pos.stop_loss = new_trailing_stop
                    self.logger.info(f"📈 Trailing stop updated for {pos.symbol}: ${new_trailing_stop:.2f}")
        else:  # SHORT
            new_trailing_stop = pos.lowest_price + trail_distance
            # Only move stop down, never up
            if pos.trailing_stop is None or new_trailing_stop < pos.trailing_stop:
                pos.trailing_stop = new_trailing_stop
                # Update main stop loss if trailing stop is better
                if new_trailing_stop < pos.stop_loss:
                    pos.stop_loss = new_trailing_stop
                    self.logger.info(f"📈 Trailing stop updated for {pos.symbol}: ${new_trailing_stop:.2f}")
    
    def _execute_partial_exit(self, pos: Position, exit_size: float, level_label: str, profit_r: float):
        """
        Execute a partial exit at a profit milestone
        
        Args:
            pos: Position to partially exit
            exit_size: Size to exit
            level_label: Label for this profit level (e.g., "1R", "2R")
            profit_r: Current profit in R-multiples
        """
        try:
            # Round exit size to stepSize
            # Use same stepSize logic as trading_orchestrator
            step_sizes = {
                "cmt_btcusdt": 0.001,
                "cmt_ethusdt": 0.01,
                "cmt_solusdt": 0.1,
                "cmt_dogeusdt": 100,
                "cmt_xrpusdt": 1,
                "cmt_adausdt": 10,
                "cmt_bnbusdt": 0.01,
                "cmt_ltcusdt": 0.1
            }
            step_size = step_sizes.get(pos.symbol, 1)
            exit_size = (exit_size // step_size) * step_size
            
            if exit_size < step_size:
                self.logger.warning(f"Partial exit size {exit_size} too small for {pos.symbol} (stepSize: {step_size})")
                return
            
            # Execute partial close order
            close_side = "sell" if pos.direction == "LONG" else "buy"
            
            self.logger.info(f"💰 Partial exit at {level_label} ({profit_r:.2f}R): {exit_size} of {pos.size} contracts")
            
            result = self.client.place_order(
                side=close_side,
                size=str(exit_size),
                symbol=pos.symbol,
                order_type="market"
            )
            
            if result and result.get("code") == "00000":
                # Calculate PnL for this partial
                if pos.direction == "LONG":
                    partial_pnl = (pos.current_price - pos.entry_price) * exit_size
                else:
                    partial_pnl = (pos.entry_price - pos.current_price) * exit_size
                
                # Track partial exit
                partial_record = {
                    "timestamp": datetime.now(),
                    "level": level_label,
                    "size": exit_size,
                    "price": pos.current_price,
                    "pnl": partial_pnl,
                    "r_multiple": profit_r
                }
                pos.partial_exits.append(partial_record)
                pos.profit_levels_hit.append(level_label)
                
                # Reduce position size
                pos.size -= exit_size
                
                self.logger.info(f"✅ Partial exit executed: Locked in ${partial_pnl:.2f} profit")
                self.logger.info(f"   Remaining position: {pos.size} contracts")
                
                # Move stop to breakeven after first profit take
                if level_label == "1R":
                    pos.stop_loss = pos.entry_price
                    self.logger.info(f"🔒 Stop loss moved to breakeven: ${pos.entry_price:.2f}")
                    
            else:
                self.logger.error(f"❌ Partial exit failed: {result}")
                
        except Exception as e:
            self.logger.error(f"Error executing partial exit: {e}", exc_info=True)
    
    def check_exit_conditions(self, pos: Position) -> Tuple[bool, str]:
        """
        Check if position should be exited (full or partial)
        
        Returns: (should_exit, reason)
        """
        # Check stop loss
        if pos.direction == "LONG" and pos.current_price <= pos.stop_loss:
            return True, "Stop loss hit"
        elif pos.direction == "SHORT" and pos.current_price >= pos.stop_loss:
            return True, "Stop loss hit"
        
        # Calculate profit in R-multiples for partial exits
        risk_distance = abs(pos.entry_price - pos.stop_loss)
        if risk_distance > 0 and self.PARTIAL_EXIT_ENABLED:
            if pos.direction == "LONG":
                profit = pos.current_price - pos.entry_price
            else:
                profit = pos.entry_price - pos.current_price
            
            profit_r = profit / risk_distance
            
            # Check each partial exit level
            for level in self.PARTIAL_EXIT_LEVELS:
                level_label = level["label"]
                if profit_r >= level["r_multiple"] and level_label not in pos.profit_levels_hit:
                    # Execute partial exit
                    exit_size = pos.initial_size * level["exit_pct"]
                    if exit_size > 0 and pos.size >= exit_size:
                        self._execute_partial_exit(pos, exit_size, level_label, profit_r)
        
        # Check full take profit (only after partial exits or if disabled)
        if pos.direction == "LONG" and pos.current_price >= pos.take_profit:
            return True, "Take profit hit (full exit)"
        elif pos.direction == "SHORT" and pos.current_price <= pos.take_profit:
            return True, "Take profit hit (full exit)"
        
        # Momentum-based exit: if in profit >2R but retracing >40% from peak
        if risk_distance > 0:
            if pos.direction == "LONG":
                peak_profit = pos.highest_price - pos.entry_price
                current_profit = pos.current_price - pos.entry_price
            else:
                peak_profit = pos.entry_price - pos.lowest_price
                current_profit = pos.entry_price - pos.current_price
            
            peak_r = peak_profit / risk_distance
            current_r = current_profit / risk_distance
            
            # If hit 2R+ but now retraced >40% from peak, exit remaining position
            if peak_r >= 2.0 and current_profit > 0:
                retrace_pct = (peak_profit - current_profit) / peak_profit if peak_profit > 0 else 0
                if retrace_pct > 0.40:  # 40% retrace from peak
                    return True, f"Momentum exit (retraced {retrace_pct*100:.1f}% from {peak_r:.1f}R peak)"
        
        # Check time stop (24 hours max)
        time_in_position = (datetime.now() - pos.opened_at).total_seconds() / 3600
        if time_in_position > 24:
            return True, f"Time stop ({time_in_position:.1f}h)"
        
        return False, "Active"
    
    def close_position(self, symbol: str, exit_price: float, reason: str = "Manual") -> Optional[Dict]:
        """
        Close position and execute market order on WEEX
        
        Args:
            symbol: Trading pair
            exit_price: Exit price
            reason: Reason for closing
        
        Returns:
            Trade record dict or None
        """
        with self.position_lock:
            if symbol not in self.positions:
                self.logger.warning(f"No position found for {symbol}")
                return None
            
            pos = self.positions[symbol]
            
            # Calculate final PnL
            if pos.direction == "LONG":
                realized_pnl = (exit_price - pos.entry_price) * pos.size
            else:  # SHORT
                realized_pnl = (pos.entry_price - exit_price) * pos.size
            
            realized_pnl_pct = (realized_pnl / pos.margin_used) * 100 if pos.margin_used > 0 else 0
            hold_time = (datetime.now() - pos.opened_at).total_seconds() / 3600
            
            self.logger.info(f"🔚 Closing position: {symbol} {pos.direction}")
            self.logger.info(f"   Reason: {reason}")
            self.logger.info(f"   Entry: ${pos.entry_price:.2f} → Exit: ${exit_price:.2f}")
            self.logger.info(f"   PnL: ${realized_pnl:.2f} ({realized_pnl_pct:.2f}%)")
            self.logger.info(f"   Hold time: {hold_time:.1f}h")
            
            try:
                weex_positions = self.client.get_positions()
                position_exists_on_weex = False
                
                if weex_positions and weex_positions.get("code") == "00000":
                    positions_data = weex_positions.get("data", [])
                    for weex_pos in positions_data:
                        if weex_pos.get("symbol") == symbol:
                            position_exists_on_weex = True
                            # Verify size matches
                            weex_size = float(weex_pos.get("size", 0))
                            if abs(weex_size - pos.size) > 0.001:
                                self.logger.warning(f"Position size mismatch: Local={pos.size}, WEEX={weex_size}")
                            break
                
                if not position_exists_on_weex:
                    self.logger.warning(f"⚠️ Position {symbol} not found on WEEX - may have been closed externally (liquidation?)")
                    self.logger.info(f"Skipping close order to prevent opening new position")
                    close_order_id = None
                else:
                    # Execute close order on WEEX
                    close_side = "sell" if pos.direction == "LONG" else "buy"
                    
                    result = self.client.place_order(
                        side=close_side,
                        size=str(pos.size),
                        symbol=symbol,
                        order_type="market"
                    )
                    
                    if result and result.get("code") == "00000":
                        close_order_id = result.get("data", {}).get("orderId")
                        self.logger.info(f"✅ Close order executed: {close_order_id}")
                    else:
                        self.logger.error(f"❌ Close order failed: {result}")
                        close_order_id = None
                        
            except Exception as e:
                self.logger.error(f"Error during position close: {e}", exc_info=True)
                close_order_id = None
            
            # Calculate total PnL including partial exits
            total_pnl = realized_pnl
            partial_exits_summary = ""
            if len(pos.partial_exits) > 0:
                partial_pnl_sum = sum(p["pnl"] for p in pos.partial_exits)
                total_pnl += partial_pnl_sum
                partial_exits_summary = f" | Partials: {len(pos.partial_exits)} exits, ${partial_pnl_sum:.2f} locked"
                self.logger.info(f"   Total PnL with partials: ${total_pnl:.2f} (Final: ${realized_pnl:.2f} + Partials: ${partial_pnl_sum:.2f})")
            
            # Create trade record
            trade_record = {
                "symbol": symbol,
                "side": pos.side,
                "size": pos.initial_size,  # Record original size
                "final_size": pos.size,  # Size at close (after partials)
                "price": exit_price,
                "order_id": close_order_id or pos.order_id,
                "order_type": "market",
                "status": "filled",
                "pnl": total_pnl,  # Total PnL including partials
                "final_exit_pnl": realized_pnl,  # PnL from final exit only
                "partial_exits": pos.partial_exits,
                "fees": 0.0,  # Calculate if available
                "notes": f"{reason} | {pos.source} | Entry: {pos.entry_price:.2f} | Hold: {hold_time:.1f}h{partial_exits_summary}"
            }
            
            # Save to database
            try:
                save_trade(trade_record)
                db_close_position(symbol, pos.direction)
            except Exception as e:
                self.logger.error(f"Failed to save trade to database: {e}")
            
            # Update statistics
            self.total_positions_closed += 1
            if "Stop loss" in reason:
                self.total_sl_hits += 1
            elif "Take profit" in reason:
                self.total_tp_hits += 1
            elif "Trailing stop" in reason:
                self.total_trailing_stops += 1
            
            # Remove from tracking
            del self.positions[symbol]
            pos.status = PositionStatus.CLOSED
            
            return trade_record
    
    def monitor_loop(self):
        """Main monitoring loop - runs every 5 seconds"""
        self.logger.info(" Position monitoring started (5s interval)")
        
        while self.monitor_running:
            try:
                with self.position_lock:
                    symbols_to_close = []
                    
                    for symbol, pos in self.positions.items():
                        if pos.status != PositionStatus.MONITORING:
                            pos.status = PositionStatus.MONITORING
                        
                        # Fetch current price
                        try:
                            candles = self.client.fetch_candles(symbol=symbol, limit=1)
                            if candles and len(candles) > 0:
                                current_price = float(candles[-1][4])  # Close price
                                self.update_position_price(symbol, current_price)
                                
                                # Check exit conditions
                                should_exit, reason = self.check_exit_conditions(pos)
                                if should_exit:
                                    symbols_to_close.append((symbol, current_price, reason))
                        except Exception as e:
                            self.logger.error(f"Error fetching price for {symbol}: {e}")
                    
                    # Close positions (outside iteration to avoid dict size change)
                    for symbol, price, reason in symbols_to_close:
                        self.close_position(symbol, price, reason)
                
                # Sleep until next check
                time.sleep(self.MONITOR_INTERVAL)
                
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}", exc_info=True)
                time.sleep(self.MONITOR_INTERVAL)
    
    def start_monitoring(self):
        """Start the monitoring thread"""
        if self.monitor_running:
            self.logger.warning("Monitoring already running")
            return
        
        self.monitor_running = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("✅ Position monitoring thread started")
    
    def stop_monitoring(self):
        """Stop the monitoring thread"""
        if not self.monitor_running:
            return
        
        self.logger.info("🛑 Stopping position monitoring...")
        self.monitor_running = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10)
        
        self.logger.info("✅ Position monitoring stopped")
    
    def get_all_positions(self) -> List[Position]:
        """Get all open positions"""
        with self.position_lock:
            return list(self.positions.values())
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get specific position"""
        with self.position_lock:
            return self.positions.get(symbol)
    
    def get_statistics(self) -> Dict:
        """Get monitoring statistics"""
        return {
            "total_positions_opened": self.total_positions_opened,
            "total_positions_closed": self.total_positions_closed,
            "currently_open": len(self.positions),
            "stop_loss_hits": self.total_sl_hits,
            "take_profit_hits": self.total_tp_hits,
            "trailing_stop_hits": self.total_trailing_stops,
            "monitoring_active": self.monitor_running
        }
    
    def shutdown(self):
        """Shutdown and close all positions"""
        self.logger.info("🛑 Shutting down Position Manager...")
        
        # Stop monitoring
        self.stop_monitoring()
        
        # Close all open positions
        with self.position_lock:
            symbols = list(self.positions.keys())
        
        for symbol in symbols:
            pos = self.get_position(symbol)
            if pos:
                self.close_position(symbol, pos.current_price, "System shutdown")
        
        self.logger.info("✅ Position Manager shutdown complete")


# Global instance (initialized in api_server.py)
_position_manager_instance: Optional[UnifiedPositionManager] = None


def get_position_manager() -> Optional[UnifiedPositionManager]:
    """Get global position manager instance"""
    return _position_manager_instance


def initialize_position_manager(weex_client: WeexClient, logger: Optional[logging.Logger] = None):
    """Initialize global position manager"""
    global _position_manager_instance
    if _position_manager_instance is None:
        _position_manager_instance = UnifiedPositionManager(weex_client, logger)
    return _position_manager_instance
