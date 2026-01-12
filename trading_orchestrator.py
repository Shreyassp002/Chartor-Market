"""
Institutional Trading Orchestrator
Main engine that coordinates all components: strategy, regime, risk, execution, rotation
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
import time

from strategy.intraday_engine import IntradayMomentumEngine, SignalResult
from regime.ofras import OFRASRegimeDetector, RegimeType, RegimeState
from risk.risk_manager import RiskManager, PositionState, PortfolioRisk
from execution.execution_engine import ExecutionEngine, OrderResult
from core.sentiment_live import get_sentiment_feed
from core.llm_brain import get_trading_decision
from core.ml_analyst import MLAnalyst


@dataclass
class AssetScore:
    """Asset opportunity score for rotation"""
    symbol: str
    score: float
    signal: str
    confidence: float
    regime: str
    metadata: Dict


class TradingOrchestrator:
    """
    Main Trading System Orchestrator
    
    Runs every 30 seconds and:
    1. Scores all enabled assets
    2. Detects market regime
    3. Rotates capital to highest probability opportunity
    4. Manages open positions
    5. Enforces risk limits
    """
    
    ENABLED_SYMBOLS = [
        "cmt_btcusdt", "cmt_ethusdt", "cmt_solusdt", 
        "cmt_dogeusdt", "cmt_xrpusdt", "cmt_adausdt",
        "cmt_bnbusdt", "cmt_ltcusdt"
    ]
    
    CYCLE_INTERVAL = 30  # seconds
    MIN_SIGNAL_STRENGTH = 25.0  # Minimum signal strength to trade (lowered for testing)
    
    def __init__(self,
                 weex_client,
                 initial_equity: float = 10000.0,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize trading orchestrator
        
        Args:
            weex_client: WeexClient instance
            initial_equity: Starting capital
            logger: Optional logger
        """
        self.client = weex_client
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize components
        self.strategy_engine = IntradayMomentumEngine()
        self.regime_detector = OFRASRegimeDetector()
        self.risk_manager = RiskManager(initial_equity=initial_equity)
        self.execution_engine = ExecutionEngine(weex_client, logger=self.logger)
        self.sentiment_feed = get_sentiment_feed()  # Real-time sentiment feed
        self.ml_analyst = MLAnalyst()  # Local ML predictions
        
        self.logger.info("🤖 AI Components initialized: Gemini + Local ML + Sentiment Feed")
        
        # State tracking
        self.current_regime: Optional[RegimeState] = None
        self.asset_scores: Dict[str, AssetScore] = {}
        self.last_cycle_time = datetime.now()
        self.cycle_count = 0
        
        # Performance tracking
        self.total_signals_generated = 0
        self.trades_executed = 0
        self.signals_filtered_by_regime = 0
        self.signals_filtered_by_risk = 0
        
        self.logger.info("🚀 Institutional Trading Orchestrator initialized")
        self.logger.info(f"   Enabled symbols: {len(self.ENABLED_SYMBOLS)}")
        self.logger.info(f"   Initial equity: ${initial_equity:,.2f}")
    
    def fetch_market_data(self, symbol: str, limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch and prepare market data for a symbol"""
        try:
            self.logger.info(f"      📡 Fetching {limit} candles for {symbol}...")
            candles = self.client.fetch_candles(symbol=symbol, limit=limit)
            
            if not candles or len(candles) < 50:
                self.logger.warning(f"Insufficient data for {symbol}: got {len(candles) if candles else 0} candles")
                return None
            
            self.logger.info(f"      ✓ Received {len(candles)} candles")
            
            # Convert to DataFrame
            df = pd.DataFrame(candles, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base', 
                'taker_buy_quote', 'ignore'
            ])
            
            # Keep only needed columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Log first and last candle for verification
            first = df.iloc[0]
            last = df.iloc[-1]
            self.logger.info(f"      First candle: O={first['open']:.2f} H={first['high']:.2f} L={first['low']:.2f} C={first['close']:.2f}")
            self.logger.info(f"      Last candle:  O={last['open']:.2f} H={last['high']:.2f} L={last['low']:.2f} C={last['close']:.2f} Vol={last['volume']:.2f}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}", exc_info=True)
            return None
    
    def get_funding_rate(self, symbol: str) -> Optional[float]:
        """Get current funding rate (if available from exchange)"""
        try:
            # Placeholder - implement based on WEEX API
            # funding_data = self.client.get_funding_rate(symbol)
            # return funding_data.get("fundingRate", 0.0)
            return None  # Return None if not available
        except Exception as e:
            self.logger.debug(f"Funding rate not available for {symbol}: {e}")
            return None
    
    def get_oi_change(self, symbol: str) -> Optional[float]:
        """Get open interest change (if available)"""
        try:
            # Placeholder - implement based on WEEX API
            return None
        except Exception as e:
            self.logger.debug(f"OI data not available for {symbol}: {e}")
            return None
    
    def score_asset(self, symbol: str) -> Optional[AssetScore]:
        """
        Score an asset for opportunity (ENHANCED: AI-powered multi-factor analysis)
        
        Integrates:
        - Technical indicators (IntradayMomentumEngine)
        - Market regime detection (OFRAS)
        - Real-time sentiment analysis (CryptoPanic + FinBERT)
        - Local ML predictions (XGBoost)
        - Gemini AI final validation
        
        Formula:
        Score = 0.20×Tech + 0.15×Momentum + 0.12×Vol + 0.10×Funding + 0.08×OBI + 0.15×Sentiment + 0.15×ML + 0.10×Gemini - 0.05×Risk
        """
        try:
            # Fetch market data
            df = self.fetch_market_data(symbol)
            if df is None or len(df) < 50:
                return None
            
            # Train local ML model on latest data
            ml_trained = self.ml_analyst.train_model(df.to_dict('records'))
            
            # Detect regime
            funding_rate = self.get_funding_rate(symbol)
            oi_change = self.get_oi_change(symbol)
            
            regime_state = self.regime_detector.detect_regime(
                df, funding_rate, oi_change
            )
            
            # Generate signal
            signal_result = self.strategy_engine.generate_signal(
                df, funding_rate, oi_change
            )
            
            # Track signal generation
            if signal_result.signal in ["LONG", "SHORT"] and signal_result.strength > 0:
                self.total_signals_generated += 1
            
            # Always log signal details to diagnose issues
            self.logger.info(f"   {symbol}: {signal_result.signal} ({signal_result.signal_type}) | "
                           f"Strength: {signal_result.strength:.1f} | Regime: {regime_state.regime.value}")
            
            # Apply regime filters
            regime_filters = self.regime_detector.get_strategy_filter(regime_state)
            
            # Show allowed strategies
            if signal_result.signal in ["LONG", "SHORT"]:
                self.logger.info(f"      Allowed strategies: {[k for k,v in regime_filters.items() if v]}")
            
            # Check if signal type is enabled in current regime
            signal_allowed = False
            if signal_result.signal in ["LONG", "SHORT"]:
                # Check if the signal_type matches enabled strategies
                signal_allowed = regime_filters.get(signal_result.signal_type, False)
            
            if not signal_allowed:
                if signal_result.signal in ["LONG", "SHORT"] and signal_result.strength > 30:
                    self.logger.info(f"      ❌ FILTERED: {signal_result.signal_type} not allowed in {regime_state.regime.value} mode")
                self.logger.debug(f"{symbol}: FILTERED by regime - {regime_state.regime.value} doesn't allow {signal_result.signal_type} (filters: {regime_filters})")
                self.signals_filtered_by_regime += 1
                signal_result.strength = 0.0  # Filter out
            
            # Get AI sentiment analysis
            symbol_clean = symbol.replace("cmt_", "").replace("usdt", "").upper()
            sentiment_data = self.sentiment_feed.get_market_sentiment(symbol_clean)
            sentiment_score = sentiment_data['score']  # -1.0 to 1.0
            sentiment_label = sentiment_data['label']  # POSITIVE/NEGATIVE/NEUTRAL
            
            # Get Local ML prediction
            from core.analysis import analyze_market_structure
            market_state = analyze_market_structure(df.to_dict('records'))
            ml_direction = "NEUTRAL"
            ml_confidence = 0.0
            ml_prediction = None
            
            if ml_trained and market_state:
                ml_direction, ml_confidence = self.ml_analyst.predict_next_move(market_state)
                ml_prediction = {"direction": ml_direction, "confidence": ml_confidence}
            
            # Get Gemini AI consultation (only for strong signals to save API quota)
            gemini_decision = "NEUTRAL"
            gemini_confidence = 0.0
            gemini_approved = False
            
            if signal_result.signal in ["LONG", "SHORT"] and signal_allowed and signal_result.strength > 40:
                ai_result = get_trading_decision(
                    market_state,
                    symbol=symbol,
                    use_cache=True,  # Use cache to save quota
                    ml_prediction=ml_prediction,
                    sentiment={
                        "label": sentiment_label,
                        "score": sentiment_score,
                        "source": sentiment_data['source'],
                        "headline": sentiment_data.get('latest_headline', '')
                    }
                )
                
                gemini_decision = ai_result.get("decision", "WAIT")
                gemini_confidence = ai_result.get("confidence", 0)
                gemini_source = ai_result.get("source", "UNKNOWN")
                
                # Check if Gemini approves the signal direction
                if gemini_decision == signal_result.signal:
                    gemini_approved = True
                
            # Log all AI components (wrapped in try-except to prevent logging errors)
            if signal_result.signal in ["LONG", "SHORT"]:
                try:
                    self.logger.info(f"      📰 Sentiment ({sentiment_data['source']}): {sentiment_label} (score: {sentiment_score:.2f})")
                    if sentiment_data.get('latest_headline'):
                        self.logger.info(f"         {sentiment_data['latest_headline'][:80]}...")
                    
                    if ml_prediction:
                        self.logger.info(f"      🤖 Local ML: Predicts {ml_direction} ({ml_confidence}% confidence)")
                    
                    if gemini_decision != "NEUTRAL":
                        approval_emoji = "✅" if gemini_approved else "⚠️"
                        self.logger.info(f"      {approval_emoji} Gemini AI: {gemini_decision} ({gemini_confidence}% confidence)")
                except Exception as log_err:
                    # Silently ignore logging errors to prevent disruption
                    pass
            
            # Composite score using the enhanced formula
            score = 0.0
            
            # Technical component (20% - reduced to make room for AI)
            trend_factor = signal_result.confidence_factors.get("momentum", 0) / 100
            score += 0.20 * trend_factor * 100
            
            # Momentum component (15% - reduced)
            momentum_factor = signal_result.confidence_factors.get("trend_strength", 0) / 100
            score += 0.15 * momentum_factor * 100
            
            # Volatility component (12% - reduced)
            vol_factor = signal_result.confidence_factors.get("volatility_compression", 0) / 100
            score += 0.12 * vol_factor * 100
            
            # Funding pressure (10% - reduced)
            fp_factor = signal_result.confidence_factors.get("funding_pressure", 0) / 100 if "funding_pressure" in signal_result.confidence_factors else 0
            score += 0.10 * fp_factor * 100
            
            # Orderbook imbalance (8% - reduced)
            obi_factor = signal_result.confidence_factors.get("orderbook_imbalance", 0) / 100 if "orderbook_imbalance" in signal_result.confidence_factors else 0
            score += 0.08 * obi_factor * 100
            
            # AI Sentiment component (15%)
            # Positive sentiment boosts LONG signals, negative boosts SHORT signals
            sentiment_alignment = 0.0
            if signal_result.signal == "LONG":
                sentiment_alignment = max(0, sentiment_score)  # 0 to 1 for LONG
            elif signal_result.signal == "SHORT":
                sentiment_alignment = max(0, -sentiment_score)  # 0 to 1 for SHORT
            score += 0.15 * sentiment_alignment * 100
            
            # Local ML component (15%)
            ml_alignment = 0.0
            if ml_prediction:
                if signal_result.signal == ml_direction:
                    ml_alignment = ml_confidence / 100  # 0 to 1
                elif ml_direction == "NEUTRAL":
                    ml_alignment = 0.5  # Neutral doesn't hurt or help
                # If ML contradicts, ml_alignment stays 0 (penalty)
            score += 0.15 * ml_alignment * 100
            
            # Gemini AI component (10%)
            gemini_boost = 0.0
            if gemini_decision != "NEUTRAL":
                if gemini_approved:
                    gemini_boost = gemini_confidence / 100  # 0 to 1 boost
                else:
                    gemini_boost = -(gemini_confidence / 100) * 0.5  # Penalty for disagreement
            score += 0.10 * gemini_boost * 100
            
            # Risk penalty (5%)
            # Lower score if regime confidence is low
            risk_penalty = (100 - regime_state.confidence) / 100
            score -= 0.05 * risk_penalty * 100
            
            # Combine with signal strength
            final_score = (score * 0.5 + signal_result.strength * 0.5)
            
            return AssetScore(
                symbol=symbol,
                score=final_score,
                signal=signal_result.signal,
                confidence=signal_result.strength,
                regime=regime_state.regime.value,
                metadata={
                    "signal_result": signal_result,
                    "regime_state": regime_state,
                    "filters_applied": regime_filters,
                    "signal_allowed": signal_allowed,
                    "sentiment_label": sentiment_label,
                    "sentiment_score": sentiment_score,
                    "sentiment_source": sentiment_data['source'],
                    "ml_direction": ml_direction,
                    "ml_confidence": ml_confidence,
                    "ml_trained": ml_trained,
                    "gemini_decision": gemini_decision,
                    "gemini_confidence": gemini_confidence,
                    "gemini_approved": gemini_approved
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error scoring {symbol}: {e}")
            return None
    
    def select_best_opportunity(self) -> Optional[AssetScore]:
        """
        Select the highest-scoring asset that passes all filters
        
        Enforces:
        - Correlation group constraints
        - Minimum signal strength
        - Risk limits
        """
        # Score all assets
        self.asset_scores.clear()
        
        signals_found = 0
        for symbol in self.ENABLED_SYMBOLS:
            asset_score = self.score_asset(symbol)
            if asset_score and asset_score.score > 0:
                self.asset_scores[symbol] = asset_score
                if asset_score.signal in ["LONG", "SHORT"]:
                    signals_found += 1
        
        self.logger.info(f"   Scanned {len(self.ENABLED_SYMBOLS)} assets, found {signals_found} potential signals")
        
        if not self.asset_scores:
            self.logger.debug("No valid opportunities found")
            return None
        
        # Sort by score
        sorted_assets = sorted(
            self.asset_scores.values(),
            key=lambda x: x.score,
            reverse=True
        )
        
        # Check each asset in order
        for asset in sorted_assets:
            # Check minimum signal strength
            if asset.score < self.MIN_SIGNAL_STRENGTH:
                continue
            
            # Check if signal is actionable
            if asset.signal not in ["LONG", "SHORT"]:
                continue
            
            # Check risk constraints
            can_trade, reasons = self.risk_manager.can_open_position(asset.symbol)
            if not can_trade:
                self.logger.debug(f"Risk filter blocked {asset.symbol}: {reasons}")
                self.signals_filtered_by_risk += 1
                continue
            
            # This is our best opportunity
            self.logger.info(f"✅ Best opportunity: {asset.symbol} - Score: {asset.score:.1f}, "
                           f"Signal: {asset.signal}, Regime: {asset.regime}")
            return asset
        
        self.logger.debug("No opportunities passed all filters")
        return None
    
    def manage_position(self, symbol: str, current_price: float, atr: float):
        """
        Manage open position
        
        Checks:
        - Stop loss / take profit hit
        - Trailing stop update
        - Time stop
        """
        try:
            # Update position metrics
            position_risk = self.risk_manager.update_position(symbol, current_price, atr)
            
            self.logger.debug(f"Position {symbol}: PnL ${position_risk.unrealized_pnl:.2f} "
                            f"({position_risk.unrealized_pnl_pct:.2f}%), "
                            f"Time: {position_risk.time_in_position:.1f}h")
            
            # Check exit conditions
            should_exit, reason = self.risk_manager.should_exit_position(symbol, current_price)
            
            if should_exit:
                self.logger.info(f"🚪 Exit signal for {symbol}: {reason}")
                self.close_position(symbol, current_price, reason)
            
        except Exception as e:
            self.logger.error(f"Error managing position {symbol}: {e}")
    
    def open_position(self, asset_score: AssetScore):
        """Open a new position"""
        try:
            symbol = asset_score.symbol
            signal_result: SignalResult = asset_score.metadata["signal_result"]
            
            # Get market data for position sizing
            df = self.fetch_market_data(symbol)
            if df is None:
                self.logger.error(f"Cannot open position - no data for {symbol}")
                return
            
            # Get CURRENT WEEX PRICE (not Binance historical close)
            weex_price = self.execution_engine.get_current_price(symbol)
            if weex_price <= 0:
                self.logger.error(f"Cannot get current WEEX price for {symbol}")
                return
            
            entry_price = weex_price
            
            # Log if there's a price difference between Binance and WEEX
            last = df.iloc[-1]
            binance_price = last['close']
            price_diff_pct = abs(weex_price - binance_price) / binance_price * 100
            if price_diff_pct > 0.5:
                self.logger.warning(f"⚠️ Price difference: Binance ${binance_price:.2f} vs WEEX ${weex_price:.2f} ({price_diff_pct:.2f}%)")
            
            atr = signal_result.metadata.get("atr_pct", 1.0) * entry_price / 100
            
            # Use signal's stop loss and take profit
            stop_loss = signal_result.stop_loss
            take_profit = signal_result.take_profit
            
            # Calculate position size
            size, margin, can_trade = self.risk_manager.calculate_position_size(
                entry_price, stop_loss, atr, symbol
            )
            
            if not can_trade or size <= 0:
                self.logger.warning(f"Cannot open position - insufficient capital or risk limits")
                return
            
            # Round size based on WEEX stepSize requirements for each symbol
            # Different symbols have different minimum size increments
            step_sizes = {
                "cmt_btcusdt": 0.001,   # BTC: 0.001 increments
                "cmt_ethusdt": 0.01,    # ETH: 0.01 increments  
                "cmt_solusdt": 0.1,     # SOL: 0.1 increments
                "cmt_dogeusdt": 1,      # DOGE: 1 increment
                "cmt_xrpusdt": 1,       # XRP: 1 increment
                "cmt_adausdt": 10,      # ADA: 10 increments
                "cmt_bnbusdt": 0.01,    # BNB: 0.01 increments
                "cmt_ltcusdt": 0.1      # LTC: 0.1 increments
            }
            
            step_size = step_sizes.get(symbol, 1)
            # Round down to nearest stepSize multiple
            size = (size // step_size) * step_size
            
            # Ensure minimum size after rounding
            if size < step_size:
                self.logger.warning(f"Position size {size} too small for {symbol} (stepSize: {step_size})")
                return
            
            self.logger.info(f"   Rounded size to {size:.4f} (stepSize: {step_size})")
            
            # Validate order safety
            direction = "LONG" if asset_score.signal == "LONG" else "SHORT"
            is_safe, warnings = self.execution_engine.validate_order_safety(
                symbol, "buy" if direction == "LONG" else "sell",
                size, entry_price, stop_loss
            )
            
            if not is_safe:
                self.logger.warning(f"Order safety check failed: {warnings}")
                return
            
            # Execute order
            self.logger.info(f"🎯 Opening {direction} position on {symbol}")
            self.logger.info(f"   Entry: ${entry_price:.2f}, Stop: ${stop_loss:.2f}, "
                           f"Target: ${take_profit:.2f}")
            self.logger.info(f"   Size: {size:.4f}, Margin: ${margin:.2f}, "
                           f"R:R = {signal_result.risk_reward:.1f}")
            
            order_result = self.execution_engine.execute_market_order(
                symbol=symbol,
                side="buy" if direction == "LONG" else "sell",
                size=size
            )
            
            if order_result.success:
                # Register position with risk manager
                self.risk_manager.open_position(
                    symbol=symbol,
                    direction=direction,
                    entry_price=order_result.filled_price,
                    size=order_result.filled_size,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    margin_used=margin,
                    atr=atr
                )
                
                self.trades_executed += 1
                self.logger.info(f"✅ Position opened successfully: {order_result.order_id}")
                
                # Upload AI log to WEEX (matching sentinel service format)
                try:
                    metadata = asset_score.metadata
                    
                    ai_log_input = {
                        "market_data": {
                            "symbol": symbol,
                            "price": float(order_result.filled_price),
                            "signal_strength": float(asset_score.confidence),
                            "regime": str(asset_score.regime),
                            "stop_loss": float(stop_loss),
                            "take_profit": float(take_profit)
                        },
                        "ml_prediction": {
                            "direction": str(metadata.get("ml_direction", "UNKNOWN")),
                            "confidence": int(metadata.get("ml_confidence", 0))
                        },
                        "sentiment": {
                            "label": str(metadata.get("sentiment_label", "NEUTRAL")),
                            "score": float(metadata.get("sentiment_score", 0.0)),
                            "source": str(metadata.get("sentiment_source", "N/A"))
                        },
                        "prompt": f"Analyze {symbol} for institutional trading opportunity"
                    }
                    
                    ai_log_output = {
                        "decision": direction,
                        "confidence": int(asset_score.confidence),
                        "reasoning": f"Signal {asset_score.signal} in {asset_score.regime} regime",
                        "ml_agrees": metadata.get("ml_direction") == asset_score.signal,
                        "gemini_approved": bool(metadata.get("gemini_approved", False)),
                        "strategy": "Institutional Multi-Asset"
                    }
                    
                    ml_dir = metadata.get("ml_direction", "UNKNOWN")
                    ml_conf = metadata.get("ml_confidence", 0)
                    gem_approved = "approved" if metadata.get("gemini_approved") else "consulted"
                    sent_label = metadata.get("sentiment_label", "NEUTRAL")
                    
                    explanation = f"AI analyzed {symbol} with signal strength {asset_score.confidence:.1f}, regime {asset_score.regime}. Decision: {direction} with {asset_score.confidence:.0f}% confidence. ML predicts {ml_dir} ({ml_conf}%), Gemini {gem_approved}, Sentiment: {sent_label}."
                    
                    self.client.upload_ai_log(
                        order_id=order_result.order_id,
                        stage="Decision Making",
                        model="Gemini-2.0-Flash-Thinking",
                        input_data=ai_log_input,
                        output_data=ai_log_output,
                        explanation=explanation
                    )
                    self.logger.info(f"   AI Log uploaded for order {order_result.order_id}")
                except Exception as ai_log_err:
                    self.logger.error(f"   AI Log upload failed: {ai_log_err}")
            else:
                self.logger.error(f"❌ Order execution failed: {order_result.error_message}")
            
        except Exception as e:
            self.logger.error(f"Error opening position: {e}")
    
    def close_position(self, symbol: str, exit_price: float, reason: str):
        """Close an open position"""
        try:
            if symbol not in self.risk_manager.positions:
                self.logger.warning(f"No open position for {symbol}")
                return
            
            pos = self.risk_manager.positions[symbol]
            direction = pos["direction"]
            size = pos["size"]
            
            self.logger.info(f"🔚 Closing {direction} position on {symbol}: {reason}")
            
            # Execute close order
            order_result = self.execution_engine.execute_market_order(
                symbol=symbol,
                side="sell" if direction == "LONG" else "buy",
                size=size,
                reduce_only=True
            )
            
            if order_result.success:
                # Close position in risk manager
                trade_record = self.risk_manager.close_position(
                    symbol, order_result.filled_price, reason
                )
                
                self.logger.info(f"✅ Position closed: PnL ${trade_record['realized_pnl']:.2f} "
                               f"({trade_record['realized_pnl_pct']:.2f}%), "
                               f"Hold time: {trade_record['hold_time_hours']:.1f}h")
                
                # Upload AI log for position close (matching sentinel format)
                try:
                    ai_log_input = {
                        "market_data": {
                            "symbol": symbol,
                            "price": float(order_result.filled_price),
                            "position_direction": str(direction),
                            "close_reason": str(reason)
                        },
                        "prompt": f"Close {direction} position on {symbol} - {reason}"
                    }
                    
                    ai_log_output = {
                        "decision": "CLOSE",
                        "confidence": 100,
                        "reasoning": f"Position closed: {reason}",
                        "pnl": float(trade_record['realized_pnl']),
                        "pnl_pct": float(trade_record['realized_pnl_pct']),
                        "strategy": "Institutional Risk Manager"
                    }
                    
                    explanation = f"AI closed {direction} position on {symbol}. Reason: {reason}. PnL: ${trade_record['realized_pnl']:.2f} ({trade_record['realized_pnl_pct']:.2f}%), Hold time: {trade_record['hold_time_hours']:.1f}h."
                    
                    self.client.upload_ai_log(
                        order_id=order_result.order_id,
                        stage="Strategy Execution",
                        model="Gemini-2.0-Flash-Thinking",
                        input_data=ai_log_input,
                        output_data=ai_log_output,
                        explanation=explanation
                    )
                    self.logger.info(f"   AI Log uploaded for order {order_result.order_id}")
                except Exception as ai_log_err:
                    self.logger.error(f"   AI Log upload failed: {ai_log_err}")
            else:
                self.logger.error(f"❌ Close order failed: {order_result.error_message}")
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
    
    def run_cycle(self):
        """
        Run one trading cycle
        
        Process:
        1. Update equity from account balance
        2. Manage open positions
        3. If flat, search for new opportunity
        4. Execute highest probability setup
        """
        self.cycle_count += 1
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"CYCLE #{self.cycle_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"{'='*60}")
        
        try:
            # Get portfolio status
            portfolio_risk = self.risk_manager.get_portfolio_risk()
            
            self.logger.info(f"💰 Equity: ${portfolio_risk.total_equity:,.2f} | "
                           f"Daily PnL: ${portfolio_risk.daily_pnl:,.2f} ({portfolio_risk.daily_pnl_pct:.2f}%) | "
                           f"Exposure: {portfolio_risk.exposure_pct:.1f}% | "
                           f"Positions: {portfolio_risk.open_positions}/{portfolio_risk.max_positions_allowed}")
            
            if portfolio_risk.risk_alerts:
                for alert in portfolio_risk.risk_alerts:
                    self.logger.warning(f"⚠️ {alert}")
            
            # Manage open positions
            if portfolio_risk.open_positions > 0:
                for symbol in list(self.risk_manager.positions.keys()):
                    # Get current WEEX price for position management
                    current_price = self.execution_engine.get_current_price(symbol)
                    if current_price <= 0:
                        self.logger.warning(f"Cannot get WEEX price for {symbol}, skipping position management")
                        continue
                    
                    df = self.fetch_market_data(symbol)
                    if df is not None:
                        last = df.iloc[-1]
                        
                        # Calculate ATR
                        if 'atr' in df.columns:
                            atr = df['atr'].iloc[-1]
                        else:
                            atr = last['close'] * 0.015  # Estimate 1.5% ATR
                        
                        self.manage_position(symbol, current_price, atr)
            
            # If flat and can trade, look for new opportunity
            if portfolio_risk.open_positions == 0 and portfolio_risk.can_trade:
                self.logger.info("🔍 Scanning for opportunities (using WEEX live prices)...")
                
                best_opportunity = self.select_best_opportunity()
                
                if best_opportunity:
                    self.open_position(best_opportunity)
                else:
                    self.logger.info("⏸️ No valid opportunities - staying flat")
            
            # Update statistics
            self.logger.info(f"\n📊 Session Stats:")
            self.logger.info(f"   Trades executed: {self.trades_executed}")
            self.logger.info(f"   Signals generated: {self.total_signals_generated}")
            self.logger.info(f"   Filtered by regime: {self.signals_filtered_by_regime}")
            self.logger.info(f"   Filtered by risk: {self.signals_filtered_by_risk}")
            
            exec_stats = self.execution_engine.get_execution_statistics()
            self.logger.info(f"   Execution success rate: {exec_stats['success_rate']:.1f}%")
            
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}", exc_info=True)
        
        self.last_cycle_time = datetime.now()
    
    def run_continuous(self):
        """Run continuous trading loop"""
        self.logger.info("🚀 Starting continuous trading loop...")
        self.logger.info(f"   Cycle interval: {self.CYCLE_INTERVAL}s")
        
        try:
            while True:
                self.run_cycle()
                
                # Sleep until next cycle
                time.sleep(self.CYCLE_INTERVAL)
                
        except KeyboardInterrupt:
            self.logger.info("\n🛑 Shutting down gracefully...")
            
            # Close any open positions
            for symbol in list(self.risk_manager.positions.keys()):
                current_price = self.execution_engine.get_current_price(symbol)
                if current_price > 0:
                    self.close_position(symbol, current_price, "System shutdown")
                else:
                    self.logger.error(f"Cannot get WEEX price to close {symbol}")
            
            self.logger.info("✅ Shutdown complete")
