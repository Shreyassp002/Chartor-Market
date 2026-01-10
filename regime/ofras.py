"""
OFRAS - Opportunistic Funding Rate Adaptive Strategy
Regime detection system that switches between Wave Rider and Liquidation Hunter modes
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Literal
from dataclasses import dataclass
from enum import Enum


class RegimeType(Enum):
    """Market regime types"""
    WAVE_RIDER = "WAVE_RIDER"  # Trend following mode
    LIQUIDATION_HUNTER = "LIQUIDATION_HUNTER"  # Mean reversion mode
    NEUTRAL = "NEUTRAL"  # Choppy/unclear market


@dataclass
class RegimeState:
    """Current market regime"""
    regime: RegimeType
    confidence: float  # 0-100
    trend_score: float
    volatility_score: float
    funding_pressure: float
    liquidation_pressure: float
    metadata: Dict


class OFRASRegimeDetector:
    """
    OFRAS Regime Overlay System
    
    Analyzes:
    1. Price trend structure
    2. Volatility structure
    3. Funding rate behavior
    4. Open Interest dynamics
    5. Liquidation pressure
    
    Switches between:
    - Wave Rider: Strong trending markets
    - Liquidation Hunter: Mean-reverting, high liquidation zones
    """
    
    def __init__(self,
                 trend_lookback: int = 50,
                 vol_lookback: int = 20,
                 funding_threshold: float = 0.01):
        self.trend_lookback = trend_lookback
        self.vol_lookback = vol_lookback
        self.funding_threshold = funding_threshold
    
    def calculate_trend_structure(self, df: pd.DataFrame) -> Dict[str, float]:
        import pandas_ta as ta
        """
        Calculate trend strength and structure
        
        Uses:
        - Price position relative to EMAs
        - Slope of moving averages
        - Higher highs / lower lows pattern
        """
        if len(df) < self.trend_lookback:
            return {"score": 0, "direction": 0, "strength": 0}
        
        # Calculate EMAs if not present
        if 'ema_21' not in df.columns:
            df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        if 'ema_50' not in df.columns:
            df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        last = df.iloc[-1]
        
        # EMA hierarchy
        
        # Trend direction
        bullish_alignment = (last['close'] > last['ema_21'] > last['ema_50'])
        bearish_alignment = (last['close'] < last['ema_21'] < last['ema_50'])
        
        direction = 1.0 if bullish_alignment else (-1.0 if bearish_alignment else 0.0)
        
        # Trend strength via slope consistency
        ema_21_slope = df['ema_21'].iloc[-10:].pct_change().mean() * 100
        slope_strength = abs(ema_21_slope) * 10  # Scale to 0-100
        
        # Higher highs / lower lows pattern
        recent_highs = df['high'].iloc[-20:].values
        recent_lows = df['low'].iloc[-20:].values
        
        hh_count = sum(recent_highs[i] > recent_highs[i-5] for i in range(5, 20))
        ll_count = sum(recent_lows[i] < recent_lows[i-5] for i in range(5, 20))
        
        pattern_score = (hh_count / 15 * 100) if direction > 0 else (ll_count / 15 * 100)
        
        # Composite trend score
        trend_score = (slope_strength * 0.5 + pattern_score * 0.5)
        trend_score = min(trend_score, 100)
        
        return {
            "score": trend_score,
            "direction": direction,
            "strength": slope_strength,
            "pattern_strength": pattern_score
        }
    
    def calculate_volatility_structure(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze volatility regime
        
        Formula:
        - Realized Vol = σ(returns) * sqrt(N)
        - Vol Regime = Current Vol / Historical Vol
        """
        if len(df) < self.vol_lookback + 20:
            return {"regime": 0, "current_vol": 0, "vol_percentile": 0}
        
        # Calculate returns
        if 'returns' not in df.columns:
            df['returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Current volatility (annualized)
        current_vol = df['returns'].iloc[-self.vol_lookback:].std() * np.sqrt(252) * 100
        
        # Historical volatility distribution
        rolling_vol = df['returns'].rolling(window=self.vol_lookback).std() * np.sqrt(252) * 100
        vol_percentile = (rolling_vol < current_vol).sum() / len(rolling_vol)
        
        # Volatility regime
        # High vol (>70th percentile) = mean reversion favorable
        # Low vol (<30th percentile) = trend following favorable
        if vol_percentile > 0.7:
            regime_score = 100 * (vol_percentile - 0.7) / 0.3  # 0-100 for high vol
        elif vol_percentile < 0.3:
            regime_score = -100 * (0.3 - vol_percentile) / 0.3  # -100 to 0 for low vol
        else:
            regime_score = 0  # Neutral vol
        
        return {
            "regime": regime_score,  # -100 to 100 (negative = trend, positive = mean reversion)
            "current_vol": current_vol,
            "vol_percentile": vol_percentile
        }
    
    def calculate_funding_pressure(self,
                                   funding_rate: Optional[float],
                                   oi_change: Optional[float],
                                   df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate funding rate pressure
        
        Formula:
        FP = sign(funding) × |funding| × ΔOI
        
        High positive funding + increasing OI = overleveraged longs → mean reversion
        High negative funding = overleveraged shorts → potential squeeze
        """
        if funding_rate is None:
            return {"pressure": 0, "signal": 0, "magnitude": 0}
        
        # Funding pressure magnitude
        funding_magnitude = abs(funding_rate) * 10000  # Convert to basis points
        
        # Signal direction
        # Positive funding = longs paying shorts (too many longs)
        # Negative funding = shorts paying longs (too many shorts)
        funding_signal = -1.0 if funding_rate > 0 else 1.0  # Contrarian signal
        
        # OI confirmation
        oi_multiplier = 1.0
        if oi_change is not None:
            # Rising OI + extreme funding = building pressure
            oi_multiplier = 1.0 + abs(oi_change)
        
        # Composite pressure
        pressure_score = funding_magnitude * oi_multiplier
        pressure_score = min(pressure_score, 100)
        
        # Extreme funding = mean reversion opportunity
        is_extreme = abs(funding_rate) > self.funding_threshold
        
        return {
            "pressure": pressure_score,
            "signal": funding_signal if is_extreme else 0,
            "magnitude": funding_magnitude,
            "is_extreme": is_extreme
        }
    
    def calculate_liquidation_pressure(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Detect liquidation zones and pressure
        
        Uses:
        - Sharp price moves with high volume
        - RSI extremes
        - Volatility spikes
        """
        if len(df) < 10:
            return {"pressure": 0, "zone": None, "score": 0}
        
        last = df.iloc[-1]
        
        # Sharp price movement
        recent_returns = df['returns'].iloc[-5:] if 'returns' in df.columns else df['close'].pct_change().iloc[-5:]
        max_move = abs(recent_returns).max()
        sharp_move = max_move > 0.03  # 3% move
        
        # RSI extremes
        if 'rsi' in df.columns:
            rsi = last['rsi']
            rsi_oversold = rsi < 30
            rsi_overbought = rsi > 70
        else:
            rsi_oversold = False
            rsi_overbought = False
        
        # Volume spike
        if 'volume_zscore' in df.columns:
            volume_spike = last['volume_zscore'] > 2.0
        else:
            volume_spike = False
        
        # Volatility spike
        if 'atr_pct' in df.columns:
            vol_spike = last['atr_pct'] > df['atr_pct'].iloc[-20:].mean() * 1.5
        else:
            vol_spike = False
        
        # Liquidation pressure score
        pressure_score = 0
        liquidation_zone = None
        
        if sharp_move and volume_spike:
            pressure_score += 40
        
        if rsi_oversold and sharp_move:
            liquidation_zone = "LONG"  # Longs liquidated, reversal opportunity
            pressure_score += 30
        elif rsi_overbought and sharp_move:
            liquidation_zone = "SHORT"  # Shorts liquidated, reversal opportunity
            pressure_score += 30
        
        if vol_spike:
            pressure_score += 20
        
        pressure_score = min(pressure_score, 100)
        
        return {
            "pressure": pressure_score,
            "zone": liquidation_zone,
            "score": pressure_score
        }
    
    def detect_regime(self,
                     df: pd.DataFrame,
                     funding_rate: Optional[float] = None,
                     oi_change: Optional[float] = None) -> RegimeState:
        """
        Main regime detection logic
        
        Decision Matrix:
        1. Strong trend + low vol + normal funding → WAVE_RIDER
        2. Weak trend + high vol + extreme funding → LIQUIDATION_HUNTER
        3. Mixed signals → NEUTRAL
        """
        # Calculate all components
        trend_data = self.calculate_trend_structure(df)
        vol_data = self.calculate_volatility_structure(df)
        funding_data = self.calculate_funding_pressure(funding_rate, oi_change, df)
        liq_data = self.calculate_liquidation_pressure(df)
        
        # Scoring system
        trend_score = trend_data["score"]
        vol_regime = vol_data["regime"]
        funding_pressure = funding_data["pressure"]
        liq_pressure = liq_data["pressure"]
        
        # Decision logic
        regime = RegimeType.NEUTRAL
        confidence = 0.0
        
        # WAVE RIDER conditions:
        # - Strong trend (>50) - Relaxed from 60
        # - Low/normal volatility (regime < 10) - Relaxed
        # - Normal funding (<60) - Relaxed from 50
        wave_rider_score = 0
        if trend_score > 50 and vol_regime < 10 and funding_pressure < 60:
            wave_rider_score = (trend_score * 0.5 + 
                              abs(vol_regime) * 0.3 + 
                              (100 - funding_pressure) * 0.2)
        
        # LIQUIDATION HUNTER conditions:
        # - Weak trend (<40)
        # - High volatility (regime > 0)
        # - Extreme funding (>70) OR high liquidation pressure (>60)
        hunter_score = 0
        if trend_score < 40 and (funding_pressure > 70 or liq_pressure > 60):
            hunter_score = (funding_pressure * 0.4 + 
                          liq_pressure * 0.4 + 
                          vol_regime * 0.2)
        
        # Select regime
        # Log scores for debugging (can be removed later)
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Regime scoring - Wave Rider: {wave_rider_score:.0f}, Hunter: {hunter_score:.0f}, "
                    f"Trend: {trend_score:.0f}, Vol: {vol_regime:.0f}, Funding: {funding_pressure:.0f}, Liq: {liq_pressure:.0f}")
        
        if wave_rider_score > 50 and wave_rider_score > hunter_score:  # Relaxed from 60
            regime = RegimeType.WAVE_RIDER
            confidence = wave_rider_score
        elif hunter_score > 50 and hunter_score > wave_rider_score:  # Relaxed from 60
            regime = RegimeType.LIQUIDATION_HUNTER
            confidence = hunter_score
        else:
            regime = RegimeType.NEUTRAL
            confidence = 50.0
        
        # Metadata
        metadata = {
            "trend_direction": trend_data["direction"],
            "trend_strength": trend_data["strength"],
            "vol_percentile": vol_data["vol_percentile"],
            "current_vol": vol_data["current_vol"],
            "funding_extreme": funding_data.get("is_extreme", False),
            "liquidation_zone": liq_data["zone"],
            "wave_rider_score": wave_rider_score,
            "hunter_score": hunter_score
        }
        
        return RegimeState(
            regime=regime,
            confidence=min(confidence, 100),
            trend_score=trend_score,
            volatility_score=vol_regime,
            funding_pressure=funding_pressure,
            liquidation_pressure=liq_pressure,
            metadata=metadata
        )
    
    def get_strategy_filter(self, regime_state: RegimeState) -> Dict[str, bool]:
        """
        Get strategy filters based on regime
        
        Returns which strategies to enable/disable
        """
        regime = regime_state.regime
        
        filters = {
            "trend_following": False,
            "mean_reversion": False,
            "breakout": False,
            "liquidation_snapback": False
        }
        
        if regime == RegimeType.WAVE_RIDER:
            filters["trend_following"] = True
            filters["breakout"] = True
        elif regime == RegimeType.LIQUIDATION_HUNTER:
            filters["mean_reversion"] = True
            filters["liquidation_snapback"] = True
        else:  # NEUTRAL
            # In neutral markets, allow both breakouts and trend_following
            # (more permissive to allow trading in less extreme conditions)
            filters["breakout"] = True
            filters["trend_following"] = True
        
        return filters
