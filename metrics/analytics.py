"""
Advanced Metrics and Analytics
Performance tracking, Monte Carlo simulation, tail risk analysis
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy import stats


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Returns
    total_return_pct: float
    cagr: float
    volatility: float
    
    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    omega_ratio: float
    
    # Drawdown metrics
    max_drawdown_pct: float
    avg_drawdown_pct: float
    drawdown_duration_days: int
    
    # Trade metrics
    win_rate: float
    profit_factor: float
    expectancy: float
    avg_r_multiple: float
    
    # Tail risk
    var_95: float  
    cvar_95: float  
    tail_ratio: float
    
    # Consistency
    monthly_returns: List[float]
    best_month: float
    worst_month: float
    positive_months_pct: float


class MetricsCalculator:
    """
    Advanced metrics calculator for trading systems
    """
    
    @staticmethod
    def calculate_sharpe_ratio(returns: np.ndarray, 
                              risk_free_rate: float = 0.0,
                              periods_per_year: int = 252) -> float:
        """
        Calculate Sharpe Ratio
        
        Formula: SR = (E[R] - Rf) / σ(R) × √periods
        """
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / periods_per_year)
        sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(periods_per_year)
        
        return sharpe
    
    @staticmethod
    def calculate_sortino_ratio(returns: np.ndarray,
                               risk_free_rate: float = 0.0,
                               periods_per_year: int = 252) -> float:
        """
        Calculate Sortino Ratio (only penalizes downside volatility)
        
        Formula: SR = (E[R] - Rf) / σ_downside × √periods
        """
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / periods_per_year)
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        sortino = (excess_returns.mean() / downside_returns.std()) * np.sqrt(periods_per_year)
        
        return sortino
    
    @staticmethod
    def calculate_calmar_ratio(returns: np.ndarray,
                              max_drawdown: float,
                              periods_per_year: int = 252) -> float:
        """
        Calculate Calmar Ratio
        
        Formula: CR = Annualized Return / Max Drawdown
        """
        if max_drawdown == 0:
            return 0.0
        
        annualized_return = (1 + returns.mean()) ** periods_per_year - 1
        calmar = annualized_return / abs(max_drawdown)
        
        return calmar
    
    @staticmethod
    def calculate_omega_ratio(returns: np.ndarray,
                             threshold: float = 0.0) -> float:
        """
        Calculate Omega Ratio
        
        Formula: Ω = ∑(returns > threshold) / ∑(returns < threshold)
        """
        gains = returns[returns > threshold]
        losses = returns[returns < threshold]
        
        if len(losses) == 0 or abs(losses.sum()) == 0:
            return 0.0
        
        omega = abs(gains.sum()) / abs(losses.sum())
        
        return omega
    
    @staticmethod
    def calculate_drawdown_series(equity_curve: np.ndarray) -> np.ndarray:
        """
        Calculate drawdown series
        
        Formula: DD = (Peak - Current) / Peak
        """
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (running_max - equity_curve) / running_max
        
        return drawdown
    
    @staticmethod
    def calculate_max_drawdown(equity_curve: np.ndarray) -> Tuple[float, int, int]:
        """
        Calculate maximum drawdown and duration
        
        Returns: (max_dd, start_idx, end_idx)
        """
        drawdown = MetricsCalculator.calculate_drawdown_series(equity_curve)
        max_dd = drawdown.max()
        
        # Find drawdown period
        max_dd_idx = drawdown.argmax()
        
        # Find start (peak before drawdown)
        start_idx = 0
        for i in range(max_dd_idx, -1, -1):
            if drawdown[i] == 0:
                start_idx = i
                break
        
        # Find end (recovery)
        end_idx = max_dd_idx
        for i in range(max_dd_idx, len(drawdown)):
            if drawdown[i] == 0:
                end_idx = i
                break
        
        return max_dd, start_idx, end_idx
    
    @staticmethod
    def calculate_var(returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk
        
        VaR_α = -Percentile(returns, 1-α)
        """
        if len(returns) == 0:
            return 0.0
        
        var = -np.percentile(returns, (1 - confidence) * 100)
        
        return var
    
    @staticmethod
    def calculate_cvar(returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (Expected Shortfall)
        
        CVaR_α = E[R | R < -VaR_α]
        """
        if len(returns) == 0:
            return 0.0
        
        var = MetricsCalculator.calculate_var(returns, confidence)
        
        # Average of returns worse than VaR
        tail_returns = returns[returns < -var]
        
        if len(tail_returns) == 0:
            return var
        
        cvar = -tail_returns.mean()
        
        return cvar
    
    @staticmethod
    def calculate_tail_ratio(returns: np.ndarray) -> float:
        """
        Calculate Tail Ratio
        
        TR = |95th percentile| / |5th percentile|
        
        TR > 1 means right tail (gains) is larger than left tail (losses)
        """
        if len(returns) == 0:
            return 0.0
        
        right_tail = np.percentile(returns, 95)
        left_tail = np.percentile(returns, 5)
        
        if left_tail == 0:
            return 0.0
        
        tail_ratio = abs(right_tail) / abs(left_tail)
        
        return tail_ratio
    
    @staticmethod
    def calculate_cagr(equity_curve: np.ndarray, 
                      periods_per_year: int = 252) -> float:
        """
        Calculate Compound Annual Growth Rate
        
        CAGR = (Ending Value / Beginning Value)^(1/years) - 1
        """
        if len(equity_curve) < 2:
            return 0.0
        
        starting_value = equity_curve[0]
        ending_value = equity_curve[-1]
        years = len(equity_curve) / periods_per_year
        
        if starting_value <= 0 or years <= 0:
            return 0.0
        
        cagr = (ending_value / starting_value) ** (1 / years) - 1
        
        return cagr
    
    @staticmethod
    def calculate_comprehensive_metrics(equity_curve: List[float],
                                       returns: List[float],
                                       trades: List[Dict]) -> PerformanceMetrics:
        """
        Calculate all performance metrics
        """
        equity_arr = np.array(equity_curve)
        returns_arr = np.array(returns)
        
        if len(equity_arr) == 0 or len(returns_arr) == 0:
            return PerformanceMetrics(
                total_return_pct=0.0,
                cagr=0.0,
                volatility=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                omega_ratio=0.0,
                max_drawdown_pct=0.0,
                avg_drawdown_pct=0.0,
                drawdown_duration_days=0,
                win_rate=0.0,
                profit_factor=0.0,
                expectancy=0.0,
                avg_r_multiple=0.0,
                var_95=0.0,
                cvar_95=0.0,
                tail_ratio=0.0,
                monthly_returns=[],
                best_month=0.0,
                worst_month=0.0,
                positive_months_pct=0.0
            )
        
        # Basic returns
        total_return_pct = ((equity_arr[-1] - equity_arr[0]) / equity_arr[0]) * 100
        cagr = MetricsCalculator.calculate_cagr(equity_arr) * 100
        volatility = returns_arr.std() * np.sqrt(252) * 100
        
        # Risk-adjusted returns
        sharpe = MetricsCalculator.calculate_sharpe_ratio(returns_arr)
        sortino = MetricsCalculator.calculate_sortino_ratio(returns_arr)
        
        # Drawdown
        max_dd, dd_start, dd_end = MetricsCalculator.calculate_max_drawdown(equity_arr)
        max_dd_pct = max_dd * 100
        
        drawdown_series = MetricsCalculator.calculate_drawdown_series(equity_arr)
        avg_dd_pct = drawdown_series[drawdown_series > 0].mean() * 100 if len(drawdown_series[drawdown_series > 0]) > 0 else 0
        
        dd_duration = dd_end - dd_start
        
        calmar = MetricsCalculator.calculate_calmar_ratio(returns_arr, max_dd)
        omega = MetricsCalculator.calculate_omega_ratio(returns_arr)
        
        # Trade metrics
        if len(trades) > 0:
            winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
            losing_trades = [t for t in trades if t.get("pnl", 0) <= 0]
            
            win_rate = (len(winning_trades) / len(trades)) * 100
            
            gross_profit = sum(t.get("pnl", 0) for t in winning_trades)
            gross_loss = abs(sum(t.get("pnl", 0) for t in losing_trades))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            avg_win = np.mean([t.get("pnl", 0) for t in winning_trades]) if len(winning_trades) > 0 else 0
            avg_loss = np.mean([t.get("pnl", 0) for t in losing_trades]) if len(losing_trades) > 0 else 0
            expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
            
            avg_r = np.mean([t.get("r_multiple", 0) for t in trades])
        else:
            win_rate = 0.0
            profit_factor = 0.0
            expectancy = 0.0
            avg_r = 0.0
        
        # Tail risk
        var_95 = MetricsCalculator.calculate_var(returns_arr, 0.95) * 100
        cvar_95 = MetricsCalculator.calculate_cvar(returns_arr, 0.95) * 100
        tail_ratio = MetricsCalculator.calculate_tail_ratio(returns_arr)
        
        # Monthly returns (approximation)
        equity_df = pd.DataFrame({"equity": equity_arr})
        equity_df['period'] = equity_df.index // 21  
        monthly_equity = equity_df.groupby('period')['equity'].last()
        monthly_returns = monthly_equity.pct_change().dropna().values * 100
        
        if len(monthly_returns) > 0:
            best_month = monthly_returns.max()
            worst_month = monthly_returns.min()
            positive_months_pct = (monthly_returns > 0).sum() / len(monthly_returns) * 100
        else:
            best_month = 0.0
            worst_month = 0.0
            positive_months_pct = 0.0
        
        return PerformanceMetrics(
            total_return_pct=total_return_pct,
            cagr=cagr,
            volatility=volatility,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            omega_ratio=omega,
            max_drawdown_pct=max_dd_pct,
            avg_drawdown_pct=avg_dd_pct,
            drawdown_duration_days=dd_duration,
            win_rate=win_rate,
            profit_factor=profit_factor,
            expectancy=expectancy,
            avg_r_multiple=avg_r,
            var_95=var_95,
            cvar_95=cvar_95,
            tail_ratio=tail_ratio,
            monthly_returns=monthly_returns.tolist(),
            best_month=best_month,
            worst_month=worst_month,
            positive_months_pct=positive_months_pct
        )


class MonteCarloSimulator:
    """
    Monte Carlo simulation for stress testing
    """
    
    @staticmethod
    def run_simulation(trades: List[Dict],
                      initial_capital: float,
                      num_simulations: int = 1000) -> Dict:
        """
        Run Monte Carlo simulation by randomizing trade sequence
        
        Returns distribution of outcomes
        """
        if len(trades) == 0:
            return {
                "median_return": 0.0,
                "mean_return": 0.0,
                "best_case": 0.0,
                "worst_case": 0.0,
                "prob_profit": 0.0,
                "outcomes": []
            }
        
        outcomes = []
        
        for _ in range(num_simulations):
            # Randomize trade sequence
            shuffled_trades = np.random.choice(trades, size=len(trades), replace=True)
            
            # Calculate equity curve
            equity = initial_capital
            for trade in shuffled_trades:
                equity += trade.get("pnl", 0)
            
            final_return = ((equity - initial_capital) / initial_capital) * 100
            outcomes.append(final_return)
        
        outcomes = np.array(outcomes)
        
        return {
            "median_return": np.median(outcomes),
            "mean_return": np.mean(outcomes),
            "std_return": np.std(outcomes),
            "best_case": np.percentile(outcomes, 95),
            "worst_case": np.percentile(outcomes, 5),
            "prob_profit": (outcomes > 0).sum() / len(outcomes) * 100,
            "outcomes": outcomes.tolist()
        }
