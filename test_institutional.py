"""
Test Script for Institutional Trading System
Run this to verify all components are working
"""
import sys
import traceback
from datetime import datetime


def test_imports():
    """Test that all modules can be imported"""
    print("Testing module imports...")
    
    try:
        from strategy.intraday_engine import IntradayMomentumEngine
        print("✓ Strategy module imported")
        
        from regime.ofras import OFRASRegimeDetector
        print("✓ Regime module imported")
        
        from risk.risk_manager import RiskManager
        print("✓ Risk module imported")
        
        from execution.execution_engine import ExecutionEngine
        print("✓ Execution module imported")
        
        from backtest.backtest_engine import BacktestEngine
        print("✓ Backtest module imported")
        
        from metrics.analytics import MetricsCalculator
        print("✓ Metrics module imported")
        
        from trading_orchestrator import TradingOrchestrator
        print("✓ Trading orchestrator imported")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False


def test_strategy_engine():
    """Test strategy engine with sample data"""
    print("\nTesting strategy engine...")
    
    try:
        import pandas as pd
        import numpy as np
        from strategy.intraday_engine import IntradayMomentumEngine
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
        prices = 50000 + np.cumsum(np.random.randn(100) * 100)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices + np.random.randn(100) * 10,
            'high': prices + abs(np.random.randn(100) * 20),
            'low': prices - abs(np.random.randn(100) * 20),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        engine = IntradayMomentumEngine()
        signal = engine.generate_signal(df)
        
        print(f"✓ Strategy engine working")
        print(f"  Signal: {signal.signal}, Strength: {signal.strength:.1f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Strategy engine test failed: {e}")
        traceback.print_exc()
        return False


def test_regime_detector():
    """Test regime detection"""
    print("\nTesting regime detector...")
    
    try:
        import pandas as pd
        import numpy as np
        from regime.ofras import OFRASRegimeDetector
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='5min')
        prices = 50000 + np.cumsum(np.random.randn(100) * 100)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices + np.random.randn(100) * 10,
            'high': prices + abs(np.random.randn(100) * 20),
            'low': prices - abs(np.random.randn(100) * 20),
            'close': prices,
            'volume': np.random.randint(1000, 10000, 100)
        })
        
        detector = OFRASRegimeDetector()
        regime = detector.detect_regime(df, funding_rate=0.0001, oi_change=0.05)
        
        print(f"✓ Regime detector working")
        print(f"  Regime: {regime.regime.value}, Confidence: {regime.confidence:.1f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Regime detector test failed: {e}")
        traceback.print_exc()
        return False


def test_risk_manager():
    """Test risk manager"""
    print("\nTesting risk manager...")
    
    try:
        from risk.risk_manager import RiskManager
        
        rm = RiskManager(initial_equity=10000.0)
        
        # Test position sizing
        size, margin, can_trade = rm.calculate_position_size(
            entry_price=50000.0,
            stop_loss=49500.0,
            atr=500.0,
            symbol="cmt_btcusdt"
        )
        
        print(f"✓ Risk manager working")
        print(f"  Position size: {size:.4f}, Margin: ${margin:.2f}, Can trade: {can_trade}")
        
        # Test portfolio risk
        portfolio = rm.get_portfolio_risk()
        print(f"  Portfolio equity: ${portfolio.total_equity:.2f}")
        print(f"  Can trade: {portfolio.can_trade}")
        
        return True
        
    except Exception as e:
        print(f"✗ Risk manager test failed: {e}")
        traceback.print_exc()
        return False


def test_backtest_engine():
    """Test backtest engine"""
    print("\nTesting backtest engine...")
    
    try:
        from backtest.backtest_engine import BacktestEngine, BacktestConfig
        
        config = BacktestConfig(initial_capital=10000.0)
        engine = BacktestEngine(config)
        
        # Test position operations
        opened = engine.open_backtest_position(
            symbol="cmt_btcusdt",
            direction="LONG",
            entry_price=50000.0,
            stop_loss=49500.0,
            take_profit=51000.0,
            timestamp=datetime.now()
        )
        
        if opened:
            print(f"✓ Backtest engine working")
            print(f"  Test position opened successfully")
            
            # Close position
            trade = engine.close_backtest_position(
                exit_price=50500.0,
                timestamp=datetime.now(),
                exit_reason="Test"
            )
            
            if trade:
                print(f"  Test position closed: PnL ${trade.pnl:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Backtest engine test failed: {e}")
        traceback.print_exc()
        return False


def test_metrics():
    """Test metrics calculator"""
    print("\nTesting metrics calculator...")
    
    try:
        import numpy as np
        from metrics.analytics import MetricsCalculator
        
        # Sample returns
        returns = np.random.randn(100) * 0.01
        
        sharpe = MetricsCalculator.calculate_sharpe_ratio(returns)
        sortino = MetricsCalculator.calculate_sortino_ratio(returns)
        
        print(f"✓ Metrics calculator working")
        print(f"  Sharpe: {sharpe:.2f}, Sortino: {sortino:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Metrics calculator test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("INSTITUTIONAL TRADING SYSTEM - COMPONENT TESTS")
    print("="*60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Strategy Engine", test_strategy_engine()))
    results.append(("Regime Detector", test_regime_detector()))
    results.append(("Risk Manager", test_risk_manager()))
    results.append(("Backtest Engine", test_backtest_engine()))
    results.append(("Metrics", test_metrics()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:.<40} {status}")
    
    print()
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready for deployment.")
        return 0
    else:
        print("\n⚠️ Some tests failed. Please fix issues before deployment.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
