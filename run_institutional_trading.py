"""
Example: Start Institutional Trading System
Simple script to demonstrate system usage
"""
import logging
import sys
from datetime import datetime


def main(skip_confirmation: bool = False):
    """
    Start the institutional trading system
    
    Args:
        skip_confirmation: Skip interactive confirmation (for API/background mode)
    """
    
    # Only setup logging if it hasn't been configured yet (e.g., running standalone)
    if not logging.getLogger().hasHandlers():
        import io
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')),
                logging.FileHandler(f'trading_{datetime.now().strftime("%Y%m%d")}.log', encoding='utf-8')
            ]
        )
    
    logger = logging.getLogger("InstitutionalTrading")
    
    logger.info("="*60)
    logger.info("INSTITUTIONAL TRADING SYSTEM")
    logger.info("="*60)
    
    try:
        # Import required components
        logger.info("Importing components...")
        from trading_orchestrator import TradingOrchestrator
        from core.weex_api import WeexClient
        
        # Initialize WEEX client
        logger.info("Initializing WEEX client...")
        client = WeexClient()
        
        # Fetch real account balance
        logger.info("Fetching account balance...")
        try:
            balance_data = client.get_balance()
            logger.info(f"Balance API response: {balance_data}")
            
            if balance_data and balance_data.get('code') == '00000' and 'data' in balance_data:
                # Get USDT balance (available_balance is what we can use for new positions)
                usdt_balance = next((item for item in balance_data['data'] if item.get('coin_name') == 'USDT'), None)
                if usdt_balance:
                    available_balance = float(usdt_balance.get('available_balance', 0))
                    if available_balance > 0:
                        INITIAL_EQUITY = available_balance
                        logger.info(f"✅ Real account balance: ${INITIAL_EQUITY:.2f} USDT")
                    else:
                        logger.warning("Available balance is 0, using fallback")
                        INITIAL_EQUITY = 1000.0
                else:
                    logger.warning("USDT balance not found in response, using fallback")
                    INITIAL_EQUITY = 1000.0  # Conservative fallback
            else:
                logger.warning(f"Could not fetch balance (code: {balance_data.get('code') if balance_data else 'None'}), using fallback")
                INITIAL_EQUITY = 1000.0  # Conservative fallback
        except Exception as balance_err:
            logger.error(f"Balance fetch error: {balance_err}", exc_info=True)
            INITIAL_EQUITY = 1000.0  # Conservative fallback
        
        logger.info(f"Configuration:")
        logger.info(f"  Initial Equity: ${INITIAL_EQUITY:,.2f}")
        logger.info(f"  Leverage: 20x")
        logger.info(f"  Risk per trade: 1.25%")
        logger.info(f"  Max daily loss: 3%")
        logger.info(f"  Max drawdown: 12%")
        logger.info(f"  Cycle interval: 30 seconds")
        
        # Create orchestrator
        logger.info("Creating trading orchestrator...")
        orchestrator = TradingOrchestrator(
            weex_client=client,
            initial_equity=INITIAL_EQUITY,
            logger=logger
        )
        
        logger.info("✅ System initialized successfully!")
        logger.info("Enabled symbols:")
        for symbol in orchestrator.ENABLED_SYMBOLS:
            logger.info(f"  • {symbol}")
        
        # Confirm before starting (only in CLI mode)
        if not skip_confirmation:
            logger.warning("⚠️  WARNING: This will start LIVE TRADING with real money!")
            logger.warning("⚠️  Make sure you understand the risks and have reviewed the documentation.")
            
            try:
                response = input("Type 'YES' to start trading, or anything else to exit: ")
                
                if response.strip().upper() != "YES":
                    logger.info("❌ Trading not started. Exiting safely.")
                    return 0
            except (EOFError, OSError):
                # Running in non-interactive mode (API/background), proceed automatically
                logger.warning("⚠️  Running in non-interactive mode - starting automatically")
                pass
        else:
            logger.warning("⚠️  WARNING: Starting LIVE TRADING in background mode")
        
        logger.info("🚀 Starting institutional trading system...")
        logger.info("   Press Ctrl+C to stop")
        
        # Start continuous trading
        orchestrator.run_continuous()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Shutdown signal received...")
        logger.info("Shutting down gracefully...")
        
        # Orchestrator handles position closure in run_continuous
        logger.info("✅ Shutdown complete")
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        logger.error(f"\n❌ Error: {e}")
        logger.error("   Check the log file for details")
        return 1


if __name__ == "__main__":
    sys.exit(main())
