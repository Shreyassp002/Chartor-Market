"""
Close ALL positions using WEEX closePositions API endpoint
"""
import sys
sys.path.append('/home/ubuntu/Chartor-Market')

from core.weex_api import WeexClient
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

logger.info("="*60)
logger.info("🚨 CLOSING ALL POSITIONS (WEEX API)")
logger.info("="*60)

client = WeexClient()

# Use the closePositions endpoint - closes ALL positions when no symbol provided
endpoint = "/capi/v2/order/closePositions"

logger.info("\nClosing ALL positions at market price...")

try:
    # Empty params = close all positions
    response = client._send_weex_request("POST", endpoint, params={})
    
    logger.info(f"\nAPI Response: {response}")
    
    if response and isinstance(response, list):
        logger.info(f"\n✅ Close operation completed!")
        logger.info(f"Processed {len(response)} position(s):\n")
        
        for item in response:
            position_id = item.get('positionId')
            success = item.get('success')
            order_id = item.get('successOrderId')
            error = item.get('errorMessage', '')
            
            if success:
                logger.info(f"  ✅ Position {position_id}: CLOSED (Order: {order_id})")
            else:
                logger.error(f"  ❌ Position {position_id}: FAILED - {error}")
    else:
        logger.error(f"❌ Unexpected response format: {response}")

except Exception as e:
    logger.error(f"❌ Error: {e}", exc_info=True)

# Check final balance
logger.info("\n" + "="*60)
logger.info("Checking final balance...")
logger.info("="*60)

try:
    balance = client.get_balance()
    if balance and isinstance(balance, list):
        for item in balance:
            if item.get('coinName') == 'USDT':
                available = item.get('available')
                equity = item.get('equity')
                unrealized = item.get('unrealizePnl', 0)
                
                logger.info(f"\n💰 Available: ${available}")
                logger.info(f"💰 Equity: ${equity}")
                logger.info(f"💰 Unrealized PnL: ${unrealized}")
                
                if float(unrealized) == 0:
                    logger.info("\n✅✅✅ ALL POSITIONS CLOSED! ✅✅✅")
                else:
                    logger.warning(f"\n⚠️ Still have positions (Unrealized: ${unrealized})")
except Exception as e:
    logger.error(f"Balance check failed: {e}")

logger.info("\n" + "="*60)
logger.info("DONE")
logger.info("="*60)
