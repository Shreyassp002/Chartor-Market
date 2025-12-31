import { Asset } from "@/types/trading";
import { cn } from "@/lib/utils";
import { TrendingUp, TrendingDown } from "lucide-react";

// Asset image mapping
const getAssetImage = (symbol: string): string => {
  const symbolUpper = symbol.toUpperCase();
  const imageMap: Record<string, string> = {
    'BTC': '/BTC.png',
    'ETH': '/ETH.png',
    'SOL': '/SOL.png',
    'DOGE': '/DOGE.png',
    'XRP': '/XRP.png',
    'ADA': '/ADA.png',
    'LTC': '/LTC.png',
    'BNB': '/BNB.png',
  };
  return imageMap[symbolUpper] || '';
};

interface MarketTickerProps {
  assets: Asset[];
  selectedAsset: string;
  onAssetSelect: (symbol: string) => void;
  isLoading?: boolean;
}

export function MarketTicker({ assets, selectedAsset, onAssetSelect, isLoading }: MarketTickerProps) {
  const formatPrice = (price: number) => {
    if (price >= 1000) return `$${price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    if (price >= 1) return `$${price.toFixed(2)}`;
    return `$${price.toFixed(4)}`;
  };

  const formatChange = (change: number) => {
    const sign = change >= 0 ? '+' : '';
    return `${sign}${change.toFixed(2)}%`;
  };

  if (isLoading) {
    return (
      <div className="space-y-2">
        {[...Array(5)].map((_, i) => (
          <div key={i} className="h-16 bg-[#1f1f22] rounded-xl animate-pulse" />
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {assets.map((asset) => {
        const isSelected = selectedAsset === asset.symbol;
        const isPositive = asset.change24h >= 0;

        return (
          <button
            key={asset.symbol}
            onClick={() => onAssetSelect(asset.symbol)}
            className={cn(
              "w-full p-4 rounded-xl transition-all duration-200 text-left",
              "bg-[#1f1f22] border border-[#27272a]",
              "hover:bg-[#27272a] hover:border-[#3f3f46]",
              isSelected && "bg-[#1f1f22] border-primary/50 ring-2 ring-primary/20"
            )}
          >
            <div className="flex items-center justify-between">
              {/* Left: Coin Info */}
              <div className="flex items-center gap-3">
                {getAssetImage(asset.symbol) ? (
                  <div className={cn(
                    "w-10 h-10 rounded-full overflow-hidden flex items-center justify-center flex-shrink-0",
                    isSelected && "ring-2 ring-primary/50"
                  )}>
                    <img 
                      src={getAssetImage(asset.symbol)} 
                      alt={asset.symbol}
                      className="w-full h-full object-cover"
                    />
                  </div>
                ) : (
                  <div className={cn(
                    "w-10 h-10 rounded-full flex items-center justify-center font-mono font-bold text-sm flex-shrink-0",
                    isSelected ? "bg-primary/30 text-primary" : "bg-[#27272a] text-[#888]"
                  )}>
                    {asset.symbol.charAt(0)}
                  </div>
                )}
                <div className="min-w-0">
                  <div className="font-semibold text-white text-sm">{asset.symbol}</div>
                  <div className="text-xs text-[#888]">{asset.pair}</div>
                </div>
              </div>

              {/* Right: Price & Change */}
              <div className="text-right">
                <div className="font-mono text-sm text-white font-medium">
                  {formatPrice(asset.price)}
                </div>
                <div className={cn(
                  "flex items-center justify-end gap-1 text-xs font-mono mt-1",
                  isPositive ? "text-emerald-400" : "text-red-400"
                )}>
                  {isPositive ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                  {formatChange(asset.change24h)}
                </div>
              </div>
            </div>
          </button>
        );
      })}
    </div>
  );
}

