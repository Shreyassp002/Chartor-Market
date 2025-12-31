import { Asset, TradingMode, TRADING_MODES } from "@/types/trading";
import { cn } from "@/lib/utils";
import { Activity, TrendingUp, TrendingDown, BarChart3, Volume2 } from "lucide-react";
import { TradingChart } from "./TradingChart";

interface ChartPanelProps {
  asset: Asset;
  tradingMode: TradingMode;
  data: any[];
}

export function ChartPanel({ asset, tradingMode, data }: ChartPanelProps) {
  const modeConfig = TRADING_MODES[tradingMode];
  const isPositive = asset.change24h >= 0;

  const formatPrice = (price: number) => {
    if (price >= 1000) return `$${price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    if (price >= 1) return `$${price.toFixed(2)}`;
    return `$${price.toFixed(4)}`;
  };

  const formatVolume = (volume: number) => {
    if (volume >= 1e9) return `$${(volume / 1e9).toFixed(2)}B`;
    if (volume >= 1e6) return `$${(volume / 1e6).toFixed(2)}M`;
    return `$${volume.toLocaleString()}`;
  };

  const isLoading = asset.price === 0;

  const metrics = [
    { label: 'Mark Price', value: formatPrice(asset.price), icon: Activity },
    { label: '24h High', value: formatPrice(asset.high24h), icon: TrendingUp, positive: true },
    { label: '24h Low', value: formatPrice(asset.low24h), icon: TrendingDown, negative: true },
    { label: '24h Volume', value: formatVolume(asset.volume24h), icon: BarChart3 },
  ];

  return (
    <div className="flex-1 flex flex-col h-full bg-[#111] rounded-3xl border border-[#1f1f22] overflow-hidden">
      {/* Top Bar */}
      <div className="p-4 border-b border-[#1f1f22]">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-4">
            <h2 className="text-lg font-semibold text-white">{asset.pair}</h2>
            <span className={cn(
              "px-2 py-1 rounded text-xs font-medium uppercase tracking-wide",
              tradingMode === 'scalping' && "bg-emerald-500/20 text-emerald-400",
              tradingMode === 'intraday' && "bg-blue-500/20 text-blue-400",
              tradingMode === 'swing' && "bg-purple-500/20 text-purple-400",
            )}>
              {modeConfig.label} Mode
            </span>
          </div>
          <div className={cn(
            "flex items-center gap-2 px-3 py-1.5 rounded-lg font-mono text-sm",
            isPositive ? "bg-emerald-500/10 text-emerald-400" : "bg-red-500/10 text-red-400"
          )}>
            {isPositive ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
            {isPositive ? '+' : ''}{asset.change24h.toFixed(2)}%
          </div>
        </div>

        {/* Metrics Row */}
        <div className="grid grid-cols-4 gap-4">
          {isLoading ? (
            [...Array(4)].map((_, i) => (
              <div key={i} className="flex items-center gap-2">
                <div className="w-4 h-4 rounded bg-[#27272a] animate-pulse" />
                <div>
                  <div className="h-3 bg-[#27272a] rounded w-16 mb-1 animate-pulse" />
                  <div className="h-4 bg-[#27272a] rounded w-20 animate-pulse" />
                </div>
              </div>
            ))
          ) : (
            metrics.map((metric) => (
              <div key={metric.label} className="flex items-center gap-2">
                <metric.icon className={cn(
                  "w-4 h-4",
                  metric.positive ? "text-emerald-400" : metric.negative ? "text-red-400" : "text-[#888]"
                )} />
                <div>
                  <div className="text-xs text-[#888]">{metric.label}</div>
                  <div className={cn(
                    "font-mono text-sm font-medium",
                    metric.positive ? "text-emerald-400" : metric.negative ? "text-red-400" : "text-white"
                  )}>
                    {metric.value}
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Main Chart Area */}
      <div className="flex-1 relative min-h-0 bg-[#111]">
        {/* AI Sentinel Badge */}
        <div className="absolute top-4 left-4 z-10 flex items-center gap-2 px-3 py-1.5 rounded-lg bg-[#1f1f22]/90 border border-[#27272a] backdrop-blur-sm">
          <div className="relative w-2 h-2">
            <div className="absolute inset-0 rounded-full bg-primary animate-ping opacity-75" />
            <div className="relative w-2 h-2 rounded-full bg-primary" />
          </div>
          <span className="text-xs font-mono text-[#888]">
            AI Sentinel: <span className="text-primary">SCANNING...</span>
          </span>
        </div>

        {/* Chart - Full Cover */}
        <div className="absolute inset-0 w-full h-full bg-[#111]">
          {data && data.length > 0 ? (
            <TradingChart data={data} />
          ) : (
            <div className="flex items-center justify-center h-full bg-[#111]">
              <div className="text-center animate-pulse">
                <BarChart3 className="w-16 h-16 mx-auto mb-4 text-[#888]/30" />
                <p className="text-sm text-[#888]">
                  Waiting for Live Market Data...
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
