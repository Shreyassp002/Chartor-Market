import { LogEntry } from "@/types/trading";
import { cn } from "@/lib/utils";
import { useEffect, useRef } from "react";

interface TerminalLogProps {
  logs: LogEntry[];
}

export function TerminalLog({ logs }: TerminalLogProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
  };

  const getTypeColor = (type: LogEntry['type']) => {
    switch (type) {
      case 'sentinel': return 'text-primary';
      case 'risk': return 'text-yellow-400';
      case 'trade': return 'text-emerald-400';
      case 'system': return 'text-[#888]';
      default: return 'text-white';
    }
  };

  const getTypeLabel = (type: LogEntry['type']) => {
    return `[${type.toUpperCase()}]`;
  };

  return (
    <div className="h-40 bg-[#111] border-t border-[#1f1f22] overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2 border-b border-[#1f1f22] bg-[#1a1a1a]">
        <span className="text-xs font-medium text-[#888] uppercase tracking-wide">
          System Terminal
        </span>
        <div className="flex items-center gap-1.5">
          <div className="w-2 h-2 rounded-full bg-red-500/50" />
          <div className="w-2 h-2 rounded-full bg-yellow-500/50" />
          <div className="w-2 h-2 rounded-full bg-emerald-500/50" />
        </div>
      </div>
      <div 
        ref={scrollRef}
        className="h-[calc(100%-36px)] overflow-y-auto scrollbar-thin p-3 font-mono text-xs space-y-1 bg-[#111]"
      >
        {logs.map((log) => (
          <div key={log.id} className="flex gap-2 animate-slide-up">
            <span className="text-[#666]">{formatTime(log.timestamp)}</span>
            <span className={cn("font-semibold min-w-[80px]", getTypeColor(log.type))}>
              {getTypeLabel(log.type)}
            </span>
            <span className="text-[#888]">{log.message}</span>
          </div>
        ))}
        <div className="flex items-center gap-1 text-[#666]">
          <span className="animate-pulse">▌</span>
        </div>
      </div>
    </div>
  );
}
