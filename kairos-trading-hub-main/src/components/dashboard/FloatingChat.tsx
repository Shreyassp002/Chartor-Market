import { useState } from "react";
import { X, MessageCircle, Send, Bot, User } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { ChatMessage } from "@/types/trading";

interface FloatingChatProps {
  messages: ChatMessage[];
  onSendMessage: (message: string) => void;
}

export function FloatingChat({ messages, onSendMessage }: FloatingChatProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [input, setInput] = useState("");

  const predefinedChats = [
    "What's the current market sentiment for BTC?",
    "Should I buy or sell ETH right now?",
    "Explain the RSI indicator",
    "What's the best trading strategy for today?",
    "Analyze the current trend",
  ];

  const handlePredefinedClick = (message: string) => {
    onSendMessage(message);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (input.trim()) {
      onSendMessage(input.trim());
      setInput("");
    }
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' });
  };

  const formatMessageWithHighlights = (text: string): React.ReactNode => {
    const parts: React.ReactNode[] = [];
    let lastIndex = 0;
    let keyCounter = 0;

    const pricePattern = /\$[\d,]+(?:\.\d{1,2})?/g;
    const percentPattern = /\d+\.?\d*%/g;
    const boldPattern = /\*\*([^*]+)\*\*/g;

    const matches: Array<{ index: number; end: number; type: 'price' | 'percent' | 'bold'; value: string }> = [];

    let match;
    while ((match = pricePattern.exec(text)) !== null) {
      matches.push({
        index: match.index,
        end: match.index + match[0].length,
        type: 'price',
        value: match[0]
      });
    }

    pricePattern.lastIndex = 0;
    while ((match = percentPattern.exec(text)) !== null) {
      matches.push({
        index: match.index,
        end: match.index + match[0].length,
        type: 'percent',
        value: match[0]
      });
    }

    percentPattern.lastIndex = 0;
    while ((match = boldPattern.exec(text)) !== null) {
      matches.push({
        index: match.index,
        end: match.index + match[0].length,
        type: 'bold',
        value: match[1]
      });
    }

    matches.sort((a, b) => a.index - b.index);

    matches.forEach((match) => {
      if (match.index > lastIndex) {
        parts.push(<span key={`text-${keyCounter++}`}>{text.substring(lastIndex, match.index)}</span>);
      }

      if (match.type === 'price') {
        parts.push(
          <span key={`price-${keyCounter++}`} className="text-emerald-400 font-mono font-semibold">
            {match.value}
          </span>
        );
      } else if (match.type === 'percent') {
        const isPositive = !match.value.startsWith('-');
        parts.push(
          <span
            key={`percent-${keyCounter++}`}
            className={cn(
              "font-mono font-semibold",
              isPositive ? "text-emerald-400" : "text-red-400"
            )}
          >
            {match.value}
          </span>
        );
      } else if (match.type === 'bold') {
        parts.push(
          <strong key={`bold-${keyCounter++}`} className="font-semibold text-foreground">
            {match.value}
          </strong>
        );
      }

      lastIndex = match.end;
    });

    if (lastIndex < text.length) {
      parts.push(<span key={`text-end-${keyCounter++}`}>{text.substring(lastIndex)}</span>);
    }

    return parts.length > 0 ? parts : text;
  };

  return (
    <>
      {/* Floating Action Button */}
      <button
        onClick={() => setIsOpen(true)}
        className={cn(
          "fixed bottom-6 right-6 z-40 w-14 h-14 rounded-full bg-[#1f1f22] text-white",
          "flex items-center justify-center",
          "transition-all duration-300 hover:scale-110",
          "hover:bg-[#27272a] overflow-hidden border border-[#27272a]"
        )}
        style={{
          boxShadow: "0 4px 20px rgba(255, 255, 255, 0.1), 0 0 0 1px rgba(255, 255, 255, 0.05)"
        }}
      >
        <img
          src="/ChartorLogo.png"
          alt="Chartor"
          className="w-8 h-8 object-contain invert brightness-0"
        />
      </button>

      {/* Slide-Out Drawer */}
      <div
        className={cn(
          "fixed top-0 right-0 h-full z-50 bg-[#111] border-l border-[#1f1f22]",
          "transition-transform duration-300 ease-in-out",
          "flex flex-col",
          isOpen ? "translate-x-0" : "translate-x-full",
          "w-[400px] max-w-[90vw]"
        )}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-[#1f1f22]">
          <div className="flex items-center gap-2">
            <img
              src="/ChartorLogo.png"
              alt="Chartor"
              className="w-6 h-6 object-contain invert brightness-0"
            />
            <h2 className="text-lg font-semibold text-white">Chartor AI</h2>
          </div>
          <button
            onClick={() => setIsOpen(false)}
            className="p-2 text-[#888] hover:text-white transition-colors rounded-lg hover:bg-[#1f1f22]"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Messages Container */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <img
                src="/ChartorLogo.png"
                alt="Chartor"
                className="w-16 h-16 object-contain mb-4 invert brightness-0"
              />
              <p className="text-[#888] text-sm mb-4">
                Start a conversation with Chartor AI
              </p>
              <p className="text-[#666] text-xs mb-6">
                Ask about market analysis, trading strategies, or get insights
              </p>

              {/* Predefined Chat Suggestions */}
              <div className="w-full space-y-2">
                {predefinedChats.map((chat, index) => (
                  <button
                    key={index}
                    onClick={() => handlePredefinedClick(chat)}
                    className="w-full text-left px-4 py-2.5 rounded-lg bg-[#1f1f22] border border-[#27272a] text-sm text-[#d4d4d8] hover:bg-[#27272a] hover:border-[#3f3f46] transition-colors"
                  >
                    {chat}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((msg) => (
              <div
                key={msg.id}
                className={cn(
                  "flex gap-3",
                  msg.role === "user" ? "justify-end" : "justify-start"
                )}
              >
                {msg.role === "assistant" && (
                  <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center flex-shrink-0 overflow-hidden">
                    <img
                      src="/ChartorLogo.png"
                      alt="Chartor"
                      className="w-full h-full object-contain p-1 invert brightness-0"
                    />
                  </div>
                )}
                <div
                  className={cn(
                    "max-w-[80%] rounded-2xl px-4 py-2.5",
                    msg.role === "user"
                      ? "bg-primary text-white"
                      : "bg-[#1f1f22] text-[#d4d4d8]"
                  )}
                >
                  <div className="text-sm whitespace-pre-wrap">
                    {formatMessageWithHighlights(msg.content)}
                  </div>
                  <div className="text-xs mt-1 opacity-70">
                    {formatTime(msg.timestamp)}
                  </div>
                </div>
                {msg.role === "user" && (
                  <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center flex-shrink-0">
                    <User className="w-4 h-4 text-primary" />
                  </div>
                )}
              </div>
            ))
          )}
        </div>

        {/* Input Form */}
        <form onSubmit={handleSubmit} className="p-4 border-t border-[#1f1f22]">
          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask Chartor AI..."
              className="flex-1 px-4 py-2.5 rounded-lg bg-[#1f1f22] border border-[#27272a] text-white placeholder-[#666] focus:outline-none focus:ring-2 focus:ring-primary"
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit(e);
                }
              }}
            />
            <Button
              type="submit"
              size="icon"
              className="bg-primary hover:bg-primary/90"
            >
              <Send className="w-4 h-4" />
            </Button>
          </div>
        </form>
      </div>

      {/* Backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/50 z-40"
          onClick={() => setIsOpen(false)}
        />
      )}
    </>
  );
}

