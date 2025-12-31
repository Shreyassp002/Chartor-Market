import { CandlestickSeries, ColorType, IChartApi, UTCTimestamp, createChart } from 'lightweight-charts';
import React, { useEffect, useRef } from 'react';

export const TradingChart = ({ data }) => {
    const chartContainerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);

    useEffect(() => {
        if (!chartContainerRef.current) return;

        const chart = createChart(chartContainerRef.current, {
            layout: {
                background: { type: ColorType.Solid, color: '#111111' },
                textColor: '#d4d4d8', 
            },
            grid: {
                vertLines: { color: '#1f1f22' }, 
                horzLines: { color: '#1f1f22' },
            },
            width: chartContainerRef.current.clientWidth,
            height: chartContainerRef.current.clientHeight || 500,
            timeScale: {
                timeVisible: true,
                secondsVisible: false,
                borderColor: '#27272a',
            },
            rightPriceScale: {
                borderColor: '#27272a',
            },
        });

        const candlestickSeries = chart.addSeries(CandlestickSeries, {
            upColor: '#10b981',
            downColor: '#ef4444',
            borderVisible: false,
            wickUpColor: '#10b981',
            wickDownColor: '#ef4444',
        });

        if (data && data.length > 0) {
            const formattedData = data.map((c) => ({
                ...c,
                time: c.time as UTCTimestamp,
            }));

            candlestickSeries.setData(formattedData);

            const totalCandles = formattedData.length;
            const visibleCandles = 150;

            if (totalCandles > visibleCandles) {
                const startIndex = totalCandles - visibleCandles;
                const startTime = formattedData[startIndex].time as UTCTimestamp;
                const endTime = formattedData[totalCandles - 1].time as UTCTimestamp;

                chart.timeScale().setVisibleRange({
                    from: startTime,
                    to: endTime,
                });
            } else {
                chart.timeScale().fitContent();
            }
        } else {
            chart.timeScale().fitContent();
        }

        chartRef.current = chart;

        const handleResize = () => {
            if (chartContainerRef.current) {
                chart.applyOptions({
                    width: chartContainerRef.current.clientWidth,
                    height: chartContainerRef.current.clientHeight || 500
                });
            }
        };
        window.addEventListener('resize', handleResize);

        const resizeObserver = new ResizeObserver(entries => {
            for (const entry of entries) {
                const { width, height } = entry.contentRect;
                chart.applyOptions({ width, height: height || 500 });
            }
        });

        const containerElement = chartContainerRef.current;
        if (containerElement) {
            resizeObserver.observe(containerElement);
        }

        return () => {
            window.removeEventListener('resize', handleResize);
            if (containerElement) {
                resizeObserver.unobserve(containerElement);
            }
            chart.remove();
        };
    }, [data]);

    return (
        <div ref={chartContainerRef} className="w-full h-full" />
    );
};