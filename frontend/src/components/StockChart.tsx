
import { useEffect, useMemo, useRef, useState } from "react";
import {
  createChart,
  ColorType,
  CandlestickSeries,
  BarSeries,
  LineSeries,
  AreaSeries,
  HistogramSeries,
  type UTCTimestamp,
} from "lightweight-charts";

type Mode = "candles" | "bars" | "line" | "area";

type PriceBar = {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

type ChartPoint = {
  time: UTCTimestamp;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

type Props = {
  bars: PriceBar[];
  symbol: string;
  forecast?: {
    timestamp: string;
    predicted_close: number;
    last_bar_timestamp: string;
    last_close: number;
  } | null;
};

export default function StockChart({ bars, symbol, forecast }: Props) {
  const [mode, setMode] = useState<Mode>("candles");

  const priceHost = useRef<HTMLDivElement>(null);
  const volumeHost = useRef<HTMLDivElement>(null);

  const priceChart = useRef<any>(null);
  const volumeChart = useRef<any>(null);
  const priceSeries = useRef<any>(null);
  const volumeSeries = useRef<any>(null);
  const forecastSeries = useRef<any>(null);

  const data: ChartPoint[] = useMemo(() => {
    return bars
      .map((bar) => ({
        ...bar,
        time: Math.floor(new Date(bar.timestamp).getTime() / 1000) as UTCTimestamp,
      }))
      .filter((bar) => Number.isFinite(Number(bar.time)))
      .sort((a, b) => Number(a.time) - Number(b.time))
      .filter(
        (bar, index, all) =>
          index === 0 || bar.time !== all[index - 1].time
      );
  }, [bars]);

  // Build the price and volume chart instances.
  useEffect(() => {
    if (!priceHost.current || !volumeHost.current) return;

    const price = createChart(priceHost.current, {
      width: priceHost.current.clientWidth,
      height: 430,
      layout: {
        background: {
          type: ColorType.Solid,
          color: "#0b1220",
        },
        textColor: "#cbd5e1",
      },
      grid: {
        vertLines: { color: "#1e293b" },
        horzLines: { color: "#1e293b" },
      },
      rightPriceScale: { borderColor: "#334155" },
      timeScale: {
        borderColor: "#334155",
        timeVisible: true,
        secondsVisible: false,
      },
    });

    const volume = createChart(volumeHost.current, {
      width: volumeHost.current.clientWidth,
      height: 125,
      layout: {
        background: {
          type: ColorType.Solid,
          color: "#0b1220",
        },
        textColor: "#cbd5e1",
      },
      grid: {
        vertLines: { color: "#1e293b" },
        horzLines: { color: "#1e293b" },
      },
      rightPriceScale: { borderColor: "#334155" },
      timeScale: { visible: false },
    });

    let series: any;

    if (mode === "candles") {
      series = price.addSeries(CandlestickSeries, {
        upColor: "#22c55e",
        downColor: "#ef4444",
        wickUpColor: "#22c55e",
        wickDownColor: "#ef4444",
        borderVisible: false,
      });
    } else if (mode === "bars") {
      series = price.addSeries(BarSeries, {
        upColor: "#22c55e",
        downColor: "#ef4444",
      });
    } else if (mode === "line") {
      series = price.addSeries(LineSeries, {
        color: "#38bdf8",
        lineWidth: 2,
      });
    } else {
      series = price.addSeries(AreaSeries, {
        lineColor: "#38bdf8",
        topColor: "rgba(56,189,248,0.35)",
        bottomColor: "rgba(56,189,248,0.02)",
      });
    }

    const volumeBars = volume.addSeries(HistogramSeries, {
      priceFormat: { type: "volume" },
      priceScaleId: "",
    });
    const forecastLine = price.addSeries(LineSeries, {
      color: "#b6f36b",
      lineWidth: 2,
      lineStyle: 2,
      pointMarkersVisible: true,
      priceLineVisible: false,
      lastValueVisible: true,
      title: "Forecast",
    });

    // Keep price and volume panes horizontally synchronized.
    price.timeScale().subscribeVisibleLogicalRangeChange((range: any) => {
      if (range) volume.timeScale().setVisibleLogicalRange(range);
    });

    priceChart.current = price;
    volumeChart.current = volume;
    priceSeries.current = series;
    volumeSeries.current = volumeBars;
    forecastSeries.current = forecastLine;

    const resize = () => {
      if (priceHost.current && volumeHost.current) {
        price.applyOptions({
          width: priceHost.current.clientWidth,
        });
        volume.applyOptions({
          width: volumeHost.current.clientWidth,
        });
      }
    };

    window.addEventListener("resize", resize);

    return () => {
      window.removeEventListener("resize", resize);
      price.remove();
      volume.remove();
      priceChart.current = null;
      volumeChart.current = null;
      priceSeries.current = null;
      volumeSeries.current = null;
      forecastSeries.current = null;
    };
  }, [mode]);

  // Update the chart when new market data arrives.
  useEffect(() => {
    if (
      !priceSeries.current ||
      !volumeSeries.current ||
      !forecastSeries.current ||
      data.length === 0
    ) {
      return;
    }

    if (mode === "candles" || mode === "bars") {
      priceSeries.current.setData(
        data.map(({ time, open, high, low, close }) => ({
          time, open, high, low, close,
        }))
      );
    } else {
      priceSeries.current.setData(
        data.map(({ time, close }) => ({
          time,
          value: close,
        }))
      );
    }

    volumeSeries.current.setData(
      data.map((bar) => ({
        time: bar.time,
        value: bar.volume,
        color: bar.close >= bar.open ? "#22c55e" : "#ef4444",
      }))
    );

    if (forecast) {
      const lastTime = Math.floor(
        new Date(forecast.last_bar_timestamp).getTime() / 1000,
      );
      const futureTime = Math.floor(
        new Date(forecast.timestamp).getTime() / 1000,
      );
      if (Number.isFinite(lastTime) && Number.isFinite(futureTime)) {
        forecastSeries.current.setData([
          { time: lastTime as UTCTimestamp, value: forecast.last_close },
          { time: futureTime as UTCTimestamp, value: forecast.predicted_close },
        ]);
      } else {
        forecastSeries.current.setData([]);
      }
    } else {
      forecastSeries.current.setData([]);
    }

    priceChart.current?.timeScale().fitContent();
    volumeChart.current?.timeScale().fitContent();
  }, [data, mode, forecast]);

  return (
    <section className="stock-chart">
      <header className="chart-header">
        <div>
          <h2>{symbol}</h2>
          <p>
            Price action and volume
            {forecast ? " · dotted line marks selected-model forecast" : ""}
          </p>
        </div>

        <label>
          Chart type{" "}
          <select
            value={mode}
            onChange={(event) =>
              setMode(event.target.value as Mode)
            }
          >
            <option value="candles">Candlesticks</option>
            <option value="bars">OHLC bars</option>
            <option value="line">Line</option>
            <option value="area">Area</option>
          </select>
        </label>
      </header>

      <div ref={priceHost} style={{ width: "100%" }} />
      <div ref={volumeHost} style={{ width: "100%" }} />

      <small>
        Market timestamps and volume depend on the selected data provider.
      </small>
    </section>
  );
}
