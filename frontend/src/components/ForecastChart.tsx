import { useMemo, useRef, useState } from "react";
import type { PointerEvent, WheelEvent } from "react";
import type { Bar, ForecastPoint } from "../api";

type Props = {
  bars: Bar[];
  symbol: string;
  future: ForecastPoint[];
  lastClose: number;
};

type PlotPoint = {
  timestamp: string;
  value: number;
  lower: number | null;
  upper: number | null;
  forecast: boolean;
};

const WIDTH = 900;
const HEIGHT = 340;
const LEFT = 76;
const RIGHT = 18;
const TOP = 20;
const BOTTOM = 42;
const MIN_VISIBLE_POINTS = 20;

function formatValue(value: number) {
  return new Intl.NumberFormat(undefined, {
    maximumFractionDigits: 2,
  }).format(value);
}

function formatDate(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : date.toLocaleDateString(undefined, {
        month: "short",
        day: "numeric",
        year: "2-digit",
      });
}

export default function ForecastChart({ bars, symbol, future, lastClose }: Props) {
  const points = useMemo<PlotPoint[]>(
    () => [
      ...bars.map((bar) => ({
        timestamp: bar.timestamp,
        value: bar.close,
        lower: null,
        upper: null,
        forecast: false,
      })),
      ...future.map((point) => ({
        timestamp: point.timestamp,
        value: point.predicted_close,
        lower: point.lower_80,
        upper: point.upper_80,
        forecast: true,
      })),
    ],
    [bars, future],
  );
  const [rangeState, setRangeState] = useState<{
    points: PlotPoint[];
    range: [number, number];
  } | null>(null);
  const [hoverIndex, setHoverIndex] = useState<number | null>(null);
  const drag = useRef<{ pointerX: number; range: [number, number] } | null>(null);
  const defaultRange: [number, number] = [
    Math.max(0, points.length - Math.max(MIN_VISIBLE_POINTS, future.length + 45)),
    Math.max(0, points.length - 1),
  ];
  const visibleRange =
    rangeState?.points === points ? rangeState.range : defaultRange;
  const [start, end] = visibleRange;
  const visiblePoints = points.slice(start, end + 1);
  const visibleValues = visiblePoints.flatMap((point) => [
    point.value,
    ...(point.lower === null ? [] : [point.lower]),
    ...(point.upper === null ? [] : [point.upper]),
  ]);
  const rawMin = visibleValues.length ? Math.min(...visibleValues) : 0;
  const rawMax = visibleValues.length ? Math.max(...visibleValues) : 1;
  const padding = (rawMax - rawMin) * 0.1 || Math.max(Math.abs(rawMin) * 0.01, 1);
  const min = Math.max(0, rawMin - padding);
  const max = rawMax + padding;
  const plotWidth = WIDTH - LEFT - RIGHT;
  const plotHeight = HEIGHT - TOP - BOTTOM;
  const x = (index: number) =>
    LEFT + ((index - start) / Math.max(1, end - start)) * plotWidth;
  const y = (value: number) => TOP + ((max - value) / (max - min)) * plotHeight;
  const makePolyline = (indices: number[], valueAt: (point: PlotPoint) => number) =>
    indices
      .filter((index) => index >= start && index <= end)
      .map((index) => `${x(index)},${y(valueAt(points[index]))}`)
      .join(" ");

  const actualPoints = Array.from({ length: bars.length }, (_, index) => index);
  const forecastPoints = [
    Math.max(0, bars.length - 1),
    ...Array.from({ length: future.length }, (_, index) => bars.length + index),
  ];
  const forecastPolyline = makePolyline(forecastPoints, (point) => point.value);
  const actualPolyline = makePolyline(actualPoints, (point) => point.value);
  const hasInterval =
    future.length > 0 &&
    future.every((point) => point.lower_80 !== null && point.upper_80 !== null);
  const lowerPoints = [
    Math.max(0, bars.length - 1),
    ...Array.from({ length: future.length }, (_, index) => bars.length + index),
  ];
  const intervalPolygon = hasInterval
    ? [
        ...lowerPoints
          .filter((index) => index >= start && index <= end)
          .map((index) => `${x(index)},${y(index === bars.length - 1 ? lastClose : points[index].lower!)}`),
        ...lowerPoints
          .filter((index) => index >= start && index <= end)
          .reverse()
          .map((index) => `${x(index)},${y(index === bars.length - 1 ? lastClose : points[index].upper!)}`),
      ].join(" ")
    : "";
  const labels = [start, Math.round((start + end) / 2), end]
    .filter((index, position, all) => all.indexOf(index) === position)
    .map((index) => ({ index, label: formatDate(points[index]?.timestamp ?? "") }));
  const hoveredPoint =
    hoverIndex !== null && hoverIndex >= start && hoverIndex <= end
      ? points[hoverIndex]
      : null;

  function setRange(nextRange: [number, number]) {
    setRangeState({ points, range: nextRange });
  }

  function zoom(factor: number) {
    const currentCount = Math.max(MIN_VISIBLE_POINTS, end - start + 1);
    const nextCount = Math.min(points.length, Math.max(MIN_VISIBLE_POINTS, Math.round(currentCount * factor)));
    const center = (start + end) / 2;
    const nextStart = Math.max(0, Math.min(points.length - nextCount, Math.round(center - nextCount / 2)));
    setRange([nextStart, nextStart + nextCount - 1]);
  }

  function onWheel(event: WheelEvent<SVGSVGElement>) {
    event.preventDefault();
    zoom(event.deltaY < 0 ? 0.8 : 1.25);
  }

  function onPointerMove(event: PointerEvent<SVGSVGElement>) {
    if (drag.current) {
      const count = end - start + 1;
      const shift = Math.round(
        ((drag.current.pointerX - event.clientX) / event.currentTarget.clientWidth) * count,
      );
      const nextStart = Math.max(
        0,
        Math.min(points.length - count, drag.current.range[0] + shift),
      );
      setRange([nextStart, nextStart + count - 1]);
      return;
    }

    const bounds = event.currentTarget.getBoundingClientRect();
    const localX = ((event.clientX - bounds.left) / bounds.width) * WIDTH;
    const index = Math.round(start + ((localX - LEFT) / plotWidth) * (end - start));
    setHoverIndex(Math.max(start, Math.min(end, index)));
  }

  return (
    <section className="forecast-chart-section" aria-label={`${symbol} interactive forecast chart`}>
      <div className="forecast-chart-heading">
        <div>
          <h3>Historical prices and forecast</h3>
          <p>Observed closes in blue; the green dotted line is the selected model estimate.</p>
        </div>
        <div className="forecast-chart-controls" aria-label="Forecast chart zoom controls">
          <button type="button" onClick={() => zoom(0.8)} aria-label="Zoom in">+</button>
          <button type="button" onClick={() => zoom(1.25)} aria-label="Zoom out">−</button>
          <button
            type="button"
            onClick={() => setRange([
              Math.max(0, points.length - Math.max(MIN_VISIBLE_POINTS, future.length + 45)),
              Math.max(0, points.length - 1),
            ])}
          >
            Reset
          </button>
        </div>
      </div>
      {hasInterval && (
        <span className="interval-legend">
          <i aria-hidden="true" />
          Empirical 80% walk-forward residual range
        </span>
      )}
      {!hasInterval && (
        <p className="unsupported-forecast-note">
          The validation folds did not provide enough residuals to calculate an interval for every forecast date.
        </p>
      )}
      <div className="forecast-svg-wrap">
        <svg
          className="forecast-svg"
          viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
          role="img"
          aria-label={`${symbol} historical closes, model forecast, and empirical uncertainty range`}
          onWheel={onWheel}
          onPointerDown={(event) => {
            drag.current = {
              pointerX: event.clientX,
              range: [start, end],
            };
            event.currentTarget.setPointerCapture(event.pointerId);
          }}
          onPointerMove={onPointerMove}
          onPointerUp={() => {
            drag.current = null;
          }}
          onPointerCancel={() => {
            drag.current = null;
          }}
          onPointerLeave={() => {
            if (!drag.current) setHoverIndex(null);
          }}
        >
          <defs>
            <clipPath id="forecast-plot-clip">
              <rect x={LEFT} y={TOP} width={plotWidth} height={plotHeight} />
            </clipPath>
          </defs>
          {[0, 0.5, 1].map((fraction) => {
            const value = min + (max - min) * fraction;
            const position = TOP + plotHeight * (1 - fraction);
            return (
              <g key={fraction}>
                <line x1={LEFT} x2={WIDTH - RIGHT} y1={position} y2={position} className="forecast-gridline" />
                <text x={LEFT - 9} y={position + 3} textAnchor="end" className="forecast-axis-label">
                  {formatValue(value)}
                </text>
              </g>
            );
          })}
          <g clipPath="url(#forecast-plot-clip)">
            {hasInterval && <polygon points={intervalPolygon} className="forecast-interval-band" />}
            <polyline points={actualPolyline} className="forecast-actual-line" />
            <polyline points={forecastPolyline} className="forecast-estimate-line" />
            {future.map((point, index) => {
              const pointIndex = bars.length + index;
              if (pointIndex < start || pointIndex > end) return null;
              return (
                <circle
                  key={point.timestamp}
                  cx={x(pointIndex)}
                  cy={y(point.predicted_close)}
                  r="3"
                  className="forecast-estimate-dot"
                >
                  <title>
                    {formatDate(point.timestamp)} · Estimate {formatValue(point.predicted_close)}
                    {point.lower_80 !== null && point.upper_80 !== null
                      ? ` · 80% empirical range ${formatValue(point.lower_80)}–${formatValue(point.upper_80)}`
                      : ""}
                  </title>
                </circle>
              );
            })}
            {hoveredPoint && (
              <>
                <line
                  x1={x(hoverIndex!)}
                  x2={x(hoverIndex!)}
                  y1={TOP}
                  y2={HEIGHT - BOTTOM}
                  className="forecast-crosshair"
                />
                <circle
                  cx={x(hoverIndex!)}
                  cy={y(hoveredPoint.value)}
                  r="4"
                  className={hoveredPoint.forecast ? "forecast-hover-dot" : "history-hover-dot"}
                />
              </>
            )}
          </g>
          {labels.map(({ index, label }) => (
            <text
              key={index}
              x={x(index)}
              y={HEIGHT - 14}
              textAnchor={index === start ? "start" : index === end ? "end" : "middle"}
              className="forecast-axis-label"
            >
              {label}
            </text>
          ))}
          {hoveredPoint && hoverIndex !== null && (
            <g className="forecast-hover-label">
              <rect x={Math.min(WIDTH - 190, x(hoverIndex) + 10)} y={TOP + 4} width="178" height="43" rx="4" />
              <text x={Math.min(WIDTH - 180, x(hoverIndex) + 19)} y={TOP + 21}>
                {formatDate(hoveredPoint.timestamp)}
              </text>
              <text x={Math.min(WIDTH - 180, x(hoverIndex) + 19)} y={TOP + 37}>
                {hoveredPoint.forecast ? "Estimate" : "Observed close"}: {formatValue(hoveredPoint.value)}
              </text>
            </g>
          )}
        </svg>
      </div>
      <small>
        Drag to pan; use the mouse wheel or +/− to zoom. The empirical band is derived from same-horizon walk-forward errors and is not guaranteed future coverage.
      </small>
    </section>
  );
}
