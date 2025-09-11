"use client";

// frontend/pages/index.tsx
import React, { useEffect, useState } from "react";
import axios from "axios";
import { Chart as ChartJS, TimeScale, LinearScale, PointElement, LineElement, Tooltip, Legend } from 'chart.js';
import "chartjs-adapter-date-fns";
import { Line } from "react-chartjs-2";

ChartJS.register(TimeScale, LinearScale, PointElement, LineElement, Tooltip, Legend);

type ApiResponse = {
  symbol: string;
  market_open: string;
  market_close: string;
  timestamps: string[];   // ISO strings
  prices: (number | null)[];
  predicted: {
    timestamp: string;
    ensemble: number;
    models: {
      prophet?: number | null;
      random_forest?: number | null;
      lstm?: number | null;
      naive_last_price?: number | null;
    };
  };
};

export default function Home() {
  const [symbol, setSymbol] = useState("GOOG");
  const [data, setData] = useState<ApiResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchData();
  }, []);

  async function fetchData() {
    setLoading(true);
    setError(null);
    try {
      const res = await axios.get<ApiResponse>(`http://localhost:8000/predict?symbol=${symbol}`);
      setData(res.data);
    } catch (e: any) {
      setError(e.message || "Error fetching data");
    } finally {
      setLoading(false);
    }
  }

  const chartData = React.useMemo(() => {
    if (!data) return null;

    // Build datasets:
    // 1) actual price series up to now (nulls allowed) - line
    // 2) predicted ensemble point - scatter or single point connected
    // We'll create a dataset where predicted value is present only at that timestamp.

    const labels = data.timestamps.map(t => new Date(t));

    const actualData = data.prices.map((p, idx) => {
      // show actual values until now (some may be null)
      return p === null ? null : p;
    });

    // Predicted: put value at predicted timestamp; else null
    const predictedTime = new Date(data.predicted.timestamp);
    // find index nearest timestamp in labels
    let nearestIndex = labels.findIndex(d => d.getTime() === predictedTime.getTime());
    if (nearestIndex === -1) {
      // if exact match not found, find nearest
      let minDiff = Number.MAX_SAFE_INTEGER;
      let minIdx = -1;
      for (let i = 0; i < labels.length; i++) {
        const diff = Math.abs(labels[i].getTime() - predictedTime.getTime());
        if (diff < minDiff) {
          minDiff = diff; minIdx = i;
        }
      }
      nearestIndex = minIdx;
    }
    const predictedArray = Array(labels.length).fill(null);
    if (nearestIndex >= 0) predictedArray[nearestIndex] = data.predicted.ensemble;

    return {
      labels,
      datasets: [
        {
          label: `${data.symbol} actual`,
          data: actualData,
          borderWidth: 1.5,
          tension: 0.2,
          spanGaps: false,
          pointRadius: 0,
        },
        {
          label: `Predicted (1h)`,
          data: predictedArray,
          showLine: false,
          pointRadius: 6,
          pointHoverRadius: 8,
          // ChartJS default colors are fine
        },
      ]
    };
  }, [data]);

  const options = {
    responsive: true,
    scales: {
      x: {
        type: "time" as const,
        time: {
          unit: "minute" as const,
          displayFormats: { minute: "HH:mm" }
        },
        ticks: {
          maxRotation: 0,
          autoSkip: true,
          maxTicksLimit: 12
        }
      },
      y: {
        beginAtZero: false,
        ticks: {
          // nice formatting
        }
      }
    },
    plugins: {
      tooltip: {
        mode: "index" as const,
        intersect: false
      },
      legend: { position: "top" as const }
    },
    elements: {
      point: {
        radius: 2
      }
    }
  };

  return (
    <div style={{ padding: 24, fontFamily: "Inter, system-ui, sans-serif" }}>
      <h1>Stock Hour-Ahead Predictor (demo)</h1>

      <div style={{ marginBottom: 12, display: "flex", gap: 8, alignItems: "center" }}>
        <input value={symbol} onChange={(e) => setSymbol(e.target.value.toUpperCase())} />
        <button onClick={fetchData} disabled={loading}>Fetch</button>
        <span style={{ marginLeft: 8 }}>{loading ? "Loading..." : ""}</span>
      </div>

      {error && <div style={{ color: "red" }}>{error}</div>}

      {data && chartData ? (
        <div style={{ width: "100%", maxWidth: 1200 }}>
          <Line data={chartData} options={options} />
          <div style={{ marginTop: 12 }}>
            <strong>Predicted (1 hour):</strong> {data.predicted.ensemble.toFixed(2)} at {new Date(data.predicted.timestamp).toLocaleTimeString()}
            <div style={{ marginTop: 6, fontSize: 14 }}>
              <details>
                <summary>Model breakdown</summary>
                <ul>
                  <li>Prophet: {data.predicted.models.prophet ?? "n/a"}</li>
                  <li>RandomForest: {data.predicted.models.random_forest ?? "n/a"}</li>
                  <li>LSTM: {data.predicted.models.lstm ?? "n/a"}</li>
                  <li>Naive last price: {data.predicted.models.naive_last_price}</li>
                </ul>
              </details>
            </div>
          </div>
        </div>
      ) : (
        <div>No data yet. Click Fetch.</div>
      )}

    </div>
  );
}