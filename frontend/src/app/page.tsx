'use client';

import { useEffect, useState } from "react";
import axios from "axios";

type ProphetPrediction = {
  ds: string;
  yhat: number;
  yhat_lower: number;
  yhat_upper: number;
};

export default function Home() {
  const [prophetData, setProphetData] = useState<ProphetPrediction[]>([]);
  const [xgbData, setXgbData] = useState<number[]>([]);

  useEffect(() => {
    axios.get("http://localhost:8000/predict/prophet").then((res) => {
      setProphetData(res.data);
    });
    axios.get("http://localhost:8000/predict/xgboost").then((res) => {
      setXgbData(res.data.predictions);
    });
  }, []);

  return (
    <main style={{ padding: "2rem" }}>
      <h1>📈 GOOG Stock Predictions</h1>
      <h2>Prophet (Next 30 Days)</h2>
      <ul>
        {prophetData.map((entry) => (
          <li key={entry.ds}>
            {entry.ds}: {entry.yhat.toFixed(2)} (±{(entry.yhat_upper - entry.yhat_lower).toFixed(2)})
          </li>
        ))}
      </ul>

      <h2>XGBoost (Next 30 Days)</h2>
      <ul>
        {xgbData.map((val, idx) => (
          <li key={idx}>Day {idx + 1}: {val.toFixed(2)}</li>
        ))}
      </ul>
    </main>
  );
}