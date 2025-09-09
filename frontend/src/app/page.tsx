"use client";
import { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from "recharts";

export default function Home() {
  const [data, setData] = useState<any[]>([]);

  useEffect(() => {
    async function loadData() {
      const res = await fetch("http://localhost:8000/predict/GOOG");
      const json = await res.json();

      const hist = json.historical.map((d: any) => ({
        time: new Date(d.ds).toLocaleTimeString(), // safe inside useEffect
        price: d.y,
      }));

      const pred = json.prediction.map((d: any) => ({
        time: new Date(d.ds).toLocaleTimeString(),
        price: d.yhat,
        predicted: true,
      }));

      setData([...hist, ...pred]);
    }

    loadData();
  }, []);

  return (
    <main className="flex flex-col items-center justify-center min-h-screen p-8">
      <h1 className="text-2xl font-bold mb-4">
        GOOG Stock Price + Next Hour Prediction
      </h1>
      {data.length > 0 && (
        <LineChart width={900} height={500} data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="time" />
          <YAxis domain={["auto", "auto"]} />
          <Tooltip />
          <Legend />
          <Line
            type="monotone"
            dataKey="price"
            stroke="#8884d8"
            dot={false}
          />
        </LineChart>
      )}
    </main>
  );
}