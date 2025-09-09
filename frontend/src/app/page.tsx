"use client";
import { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, Legend } from "recharts";

export default function Home() {
  const [data, setData] = useState<any[]>([]);

  useEffect(() => {
    fetch("http://localhost:8000/predict/GOOG")
      .then((res) => res.json())
      .then((res) => {
        const hist = res.historical.map((d: any) => ({
          time: new Date(d.ds).toLocaleTimeString(),
          price: d.y,
        }));
        const pred = res.prediction.map((d: any) => ({
          time: new Date(d.ds).toLocaleTimeString(),
          price: d.yhat,
          predicted: true,
        }));
        setData([...hist, ...pred]);
      });
  }, []);

  return (
    <main className="flex flex-col items-center justify-center min-h-screen p-8">
      <h1 className="text-2xl font-bold mb-4">GOOG Stock Price + Next Hour Prediction</h1>
      <LineChart width={900} height={500} data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="time" />
        <YAxis domain={["auto", "auto"]} />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="price" stroke="#8884d8" dot={false} />
      </LineChart>
    </main>
  );
}