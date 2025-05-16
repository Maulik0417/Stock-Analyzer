"use client";
import { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { useMemo } from 'react';

const Home = () => {
  const [prophetData, setProphetData] = useState([]);
  const [xgboostData, setXgboostData] = useState([]);

  const calculateYDomain = (data: any[], key: string) => {
  const values = data.map(d => d[key]);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const flooredMin = Math.floor(min / 10) * 10-5;
  const ceiledMax = Math.ceil(max / 10) * 10+5;
  return [flooredMin, ceiledMax];
};

const prophetYDomain = useMemo(() => {
  return calculateYDomain(prophetData, 'y');
}, [prophetData]);

const xgboostYDomain = useMemo(() => {
  return calculateYDomain(xgboostData, 'prediction');
}, [xgboostData]);

  useEffect(() => {
    // Fetch Prophet predictions
    axios.get('http://127.0.0.1:8000/predict/prophet')
      .then(response => {
        setProphetData(response.data.predictions);
      })
      .catch(error => {
        console.error('Error fetching Prophet data:', error);
      });

    // Fetch XGBoost predictions
    axios.get('http://127.0.0.1:8000/predict/xgboost')
      .then(response => {
        setXgboostData(response.data.predictions);
      })
      .catch(error => {
        console.error('Error fetching XGBoost data:', error);
      });
  }, []);



  return (
    <div>
      <h1>📈 Stock Predictions</h1>
      
      <h2>Prophet (Next 20 Days)</h2>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={prophetData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="ds"
            tickFormatter={(tick) => {
              const date = new Date(tick);
              return `${date.toLocaleString("default", { month: "short" })}-${date.getDate()}`;
            }}
            interval={1}
          />
          <YAxis
            domain={prophetYDomain}
          />
          <Tooltip />
          
          <Line type="monotone" dataKey="y" stroke="#8884d8" dot={false} />
        </LineChart>
      </ResponsiveContainer>

      <h2>XGBoost (Next 30 Days)</h2>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={xgboostData}>
          <CartesianGrid strokeDasharray="3 3" />
       <XAxis
        dataKey="ds"
        tickFormatter={(tick) => {
          const date = new Date(tick);
          return `${date.toLocaleString("default", { month: "short" })}-${date.getDate()}`;
        }}
        interval={1}
      />
<YAxis domain={xgboostYDomain} />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="prediction" stroke="#82ca9d" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default Home;