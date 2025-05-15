"use client";
import { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const Home = () => {
  const [prophetData, setProphetData] = useState([]);
  const [xgboostData, setXgboostData] = useState([]);

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
      <h1>📈 GOOG Stock Predictions</h1>
      
      <h2>Prophet (Next 30 Days)</h2>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={prophetData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="ds" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="y" stroke="#8884d8" dot={false} />
        </LineChart>
      </ResponsiveContainer>

      <h2>XGBoost (Next 30 Days)</h2>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={xgboostData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="day" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="prediction" stroke="#82ca9d" />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default Home;