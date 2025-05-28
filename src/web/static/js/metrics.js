import React, { useState, useEffect } from 'react';
import { Box, Grid, Paper, Typography, CircularProgress } from '@mui/material';
import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
} from 'chart.js';

ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
);

const MetricsDashboard = () => {
    const [metrics, setMetrics] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchMetrics = async () => {
            try {
                const response = await fetch('/api/metrics/summary');
                const data = await response.json();
                setMetrics(data);
                setLoading(false);
            } catch (err) {
                setError(err.message);
                setLoading(false);
            }
        };

        fetchMetrics();
        const interval = setInterval(fetchMetrics, 5000); // Update every 5 seconds

        return () => clearInterval(interval);
    }, []);

    if (loading) {
        return (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                <CircularProgress />
            </Box>
        );
    }

    if (error) {
        return (
            <Box sx={{ p: 2 }}>
                <Typography color="error">Error loading metrics: {error}</Typography>
            </Box>
        );
    }

    const systemData = {
        labels: ['CPU', 'Memory', 'Disk', 'Network'],
        datasets: [
            {
                label: 'Usage (%)',
                data: [
                    metrics.system.cpu_usage,
                    metrics.system.memory_usage,
                    metrics.system.disk_usage,
                    metrics.system.network_io / 1024 / 1024 // Convert to MB
                ],
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1
            }
        ]
    };

    const tradingData = {
        labels: ['Trades', 'Volume', 'Profit/Loss', 'Win Rate'],
        datasets: [
            {
                label: 'Trading Metrics',
                data: [
                    metrics.trading.trade_count,
                    metrics.trading.trade_volume,
                    metrics.trading.profit_loss,
                    metrics.trading.win_rate * 100
                ],
                borderColor: 'rgb(255, 99, 132)',
                tension: 0.1
            }
        ]
    };

    const modelData = {
        labels: ['Accuracy', 'Loss', 'Latency'],
        datasets: [
            {
                label: 'Model Performance',
                data: [
                    metrics.model.accuracy * 100,
                    metrics.model.loss,
                    metrics.model.avg_prediction_latency
                ],
                borderColor: 'rgb(54, 162, 235)',
                tension: 0.1
            }
        ]
    };

    return (
        <Box>
            <Grid container spacing={2}>
                {/* System Metrics */}
                <Grid item xs={12} md={6}>
                    <Paper sx={{ p: 2 }}>
                        <Typography variant="h6" gutterBottom>
                            System Metrics
                        </Typography>
                        <Line data={systemData} />
                    </Paper>
                </Grid>

                {/* Trading Metrics */}
                <Grid item xs={12} md={6}>
                    <Paper sx={{ p: 2 }}>
                        <Typography variant="h6" gutterBottom>
                            Trading Metrics
                        </Typography>
                        <Line data={tradingData} />
                    </Paper>
                </Grid>

                {/* Model Metrics */}
                <Grid item xs={12}>
                    <Paper sx={{ p: 2 }}>
                        <Typography variant="h6" gutterBottom>
                            Model Performance
                        </Typography>
                        <Line data={modelData} />
                    </Paper>
                </Grid>

                {/* Performance Metrics */}
                <Grid item xs={12}>
                    <Paper sx={{ p: 2 }}>
                        <Typography variant="h6" gutterBottom>
                            Performance Metrics
                        </Typography>
                        <Grid container spacing={2}>
                            <Grid item xs={12} sm={4}>
                                <Typography variant="subtitle1">
                                    Error Count: {metrics.performance.error_count}
                                </Typography>
                            </Grid>
                            <Grid item xs={12} sm={4}>
                                <Typography variant="subtitle1">
                                    Avg Response Time: {metrics.performance.avg_response_time.toFixed(2)}ms
                                </Typography>
                            </Grid>
                            <Grid item xs={12} sm={4}>
                                <Typography variant="subtitle1">
                                    Avg Task Duration: {metrics.performance.avg_task_duration.toFixed(2)}ms
                                </Typography>
                            </Grid>
                        </Grid>
                    </Paper>
                </Grid>
            </Grid>
        </Box>
    );
};

export default MetricsDashboard; 