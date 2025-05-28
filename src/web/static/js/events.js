import React, { useState, useEffect } from 'react';
import {
    Box,
    Paper,
    Typography,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    TablePagination,
    Chip,
    CircularProgress
} from '@mui/material';

const EventLog = () => {
    const [events, setEvents] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [page, setPage] = useState(0);
    const [rowsPerPage, setRowsPerPage] = useState(10);

    useEffect(() => {
        const fetchEvents = async () => {
            try {
                const response = await fetch('/api/events/summary');
                const data = await response.json();
                setEvents(data.latest_events);
                setLoading(false);
            } catch (err) {
                setError(err.message);
                setLoading(false);
            }
        };

        fetchEvents();
        const interval = setInterval(fetchEvents, 5000); // Update every 5 seconds

        return () => clearInterval(interval);
    }, []);

    const handleChangePage = (event, newPage) => {
        setPage(newPage);
    };

    const handleChangeRowsPerPage = (event) => {
        setRowsPerPage(parseInt(event.target.value, 10));
        setPage(0);
    };

    const getEventTypeColor = (type) => {
        switch (type) {
            case 'trade':
                return 'primary';
            case 'system':
                return 'info';
            case 'model':
                return 'success';
            default:
                return 'default';
        }
    };

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
                <Typography color="error">Error loading events: {error}</Typography>
            </Box>
        );
    }

    return (
        <Box>
            <Paper sx={{ p: 2, mb: 2 }}>
                <Typography variant="h6" gutterBottom>
                    Event Summary
                </Typography>
                <Box sx={{ display: 'flex', gap: 2 }}>
                    <Chip
                        label={`Total Events: ${events.length}`}
                        color="primary"
                        variant="outlined"
                    />
                    <Chip
                        label={`Trade Events: ${events.filter(e => e.type === 'trade').length}`}
                        color="primary"
                        variant="outlined"
                    />
                    <Chip
                        label={`System Events: ${events.filter(e => e.type === 'system').length}`}
                        color="info"
                        variant="outlined"
                    />
                    <Chip
                        label={`Model Events: ${events.filter(e => e.type === 'model').length}`}
                        color="success"
                        variant="outlined"
                    />
                </Box>
            </Paper>

            <TableContainer component={Paper}>
                <Table>
                    <TableHead>
                        <TableRow>
                            <TableCell>Timestamp</TableCell>
                            <TableCell>Type</TableCell>
                            <TableCell>Details</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {events
                            .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                            .map((event, index) => (
                                <TableRow key={index}>
                                    <TableCell>
                                        {new Date(event.timestamp).toLocaleString()}
                                    </TableCell>
                                    <TableCell>
                                        <Chip
                                            label={event.type}
                                            color={getEventTypeColor(event.type)}
                                            size="small"
                                        />
                                    </TableCell>
                                    <TableCell>
                                        {event.type === 'trade' && (
                                            <>
                                                Trade ID: {event.trade_id}<br />
                                                Symbol: {event.symbol}<br />
                                                Side: {event.side}<br />
                                                Price: {event.price}<br />
                                                Volume: {event.volume}<br />
                                                P/L: {event.profit_loss}
                                            </>
                                        )}
                                        {event.type === 'system' && (
                                            <>
                                                Event Type: {event.event_type}<br />
                                                {Object.entries(event)
                                                    .filter(([key]) => !['timestamp', 'type', 'event_type'].includes(key))
                                                    .map(([key, value]) => (
                                                        <React.Fragment key={key}>
                                                            {key}: {value}<br />
                                                        </React.Fragment>
                                                    ))}
                                            </>
                                        )}
                                        {event.type === 'model' && (
                                            <>
                                                Model Type: {event.model_type}<br />
                                                Accuracy: {(event.accuracy * 100).toFixed(2)}%<br />
                                                Loss: {event.loss.toFixed(4)}<br />
                                                Epoch: {event.epoch}
                                            </>
                                        )}
                                    </TableCell>
                                </TableRow>
                            ))}
                    </TableBody>
                </Table>
                <TablePagination
                    rowsPerPageOptions={[5, 10, 25]}
                    component="div"
                    count={events.length}
                    rowsPerPage={rowsPerPage}
                    page={page}
                    onPageChange={handleChangePage}
                    onRowsPerPageChange={handleChangeRowsPerPage}
                />
            </TableContainer>
        </Box>
    );
};

export default EventLog; 