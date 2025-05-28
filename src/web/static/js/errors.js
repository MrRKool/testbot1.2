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
    CircularProgress,
    Collapse,
    IconButton
} from '@mui/material';
import { KeyboardArrowDown, KeyboardArrowUp } from '@mui/icons-material';

const ErrorLog = () => {
    const [errors, setErrors] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [page, setPage] = useState(0);
    const [rowsPerPage, setRowsPerPage] = useState(10);
    const [expandedRows, setExpandedRows] = useState({});

    useEffect(() => {
        const fetchErrors = async () => {
            try {
                const response = await fetch('/api/errors/summary');
                const data = await response.json();
                setErrors(data.latest_errors);
                setLoading(false);
            } catch (err) {
                setError(err.message);
                setLoading(false);
            }
        };

        fetchErrors();
        const interval = setInterval(fetchErrors, 5000); // Update every 5 seconds

        return () => clearInterval(interval);
    }, []);

    const handleChangePage = (event, newPage) => {
        setPage(newPage);
    };

    const handleChangeRowsPerPage = (event) => {
        setRowsPerPage(parseInt(event.target.value, 10));
        setPage(0);
    };

    const toggleRow = (index) => {
        setExpandedRows(prev => ({
            ...prev,
            [index]: !prev[index]
        }));
    };

    const getErrorTypeColor = (type) => {
        switch (type) {
            case 'uncaught_exception':
                return 'error';
            case 'api_error':
                return 'warning';
            case 'validation_error':
                return 'info';
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
                <Typography color="error">Error loading errors: {error}</Typography>
            </Box>
        );
    }

    return (
        <Box>
            <Paper sx={{ p: 2, mb: 2 }}>
                <Typography variant="h6" gutterBottom>
                    Error Summary
                </Typography>
                <Box sx={{ display: 'flex', gap: 2 }}>
                    <Chip
                        label={`Total Errors: ${errors.length}`}
                        color="error"
                        variant="outlined"
                    />
                    {Object.entries(errors.reduce((acc, err) => {
                        acc[err.type] = (acc[err.type] || 0) + 1;
                        return acc;
                    }, {})).map(([type, count]) => (
                        <Chip
                            key={type}
                            label={`${type}: ${count}`}
                            color={getErrorTypeColor(type)}
                            variant="outlined"
                        />
                    ))}
                </Box>
            </Paper>

            <TableContainer component={Paper}>
                <Table>
                    <TableHead>
                        <TableRow>
                            <TableCell />
                            <TableCell>Timestamp</TableCell>
                            <TableCell>Type</TableCell>
                            <TableCell>Message</TableCell>
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {errors
                            .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                            .map((error, index) => (
                                <React.Fragment key={index}>
                                    <TableRow>
                                        <TableCell>
                                            <IconButton
                                                size="small"
                                                onClick={() => toggleRow(index)}
                                            >
                                                {expandedRows[index] ? <KeyboardArrowUp /> : <KeyboardArrowDown />}
                                            </IconButton>
                                        </TableCell>
                                        <TableCell>
                                            {new Date(error.timestamp).toLocaleString()}
                                        </TableCell>
                                        <TableCell>
                                            <Chip
                                                label={error.type}
                                                color={getErrorTypeColor(error.type)}
                                                size="small"
                                            />
                                        </TableCell>
                                        <TableCell>
                                            {error.type === 'uncaught_exception'
                                                ? error.exception_message
                                                : error.message}
                                        </TableCell>
                                    </TableRow>
                                    <TableRow>
                                        <TableCell style={{ paddingBottom: 0, paddingTop: 0 }} colSpan={4}>
                                            <Collapse in={expandedRows[index]} timeout="auto" unmountOnExit>
                                                <Box sx={{ margin: 1 }}>
                                                    <Typography variant="h6" gutterBottom component="div">
                                                        Stack Trace
                                                    </Typography>
                                                    <pre style={{
                                                        backgroundColor: '#f5f5f5',
                                                        padding: '1rem',
                                                        borderRadius: '4px',
                                                        overflow: 'auto'
                                                    }}>
                                                        {error.type === 'uncaught_exception'
                                                            ? error.stack_trace
                                                            : error.stack_trace || 'No stack trace available'}
                                                    </pre>
                                                </Box>
                                            </Collapse>
                                        </TableCell>
                                    </TableRow>
                                </React.Fragment>
                            ))}
                    </TableBody>
                </Table>
                <TablePagination
                    rowsPerPageOptions={[5, 10, 25]}
                    component="div"
                    count={errors.length}
                    rowsPerPage={rowsPerPage}
                    page={page}
                    onPageChange={handleChangePage}
                    onRowsPerPageChange={handleChangeRowsPerPage}
                />
            </TableContainer>
        </Box>
    );
};

export default ErrorLog; 