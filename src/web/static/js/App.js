import React, { useState } from 'react';
import { Box, Grid, Paper, Typography, Tabs, Tab } from '@mui/material';
import { styled } from '@mui/material/styles';
import ChatWindow from './chat';
import MetricsDashboard from './metrics';
import EventLog from './events';
import ErrorLog from './errors';

const StyledPaper = styled(Paper)(({ theme }) => ({
    padding: theme.spacing(2),
    height: '100%',
    display: 'flex',
    flexDirection: 'column'
}));

const App = () => {
    const [activeTab, setActiveTab] = useState(0);

    const handleTabChange = (event, newValue) => {
        setActiveTab(newValue);
    };

    return (
        <Box sx={{ height: '100vh', p: 2, bgcolor: 'background.default' }}>
            <Grid container spacing={2} sx={{ height: '100%' }}>
                {/* Main content */}
                <Grid item xs={12} md={8}>
                    <StyledPaper>
                        <Typography variant="h5" gutterBottom>
                            AI Trading Bot Dashboard
                        </Typography>
                        <Tabs value={activeTab} onChange={handleTabChange}>
                            <Tab label="Metrics" />
                            <Tab label="Events" />
                            <Tab label="Errors" />
                        </Tabs>
                        <Box sx={{ flex: 1, mt: 2, overflow: 'auto' }}>
                            {activeTab === 0 && <MetricsDashboard />}
                            {activeTab === 1 && <EventLog />}
                            {activeTab === 2 && <ErrorLog />}
                        </Box>
                    </StyledPaper>
                </Grid>

                {/* Chat window */}
                <Grid item xs={12} md={4}>
                    <StyledPaper>
                        <ChatWindow />
                    </StyledPaper>
                </Grid>
            </Grid>
        </Box>
    );
};

export default App; 