import React, { useState, useEffect, useRef } from 'react';
import { Box, TextField, Button, Paper, Typography, List, ListItem, ListItemText, IconButton } from '@mui/material';
import { Send as SendIcon, Clear as ClearIcon } from '@mui/icons-material';

const ChatWindow = () => {
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState('');
    const [connected, setConnected] = useState(false);
    const ws = useRef(null);
    const messagesEndRef = useRef(null);

    useEffect(() => {
        // Connect to WebSocket
        ws.current = new WebSocket('ws://localhost:8000/ws/chat');

        ws.current.onopen = () => {
            setConnected(true);
        };

        ws.current.onclose = () => {
            setConnected(false);
        };

        ws.current.onmessage = (event) => {
            const message = JSON.parse(event.data);
            setMessages(prev => [...prev, message]);
        };

        // Load chat history
        fetch('/api/chat/history')
            .then(response => response.json())
            .then(data => {
                setMessages(data);
            });

        return () => {
            if (ws.current) {
                ws.current.close();
            }
        };
    }, []);

    useEffect(() => {
        // Scroll to bottom when messages change
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const handleSend = () => {
        if (input.trim() && ws.current) {
            ws.current.send(JSON.stringify({
                type: 'user',
                message: input
            }));
            setInput('');
        }
    };

    const handleClear = () => {
        fetch('/api/chat/clear', { method: 'POST' })
            .then(() => {
                setMessages([]);
            });
    };

    const handleKeyPress = (event) => {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            handleSend();
        }
    };

    return (
        <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <Paper sx={{ p: 2, mb: 2 }}>
                <Typography variant="h6" gutterBottom>
                    AI Trading Bot Chat
                    {connected ? (
                        <Typography component="span" color="success.main" sx={{ ml: 1 }}>
                            ●
                        </Typography>
                    ) : (
                        <Typography component="span" color="error.main" sx={{ ml: 1 }}>
                            ●
                        </Typography>
                    )}
                </Typography>
            </Paper>

            <Paper sx={{ flex: 1, mb: 2, overflow: 'auto' }}>
                <List>
                    {messages.map((message, index) => (
                        <ListItem
                            key={index}
                            sx={{
                                justifyContent: message.type === 'user' ? 'flex-end' : 'flex-start',
                                mb: 1
                            }}
                        >
                            <Paper
                                sx={{
                                    p: 2,
                                    maxWidth: '70%',
                                    bgcolor: message.type === 'user' ? 'primary.light' : 'grey.100'
                                }}
                            >
                                <ListItemText
                                    primary={message.message}
                                    secondary={new Date(message.timestamp).toLocaleString()}
                                />
                            </Paper>
                        </ListItem>
                    ))}
                    <div ref={messagesEndRef} />
                </List>
            </Paper>

            <Box sx={{ display: 'flex', gap: 1 }}>
                <TextField
                    fullWidth
                    multiline
                    maxRows={4}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Type je bericht..."
                    variant="outlined"
                />
                <Button
                    variant="contained"
                    color="primary"
                    onClick={handleSend}
                    disabled={!input.trim()}
                    endIcon={<SendIcon />}
                >
                    Verstuur
                </Button>
                <IconButton
                    color="error"
                    onClick={handleClear}
                    title="Wis chatgeschiedenis"
                >
                    <ClearIcon />
                </IconButton>
            </Box>
        </Box>
    );
};

export default ChatWindow; 