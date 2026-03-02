import { useRef, useState, useCallback, useEffect } from 'react';
import type { ScanUpdate, ScanState } from '../types';

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/scan';

interface UseWebSocketReturn {
    scanState: ScanState;
    lastUpdate: ScanUpdate | null;
    connect: () => void;
    disconnect: () => void;
    startScan: () => void;
    stopScan: () => void;
    sendFrame: (frameData: string) => void;
    isConnected: boolean;
    error: string | null;
}

export function useWebSocket(): UseWebSocketReturn {
    const wsRef = useRef<WebSocket | null>(null);
    const reconnectRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const [scanState, setScanState] = useState<ScanState>('idle');
    const [lastUpdate, setLastUpdate] = useState<ScanUpdate | null>(null);
    const [isConnected, setIsConnected] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const cleanup = useCallback(() => {
        if (reconnectRef.current) {
            clearTimeout(reconnectRef.current);
            reconnectRef.current = null;
        }
        if (wsRef.current) {
            wsRef.current.onclose = null;
            wsRef.current.onerror = null;
            wsRef.current.onmessage = null;
            wsRef.current.onopen = null;
            if (wsRef.current.readyState === WebSocket.OPEN ||
                wsRef.current.readyState === WebSocket.CONNECTING) {
                wsRef.current.close();
            }
            wsRef.current = null;
        }
    }, []);

    const connect = useCallback(() => {
        cleanup();
        setScanState('connecting');
        setError(null);

        try {
            const ws = new WebSocket(WS_URL);
            wsRef.current = ws;

            ws.onopen = () => {
                setIsConnected(true);
                setScanState('idle');
                setError(null);
                console.log('[WS] Connected to rPPG server');
            };

            ws.onmessage = (event) => {
                try {
                    const data: ScanUpdate = JSON.parse(event.data);

                    if (data.type === 'status') {
                        if (data.status === 'started') {
                            setScanState('scanning');
                        }
                    } else if (data.type === 'results') {
                        setScanState('completed');
                    } else if (data.type === 'error') {
                        setError(data.message || 'Unknown error');
                    }

                    setLastUpdate(data);
                } catch (e) {
                    console.error('[WS] Parse error:', e);
                }
            };

            ws.onclose = (event) => {
                setIsConnected(false);
                console.log('[WS] Disconnected:', event.code, event.reason);

                if (scanState === 'scanning') {
                    setScanState('error');
                    setError('Connection lost during scan');
                } else {
                    setScanState('idle');
                }
            };

            ws.onerror = () => {
                setError('WebSocket connection error');
                setScanState('error');
            };

        } catch (e) {
            setError(`Failed to connect: ${e}`);
            setScanState('error');
        }
    }, [cleanup, scanState]);

    const disconnect = useCallback(() => {
        cleanup();
        setIsConnected(false);
        setScanState('idle');
        setLastUpdate(null);
    }, [cleanup]);

    const sendMessage = useCallback((message: object) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(message));
        }
    }, []);

    const startScan = useCallback(() => {
        setLastUpdate(null);
        sendMessage({ type: 'start' });
    }, [sendMessage]);

    const stopScan = useCallback(() => {
        sendMessage({ type: 'stop' });
    }, [sendMessage]);

    const sendFrame = useCallback((frameData: string) => {
        if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({
                type: 'frame',
                data: frameData,
            }));
        }
    }, []);

    // Auto-connect on mount
    useEffect(() => {
        connect();
        return () => cleanup();
    }, [connect, cleanup]);

    return {
        scanState,
        lastUpdate,
        connect,
        disconnect,
        startScan,
        stopScan,
        sendFrame,
        isConnected,
        error,
    };
}
