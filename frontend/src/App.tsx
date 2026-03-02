import { useState, useCallback, useEffect, useRef } from 'react';
import './App.css';
import CameraView from './components/Camera/CameraView';
import VitalsPanel from './components/Vitals/VitalsPanel';
import ResultsScreen from './components/Results/ResultsScreen';
import { useWebSocket } from './hooks/useWebSocket';
import { useCamera } from './hooks/useCamera';
import type { VitalSigns, FaceDetection, MotionData } from './types';

function App() {
  const {
    scanState,
    lastUpdate,
    connect,
    startScan,
    stopScan,
    sendFrame,
    isConnected,
    error: wsError,
  } = useWebSocket();

  const {
    videoRef,
    canvasRef,
    isStreaming,
    error: cameraError,
    startCamera,
    stopCamera,
    captureFrame,
  } = useCamera();

  // State
  const [vitals, setVitals] = useState<VitalSigns | null>(null);
  const [face, setFace] = useState<FaceDetection | null>(null);
  const [motion, setMotion] = useState<MotionData | null>(null);
  const [signalQuality, setSignalQuality] = useState(0);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [finalResults, setFinalResults] = useState<{
    vitals: VitalSigns;
    signalQuality: number;
  } | null>(null);

  const frameIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isScanning = scanState === 'scanning';

  // Process WebSocket updates
  useEffect(() => {
    if (!lastUpdate) return;

    if (lastUpdate.type === 'update') {
      if (lastUpdate.vitals) setVitals(lastUpdate.vitals);
      if (lastUpdate.face) setFace(lastUpdate.face);
      if (lastUpdate.motion) setMotion(lastUpdate.motion);
      if (lastUpdate.signal_quality !== undefined) setSignalQuality(lastUpdate.signal_quality);
      if (lastUpdate.elapsed_time !== undefined) setElapsedTime(lastUpdate.elapsed_time);
    } else if (lastUpdate.type === 'results') {
      if (lastUpdate.vitals) {
        setFinalResults({
          vitals: lastUpdate.vitals,
          signalQuality: lastUpdate.signal_quality ?? signalQuality,
        });
      }
      stopFrameCapture();
    }
  }, [lastUpdate]);

  // Frame capture loop
  const startFrameCapture = useCallback(() => {
    stopFrameCapture();
    frameIntervalRef.current = setInterval(() => {
      const frame = captureFrame();
      if (frame) {
        sendFrame(frame);
      }
    }, 130); // ~8fps for faster stabilization
  }, [captureFrame, sendFrame]);

  const stopFrameCapture = useCallback(() => {
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
  }, []);

  // Cleanup
  useEffect(() => {
    return () => stopFrameCapture();
  }, [stopFrameCapture]);

  // Handle Start Scan
  const handleStartScan = useCallback(async () => {
    setFinalResults(null);
    setVitals(null);
    setFace(null);
    setMotion(null);
    setSignalQuality(0);
    setElapsedTime(0);

    // Start camera
    try {
      await startCamera();

      // Connect WebSocket if needed
      if (!isConnected) {
        connect();
      }

      // Proactively poll for connection
      const checkAndStart = () => {
        if (isConnected) {
          startScan();
          startFrameCapture();
        } else {
          // If not connected yet, try again quickly (max 5 times)
          let attempts = 0;
          const poll = setInterval(() => {
            attempts++;
            if (isConnected || attempts > 20) {
              clearInterval(poll);
              if (isConnected) {
                startScan();
                startFrameCapture();
              }
            }
          }, 100);
        }
      };

      checkAndStart();
    } catch (e) {
      console.error("Failed to start scan:", e);
    }
  }, [startCamera, isConnected, connect, startScan, startFrameCapture]);

  // Handle Stop Scan
  const handleStopScan = useCallback(() => {
    stopScan();
    stopFrameCapture();
  }, [stopScan, stopFrameCapture]);

  // Handle New Scan
  const handleNewScan = useCallback(() => {
    setFinalResults(null);
    setVitals(null);
    setFace(null);
    setMotion(null);
    setSignalQuality(0);
    setElapsedTime(0);
    stopCamera();
  }, [stopCamera]);

  // Show Results Screen
  if (finalResults) {
    return (
      <ResultsScreen
        vitals={finalResults.vitals}
        signalQuality={finalResults.signalQuality}
        onNewScan={handleNewScan}
      />
    );
  }

  return (
    <div className="app">
      {/* Background Mesh */}
      <div className="app-bg-mesh" />

      {/* Header */}
      <header className="app-header animate-fade-in">
        <div className="header-left">
          <div className="header-logo">
            <span className="logo-icon">💓</span>
            <div className="logo-text">
              <h1>rPPG Health Monitor</h1>
              <span className="logo-tagline">Contactless Vital Signs</span>
            </div>
          </div>
        </div>
        <div className="header-right">
          <div className={`connection-status ${isConnected ? 'connected' : ''}`}>
            <span className="connection-dot" />
            <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="app-main">
        {/* Left: Camera Section */}
        <section className="camera-section">
          <CameraView
            videoRef={videoRef}
            canvasRef={canvasRef}
            isStreaming={isStreaming}
            face={face}
            motion={motion}
            signalQuality={signalQuality}
            isScanning={isScanning}
            elapsedTime={elapsedTime}
          />

          {/* Controls */}
          <div className="scan-controls animate-fade-in-up delay-2">
            {!isScanning ? (
              <button
                className="btn-scan btn-start"
                onClick={handleStartScan}
                disabled={scanState === 'connecting'}
              >
                <span className="btn-scan-icon">
                  {scanState === 'connecting' ? (
                    <span className="spinner" />
                  ) : (
                    '▶'
                  )}
                </span>
                <span>
                  {scanState === 'connecting' ? 'Connecting...' : 'Start Scan'}
                </span>
              </button>
            ) : (
              <button className="btn-scan btn-stop" onClick={handleStopScan}>
                <span className="btn-scan-icon">⏹</span>
                <span>Stop Scan</span>
              </button>
            )}
          </div>

          {/* Status Feedback */}
          {isScanning && (
            <div className="scan-status-feedback animate-fade-in">
              {!face?.detected ? (
                <span className="status-warning">⚠️ Face lost. Please look at camera.</span>
              ) : signalQuality < 0.3 ? (
                <span className="status-info">🔄 Stabilizing signal (stay still)...</span>
              ) : (
                <span className="status-success">✅ High quality signal. Keep still.</span>
              )}
            </div>
          )}

          {/* Error Messages */}
          {(wsError || cameraError) && (
            <div className="error-banner animate-fade-in">
              <span className="error-icon">⚠️</span>
              <span>{wsError || cameraError}</span>
            </div>
          )}

          {/* Instructions */}
          {!isScanning && !isStreaming && (
            <div className="instructions animate-fade-in-up delay-3">
              <h3>How to use</h3>
              <div className="instruction-steps">
                <div className="instruction-step">
                  <span className="step-number">1</span>
                  <span>Position your face in the camera view</span>
                </div>
                <div className="instruction-step">
                  <span className="step-number">2</span>
                  <span>Click "Start Scan" and hold still</span>
                </div>
                <div className="instruction-step">
                  <span className="step-number">3</span>
                  <span>Wait for 60 seconds for full analysis</span>
                </div>
              </div>
            </div>
          )}
        </section>

        {/* Right: Vitals Panel */}
        <section className="vitals-section animate-fade-in-up delay-1">
          <VitalsPanel vitals={vitals} isScanning={isScanning} />
        </section>
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p>
          rPPG Health Monitor — Remote Photoplethysmography Analysis System
        </p>
        <p className="footer-disclaimer">
          For informational purposes only. Not a medical device.
        </p>
      </footer>
    </div>
  );
}

export default App;
