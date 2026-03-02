import { useEffect, useRef, memo } from 'react';
import type { FaceDetection, MotionData } from '../../types';
import './CameraView.css';

interface CameraViewProps {
    videoRef: React.RefObject<HTMLVideoElement | null>;
    canvasRef: React.RefObject<HTMLCanvasElement | null>;
    isStreaming: boolean;
    face: FaceDetection | null;
    motion: MotionData | null;
    signalQuality: number;
    isScanning: boolean;
    elapsedTime: number;
}

const CameraView = memo(function CameraView({
    videoRef,
    canvasRef,
    isStreaming,
    face,
    motion,
    signalQuality,
    isScanning,
    elapsedTime,
}: CameraViewProps) {
    const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);

    // Draw face overlay
    useEffect(() => {
        const canvas = overlayCanvasRef.current;
        const video = videoRef.current;
        if (!canvas || !video) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (face?.detected && face.bbox) {
            const { x, y, width, height } = face.bbox;

            // Scale bbox to canvas
            const scaleX = canvas.width / (video.videoWidth || 640);
            const scaleY = canvas.height / (video.videoHeight || 480);
            const sx = x * scaleX;
            const sy = y * scaleY;
            const sw = width * scaleX;
            const sh = height * scaleY;

            // Glow color based on signal quality
            const glowColor = signalQuality > 0.5
                ? `rgba(52, 211, 153, ${0.4 + signalQuality * 0.4})`
                : signalQuality > 0.2
                    ? `rgba(251, 191, 36, ${0.4 + signalQuality * 0.4})`
                    : 'rgba(244, 63, 94, 0.6)';

            ctx.strokeStyle = glowColor;
            ctx.lineWidth = 2;
            ctx.shadowColor = glowColor;
            ctx.shadowBlur = 12;

            // Corner brackets
            const cornerLen = Math.min(sw, sh) * 0.15;
            const cornerRadius = 4;

            // Top-left
            ctx.beginPath();
            ctx.moveTo(sx, sy + cornerLen);
            ctx.lineTo(sx, sy + cornerRadius);
            ctx.arcTo(sx, sy, sx + cornerRadius, sy, cornerRadius);
            ctx.lineTo(sx + cornerLen, sy);
            ctx.stroke();

            // Top-right
            ctx.beginPath();
            ctx.moveTo(sx + sw - cornerLen, sy);
            ctx.lineTo(sx + sw - cornerRadius, sy);
            ctx.arcTo(sx + sw, sy, sx + sw, sy + cornerRadius, cornerRadius);
            ctx.lineTo(sx + sw, sy + cornerLen);
            ctx.stroke();

            // Bottom-left
            ctx.beginPath();
            ctx.moveTo(sx, sy + sh - cornerLen);
            ctx.lineTo(sx, sy + sh - cornerRadius);
            ctx.arcTo(sx, sy + sh, sx + cornerRadius, sy + sh, cornerRadius);
            ctx.lineTo(sx + cornerLen, sy + sh);
            ctx.stroke();

            // Bottom-right
            ctx.beginPath();
            ctx.moveTo(sx + sw - cornerLen, sy + sh);
            ctx.lineTo(sx + sw - cornerRadius, sy + sh);
            ctx.arcTo(sx + sw, sy + sh, sx + sw, sy + sh - cornerRadius, cornerRadius);
            ctx.lineTo(sx + sw, sy + sh - cornerLen);
            ctx.stroke();

            ctx.shadowBlur = 0;

            // ROI indicator (forehead region)
            if (isScanning) {
                const roiY = sy + sh * 0.08;
                const roiH = sh * 0.22;
                const roiX = sx + sw * 0.2;
                const roiW = sw * 0.6;

                ctx.fillStyle = 'rgba(59, 130, 246, 0.08)';
                ctx.strokeStyle = 'rgba(59, 130, 246, 0.3)';
                ctx.lineWidth = 1;
                ctx.setLineDash([4, 4]);
                ctx.beginPath();
                ctx.roundRect(roiX, roiY, roiW, roiH, 6);
                ctx.fill();
                ctx.stroke();
                ctx.setLineDash([]);

                // ROI Label
                ctx.fillStyle = 'rgba(59, 130, 246, 0.7)';
                ctx.font = '10px Inter, sans-serif';
                ctx.fillText('ROI', roiX + 4, roiY + 12);
            }
        }
    }, [face, signalQuality, isScanning, videoRef]);

    const getMotionIndicator = () => {
        if (!motion) return null;
        const colors: Record<string, string> = {
            low: 'var(--accent-emerald)',
            moderate: 'var(--accent-amber)',
            high: 'var(--accent-orange)',
            excessive: 'var(--accent-rose)',
            unknown: 'var(--text-tertiary)',
        };
        return colors[motion.level] || colors.unknown;
    };

    const formatTime = (seconds: number) => {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${m}:${s.toString().padStart(2, '0')}`;
    };

    return (
        <div className="camera-view">
            <div className="camera-viewport">
                <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="camera-video"
                />
                <canvas ref={canvasRef} className="camera-canvas-hidden" />
                <canvas ref={overlayCanvasRef} className="camera-overlay-canvas" />

                {/* Scanning animation */}
                {isScanning && (
                    <div className="scan-line-container">
                        <div className="scan-line" />
                    </div>
                )}

                {/* Camera not streaming */}
                {!isStreaming && (
                    <div className="camera-placeholder">
                        <div className="camera-placeholder-icon">📷</div>
                        <p>Camera not active</p>
                        <span>Click "Start Scan" to begin</span>
                    </div>
                )}

                {/* HUD Overlay */}
                {isStreaming && (
                    <div className="camera-hud">
                        {/* Top-left: Face Status */}
                        <div className="hud-top-left">
                            <div className={`hud-indicator ${face?.detected ? 'active' : 'inactive'}`}>
                                <span className="hud-dot" />
                                <span>{face?.detected ? 'Face Detected' : 'No Face'}</span>
                            </div>
                            {face?.identity && (
                                <div className="hud-identity">
                                    👤 {face.identity.name} ({Math.round(face.identity.confidence * 100)}%)
                                </div>
                            )}
                        </div>

                        {/* Top-right: Timer */}
                        {isScanning && (
                            <div className="hud-top-right">
                                <div className="hud-timer">
                                    <span className="hud-timer-dot" />
                                    <span className="hud-timer-text">{formatTime(elapsedTime)}</span>
                                    <span className="hud-timer-label">/ 1:00</span>
                                </div>
                            </div>
                        )}

                        {/* Bottom-left: Motion */}
                        {motion && (
                            <div className="hud-bottom-left">
                                <div className="hud-motion">
                                    <span className="hud-motion-bar">
                                        <span
                                            className="hud-motion-fill"
                                            style={{
                                                width: `${Math.min(100, motion.score * 20)}%`,
                                                backgroundColor: getMotionIndicator() || undefined,
                                            }}
                                        />
                                    </span>
                                    <span className="hud-motion-label">{motion.level}</span>
                                </div>
                            </div>
                        )}

                        {/* Bottom-right: Signal Quality */}
                        {isScanning && (
                            <div className="hud-bottom-right">
                                <div className="hud-signal">
                                    <div className="signal-bars">
                                        {[1, 2, 3, 4, 5].map(i => (
                                            <div
                                                key={i}
                                                className={`signal-bar ${signalQuality >= i * 0.2 ? 'active' : ''}`}
                                                style={{
                                                    height: `${i * 4 + 4}px`,
                                                    backgroundColor: signalQuality >= i * 0.2
                                                        ? signalQuality > 0.5
                                                            ? 'var(--accent-emerald)'
                                                            : 'var(--accent-amber)'
                                                        : undefined,
                                                }}
                                            />
                                        ))}
                                    </div>
                                    <span className="hud-signal-text">
                                        {Math.round(signalQuality * 100)}%
                                    </span>
                                </div>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
});

export default CameraView;
