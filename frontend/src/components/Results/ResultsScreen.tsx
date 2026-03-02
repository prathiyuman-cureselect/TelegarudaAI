import { memo } from 'react';
import type { VitalSigns } from '../../types';
import './ResultsScreen.css';

interface ResultsScreenProps {
    vitals: VitalSigns;
    signalQuality: number;
    onNewScan: () => void;
}

const ResultsScreen = memo(function ResultsScreen({
    vitals,
    signalQuality,
    onNewScan,
}: ResultsScreenProps) {
    const getHealthScore = (): number => {
        let score = 100;
        // HR penalty
        if (vitals.heart_rate > 0) {
            if (vitals.heart_rate < 60 || vitals.heart_rate > 100) score -= 10;
            if (vitals.heart_rate < 50 || vitals.heart_rate > 120) score -= 15;
        }
        // BP penalty
        if (vitals.blood_pressure.systolic > 0) {
            if (vitals.blood_pressure.systolic > 140 || vitals.blood_pressure.systolic < 90) score -= 10;
        }
        // SpO2 penalty
        if (vitals.spo2 > 0 && vitals.spo2 < 95) score -= 15;
        // Stress penalty
        if (vitals.stress_index > 60) score -= 10;
        if (vitals.stress_index > 80) score -= 10;
        // Temperature penalty
        if (vitals.temperature > 37.5) score -= 10;

        return Math.max(0, Math.min(100, score));
    };

    const healthScore = getHealthScore();
    const scoreColor =
        healthScore >= 80 ? 'var(--accent-emerald)' :
            healthScore >= 60 ? 'var(--accent-amber)' :
                'var(--accent-rose)';

    const scoreLabel =
        healthScore >= 80 ? 'Excellent' :
            healthScore >= 60 ? 'Fair' :
                'Needs Attention';

    const circumference = 2 * Math.PI * 70;
    const strokeDashoffset = circumference - (healthScore / 100) * circumference;

    const results = [
        {
            icon: '❤️',
            label: 'Heart Rate',
            value: vitals.heart_rate > 0 ? `${vitals.heart_rate}` : '--',
            unit: 'BPM',
            color: 'var(--accent-rose)',
            range: '60-100 BPM',
        },
        {
            icon: '🩸',
            label: 'Blood Pressure',
            value: vitals.blood_pressure.systolic > 0
                ? `${Math.round(vitals.blood_pressure.systolic)}/${Math.round(vitals.blood_pressure.diastolic)}`
                : '--/--',
            unit: 'mmHg',
            color: 'var(--accent-violet)',
            range: '90-120/60-80',
        },
        {
            icon: '🫁',
            label: 'SpO₂',
            value: vitals.spo2 > 0 ? `${vitals.spo2}` : '--',
            unit: '%',
            color: 'var(--accent-cyan)',
            range: '95-100%',
        },
        {
            icon: '🌬️',
            label: 'Respiration Rate',
            value: vitals.respiration_rate > 0 ? `${vitals.respiration_rate}` : '--',
            unit: 'br/min',
            color: 'var(--accent-emerald)',
            range: '12-20 br/min',
        },
        {
            icon: '🌡️',
            label: 'Temperature',
            value: vitals.temperature > 0 ? `${vitals.temperature}` : '--',
            unit: '°C',
            color: 'var(--accent-amber)',
            range: '36.1-37.2°C',
        },
        {
            icon: '🧠',
            label: 'Stress Index',
            value: vitals.stress_index > 0 ? `${vitals.stress_index}` : '--',
            unit: '/100',
            color: 'var(--accent-pink)',
            range: '0-50 (low)',
        },
    ];

    return (
        <div className="results-screen">
            {/* Background Effects */}
            <div className="results-bg-effects">
                <div className="results-orb results-orb-1" />
                <div className="results-orb results-orb-2" />
                <div className="results-orb results-orb-3" />
            </div>

            {/* Header */}
            <div className="results-header animate-fade-in-up">
                <h1 className="results-title">Scan Complete</h1>
                <p className="results-subtitle">
                    {vitals.measurement_time > 0
                        ? `Measured over ${Math.round(vitals.measurement_time)} seconds`
                        : 'Your health overview is ready'}
                </p>
            </div>

            {/* Health Score Ring */}
            <div className="results-score-section animate-fade-in-up delay-1">
                <div className="score-ring-container">
                    <svg className="score-ring" viewBox="0 0 160 160">
                        {/* Background */}
                        <circle
                            cx="80"
                            cy="80"
                            r="70"
                            fill="none"
                            stroke="rgba(255,255,255,0.05)"
                            strokeWidth="6"
                        />
                        {/* Score arc */}
                        <circle
                            cx="80"
                            cy="80"
                            r="70"
                            fill="none"
                            stroke={scoreColor}
                            strokeWidth="6"
                            strokeLinecap="round"
                            strokeDasharray={circumference}
                            strokeDashoffset={strokeDashoffset}
                            transform="rotate(-90 80 80)"
                            className="score-ring-progress"
                            style={{ filter: `drop-shadow(0 0 6px ${scoreColor})` }}
                        />
                    </svg>
                    <div className="score-content">
                        <span className="score-value" style={{ color: scoreColor }}>
                            {healthScore}
                        </span>
                        <span className="score-label">{scoreLabel}</span>
                    </div>
                </div>

                <div className="score-quality">
                    <span>Signal Quality:</span>
                    <span className="score-quality-value">{Math.round(signalQuality * 100)}%</span>
                </div>
            </div>

            {/* Vitals Grid */}
            <div className="results-vitals-grid">
                {results.map((item, i) => (
                    <div
                        key={item.label}
                        className={`results-vital-card animate-fade-in-up delay-${i + 2}`}
                    >
                        <div className="results-vital-icon">{item.icon}</div>
                        <div className="results-vital-info">
                            <span className="results-vital-label">{item.label}</span>
                            <div className="results-vital-value-row">
                                <span className="results-vital-value" style={{ color: item.color }}>
                                    {item.value}
                                </span>
                                <span className="results-vital-unit">{item.unit}</span>
                            </div>
                            <span className="results-vital-range">{item.range}</span>
                        </div>
                    </div>
                ))}
            </div>

            {/* HRV Details */}
            {vitals.hrv.sdnn > 0 && (
                <div className="results-hrv animate-fade-in-up delay-8">
                    <h3>💓 Heart Rate Variability</h3>
                    <div className="results-hrv-grid">
                        <div className="results-hrv-item">
                            <span className="results-hrv-value">{vitals.hrv.sdnn.toFixed(1)}</span>
                            <span className="results-hrv-label">SDNN (ms)</span>
                        </div>
                        <div className="results-hrv-item">
                            <span className="results-hrv-value">{vitals.hrv.rmssd.toFixed(1)}</span>
                            <span className="results-hrv-label">RMSSD (ms)</span>
                        </div>
                        <div className="results-hrv-item">
                            <span className="results-hrv-value">{vitals.hrv.pnn50.toFixed(1)}%</span>
                            <span className="results-hrv-label">pNN50</span>
                        </div>
                        <div className="results-hrv-item">
                            <span className="results-hrv-value">{vitals.perfusion_index}%</span>
                            <span className="results-hrv-label">Perfusion Index</span>
                        </div>
                    </div>
                </div>
            )}

            {/* New Scan Button */}
            <div className="results-actions animate-fade-in-up delay-8">
                <button className="btn-new-scan" onClick={onNewScan}>
                    <span className="btn-icon">🔄</span>
                    Start New Scan
                </button>
            </div>

            {/* Disclaimer */}
            <p className="results-disclaimer animate-fade-in delay-8">
                ⚠️ These readings are estimations from remote photoplethysmography (rPPG)
                and are not medically validated. Consult a healthcare professional for accurate diagnostics.
            </p>
        </div>
    );
});

export default ResultsScreen;
