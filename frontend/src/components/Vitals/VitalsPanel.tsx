import { memo, useMemo } from 'react';
import type { VitalSigns, VitalCardConfig } from '../../types';
import './VitalsPanel.css';

interface VitalsPanelProps {
    vitals: VitalSigns | null;
    isScanning: boolean;
}

const vitalConfigs: VitalCardConfig[] = [
    {
        id: 'heart_rate',
        label: 'Heart Rate',
        unit: 'BPM',
        icon: '❤️',
        color: 'var(--accent-rose)',
        gradient: 'var(--gradient-rose)',
        getValue: (v) => v.heart_rate > 0 ? `${v.heart_rate}` : '--',
        getStatus: (v) => {
            if (v.heart_rate === 0) return 'normal';
            if (v.heart_rate < 60 || v.heart_rate > 100) return 'warning';
            if (v.heart_rate < 45 || v.heart_rate > 150) return 'critical';
            return 'normal';
        },
        normalRange: '60-100',
    },
    {
        id: 'blood_pressure',
        label: 'Blood Pressure',
        unit: 'mmHg',
        icon: '🩸',
        color: 'var(--accent-violet)',
        gradient: 'var(--gradient-violet)',
        getValue: (v) =>
            v.blood_pressure.systolic > 0
                ? `${Math.round(v.blood_pressure.systolic)}/${Math.round(v.blood_pressure.diastolic)}`
                : '--/--',
        getStatus: (v) => {
            const s = v.blood_pressure.systolic;
            if (s === 0) return 'normal';
            if (s > 140 || s < 90) return 'warning';
            if (s > 180 || s < 80) return 'critical';
            return 'normal';
        },
        normalRange: '90-120/60-80',
    },
    {
        id: 'spo2',
        label: 'SpO₂',
        unit: '%',
        icon: '🫁',
        color: 'var(--accent-cyan)',
        gradient: 'var(--gradient-cyan)',
        getValue: (v) => v.spo2 > 0 ? `${v.spo2}` : '--',
        getStatus: (v) => {
            if (v.spo2 === 0) return 'normal';
            if (v.spo2 < 95) return 'warning';
            if (v.spo2 < 90) return 'critical';
            return 'normal';
        },
        normalRange: '95-100',
    },
    {
        id: 'respiration',
        label: 'Respiration',
        unit: 'br/min',
        icon: '🌬️',
        color: 'var(--accent-emerald)',
        gradient: 'var(--gradient-emerald)',
        getValue: (v) => v.respiration_rate > 0 ? `${v.respiration_rate}` : '--',
        getStatus: (v) => {
            if (v.respiration_rate === 0) return 'normal';
            if (v.respiration_rate < 12 || v.respiration_rate > 20) return 'warning';
            if (v.respiration_rate < 8 || v.respiration_rate > 30) return 'critical';
            return 'normal';
        },
        normalRange: '12-20',
    },
    {
        id: 'temperature',
        label: 'Temperature',
        unit: '°C',
        icon: '🌡️',
        color: 'var(--accent-amber)',
        gradient: 'var(--gradient-amber)',
        getValue: (v) => v.temperature > 0 ? `${v.temperature}` : '--',
        getStatus: (v) => {
            if (v.temperature === 0) return 'normal';
            if (v.temperature > 37.5 || v.temperature < 36.0) return 'warning';
            if (v.temperature > 38.5 || v.temperature < 35.0) return 'critical';
            return 'normal';
        },
        normalRange: '36.1-37.2',
    },
    {
        id: 'stress',
        label: 'Stress Index',
        unit: '',
        icon: '🧠',
        color: 'var(--accent-pink)',
        gradient: 'linear-gradient(135deg, #ec4899, #be185d)',
        getValue: (v) => v.stress_index > 0 ? `${v.stress_index}` : '--',
        getStatus: (v) => {
            if (v.stress_index === 0) return 'normal';
            if (v.stress_index > 60) return 'warning';
            if (v.stress_index > 80) return 'critical';
            return 'normal';
        },
        normalRange: '0-50',
    },
];

const VitalsPanel = memo(function VitalsPanel({ vitals, isScanning }: VitalsPanelProps) {
    const progress = vitals?.progress ?? 0;

    return (
        <div className="vitals-panel">
            <div className="vitals-header">
                <h2 className="vitals-title">
                    <span className="vitals-title-icon">📊</span>
                    Live Vitals
                </h2>
                {isScanning && (
                    <div className="vitals-progress">
                        <div className="vitals-progress-bar">
                            <div
                                className="vitals-progress-fill"
                                style={{ width: `${progress}%` }}
                            />
                        </div>
                        <span className="vitals-progress-text">{progress}%</span>
                    </div>
                )}
            </div>

            <div className="vitals-grid">
                {vitalConfigs.map((config, index) => (
                    <VitalCard
                        key={config.id}
                        config={config}
                        vitals={vitals}
                        isScanning={isScanning}
                        index={index}
                    />
                ))}
            </div>

            {/* HRV Details */}
            {vitals && vitals.hrv.sdnn > 0 && (
                <div className="hrv-section animate-fade-in-up delay-6">
                    <h3 className="hrv-title">
                        <span>💓</span> HRV Metrics
                    </h3>
                    <div className="hrv-metrics">
                        <div className="hrv-metric">
                            <span className="hrv-metric-value">
                                {vitals.hrv.sdnn.toFixed(1)}
                            </span>
                            <span className="hrv-metric-label">SDNN (ms)</span>
                        </div>
                        <div className="hrv-divider" />
                        <div className="hrv-metric">
                            <span className="hrv-metric-value">
                                {vitals.hrv.rmssd.toFixed(1)}
                            </span>
                            <span className="hrv-metric-label">RMSSD (ms)</span>
                        </div>
                        <div className="hrv-divider" />
                        <div className="hrv-metric">
                            <span className="hrv-metric-value">
                                {vitals.hrv.pnn50.toFixed(1)}%
                            </span>
                            <span className="hrv-metric-label">pNN50</span>
                        </div>
                    </div>
                </div>
            )}

            {/* Perfusion Index */}
            {vitals && vitals.perfusion_index > 0 && (
                <div className="perfusion-card animate-fade-in-up delay-7">
                    <div className="perfusion-header">
                        <span>🔴</span>
                        <span>Perfusion Index</span>
                    </div>
                    <div className="perfusion-value">{vitals.perfusion_index}%</div>
                    <div className="perfusion-bar">
                        <div
                            className="perfusion-bar-fill"
                            style={{ width: `${Math.min(100, vitals.perfusion_index * 10)}%` }}
                        />
                    </div>
                </div>
            )}
        </div>
    );
});

/* ── Individual Vital Card ── */
interface VitalCardProps {
    config: VitalCardConfig;
    vitals: VitalSigns | null;
    isScanning: boolean;
    index: number;
}

const VitalCard = memo(function VitalCard({ config, vitals, isScanning, index }: VitalCardProps) {
    const value = useMemo(
        () => (vitals ? config.getValue(vitals) : '--'),
        [vitals, config]
    );

    const status = useMemo(
        () => (vitals ? config.getStatus(vitals) : 'normal'),
        [vitals, config]
    );

    const isActive = value !== '--' && value !== '--/--';

    return (
        <div
            className={`vital-card animate-fade-in-up delay-${index + 1} ${status} ${isActive ? 'has-value' : ''}`}
        >
            <div className="vital-card-glow" style={{ background: config.gradient, opacity: isActive ? 0.06 : 0 }} />

            <div className="vital-card-header">
                <span className="vital-card-icon">{config.icon}</span>
                <span className="vital-card-label">{config.label}</span>
            </div>

            <div className="vital-card-body">
                <span
                    className={`vital-card-value ${isScanning && !isActive ? 'measuring' : ''}`}
                    style={{ color: isActive ? config.color : undefined }}
                >
                    {value}
                </span>
                <span className="vital-card-unit">{config.unit}</span>
            </div>

            <div className="vital-card-footer">
                <span className="vital-card-range">Normal: {config.normalRange}</span>
                {isActive && (
                    <span className={`vital-card-status-dot ${status}`} />
                )}
            </div>

            {isScanning && !isActive && (
                <div className="vital-card-shimmer" />
            )}
        </div>
    );
});

export default VitalsPanel;
