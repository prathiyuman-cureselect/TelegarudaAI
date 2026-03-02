// ──────────────────────────────────────────────
// rPPG Health Monitor - TypeScript Type Definitions
// ──────────────────────────────────────────────

export interface VitalSigns {
    status: 'measuring' | 'active' | 'completed' | 'idle';
    progress: number;
    frame_count: number;
    buffer_fullness: number;
    heart_rate: number;
    blood_pressure: BloodPressure;
    spo2: number;
    respiration_rate: number;
    temperature: number;
    hrv: HRVMetrics;
    stress_index: number;
    perfusion_index: number;
    measurement_time: number;
}

export interface BloodPressure {
    systolic: number;
    diastolic: number;
}

export interface HRVMetrics {
    sdnn: number;
    rmssd: number;
    pnn50: number;
}

export interface FaceDetection {
    detected: boolean;
    bbox: BoundingBox | null;
    confidence: number;
    identity: FaceIdentity | null;
}

export interface BoundingBox {
    x: number;
    y: number;
    width: number;
    height: number;
}

export interface FaceIdentity {
    name: string;
    confidence: number;
}

export interface MotionData {
    score: number;
    level: 'low' | 'moderate' | 'high' | 'excessive' | 'unknown';
    is_stable: boolean;
}

export interface LuminanceStats {
    reference_luminance: number | null;
    running_mean_luminance: number | null;
}

export interface ScanUpdate {
    type: 'update' | 'results' | 'status' | 'error' | 'pong';
    timestamp?: number;
    face?: FaceDetection;
    motion?: MotionData;
    vitals?: VitalSigns;
    signal_quality?: number;
    luminance?: LuminanceStats;
    elapsed_time?: number;
    status?: string;
    message?: string;
}

export interface ScanResults {
    vitals: VitalSigns;
    motion_stats: {
        average_motion: number;
        motion_quality: number;
        should_skip: boolean;
        buffer_size: number;
        landmark_stability_avg: number;
    };
    signal_quality: number;
}

export type ScanState = 'idle' | 'connecting' | 'scanning' | 'completed' | 'error';

export interface VitalCardConfig {
    id: string;
    label: string;
    unit: string;
    icon: string;
    color: string;
    gradient: string;
    getValue: (vitals: VitalSigns) => string;
    getStatus: (vitals: VitalSigns) => 'normal' | 'warning' | 'critical';
    normalRange: string;
}
