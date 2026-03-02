"""
rPPG (Remote Photoplethysmography) Processor Module
Extracts physiological signals from facial video:
- Heart Rate (HR)
- Blood Pressure (BP) estimation
- SpO2 (Blood Oxygen Saturation)
- Respiration Rate (RR)
- Body Temperature estimation
- HRV (Heart Rate Variability)
- Stress Index
- Perfusion Index

Uses chrominance-based rPPG (CHROM) method for robust signal extraction.
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from collections import deque
from typing import Dict, Optional, Tuple, List
import time


class RPPGProcessor:
    """Core rPPG signal processing engine."""

    def __init__(self, fps: float = 30.0, buffer_seconds: float = 10.0):
        self.fps = fps
        self.buffer_size = int(fps * buffer_seconds)
        
        # Color channel buffers (forehead + cheek ROIs)
        self._r_buffer: deque = deque(maxlen=self.buffer_size)
        self._g_buffer: deque = deque(maxlen=self.buffer_size)
        self._b_buffer: deque = deque(maxlen=self.buffer_size)
        
        # Timestamps
        self._timestamps: deque = deque(maxlen=self.buffer_size)
        
        # Derived signal buffers
        self._rppg_signal: deque = deque(maxlen=self.buffer_size)
        self._peak_intervals: deque = deque(maxlen=60)
        
        # Results cache
        self._last_vitals: Dict = {}
        self._measurement_start_time: Optional[float] = None
        self._frame_count = 0
        
        # Bandpass filter for heart rate (0.7 Hz - 3.5 Hz = 42-210 BPM)
        self._hr_low = 0.7
        self._hr_high = 3.5
        
        # Bandpass filter for respiration (0.1 Hz - 0.5 Hz = 6-30 breaths/min)
        self._rr_low = 0.1
        self._rr_high = 0.5

    def reset(self):
        """Reset all buffers for a new measurement session."""
        self._r_buffer.clear()
        self._g_buffer.clear()
        self._b_buffer.clear()
        self._timestamps.clear()
        self._rppg_signal.clear()
        self._peak_intervals.clear()
        self._last_vitals = {}
        self._measurement_start_time = None
        self._frame_count = 0

    def add_frame_roi(self, roi_pixels: np.ndarray, timestamp: Optional[float] = None):
        """
        Add ROI pixel data from a face region.
        roi_pixels: BGR pixel array from the face ROI (forehead/cheek region)
        """
        if roi_pixels is None or roi_pixels.size == 0:
            return

        if timestamp is None:
            timestamp = time.time()

        if self._measurement_start_time is None:
            self._measurement_start_time = timestamp

        # Calculate spatial mean of each channel
        mean_b = float(np.mean(roi_pixels[:, :, 0]))
        mean_g = float(np.mean(roi_pixels[:, :, 1]))
        mean_r = float(np.mean(roi_pixels[:, :, 2]))

        self._b_buffer.append(mean_b)
        self._g_buffer.append(mean_g)
        self._r_buffer.append(mean_r)
        self._timestamps.append(timestamp)
        self._frame_count += 1

        # Extract rPPG signal using CHROM method
        if len(self._r_buffer) >= 30:
            rppg_val = self._chrom_extract()
            self._rppg_signal.append(rppg_val)

    def _chrom_extract(self) -> float:
        """
        CHROMiance-based rPPG extraction.
        Based on: de Haan & Jeanne (2013) - Robust Pulse Rate from Chrominance-Based rPPG
        """
        r = np.array(list(self._r_buffer))
        g = np.array(list(self._g_buffer))
        b = np.array(list(self._b_buffer))

        # Normalize by temporal mean
        r_norm = r / (np.mean(r) + 1e-8)
        g_norm = g / (np.mean(g) + 1e-8)
        b_norm = b / (np.mean(b) + 1e-8)

        # CHROM method: X = 3R - 2G, Y = 1.5R + G - 1.5B
        x_chrom = 3.0 * r_norm - 2.0 * g_norm
        y_chrom = 1.5 * r_norm + g_norm - 1.5 * b_norm

        # Combine using alpha
        std_x = np.std(x_chrom) + 1e-8
        std_y = np.std(y_chrom) + 1e-8
        alpha = std_x / std_y

        rppg = x_chrom - alpha * y_chrom
        
        # Return the latest value
        return float(rppg[-1])

    def _bandpass_filter(self, data: np.ndarray, low: float, high: float) -> np.ndarray:
        """Apply a Butterworth bandpass filter."""
        if len(data) < 30:
            return data
        
        nyquist = self.fps / 2.0
        low_norm = max(low / nyquist, 0.01)
        high_norm = min(high / nyquist, 0.99)
        
        if low_norm >= high_norm:
            return data

        try:
            b, a = signal.butter(3, [low_norm, high_norm], btype='band')
            filtered = signal.filtfilt(b, a, data, padlen=min(3 * max(len(b), len(a)), len(data) - 1))
            return filtered
        except Exception:
            return data

    def _compute_heart_rate(self) -> float:
        """Compute heart rate from rPPG signal using FFT peak detection."""
        if len(self._rppg_signal) < 60:
            return 0.0

        sig = np.array(list(self._rppg_signal))
        
        # Bandpass filter for HR
        filtered = self._bandpass_filter(sig, self._hr_low, self._hr_high)
        
        # Zero-pad for better frequency resolution
        n = len(filtered)
        pad_length = max(512, 2 ** int(np.ceil(np.log2(n)) + 1))
        padded = np.zeros(pad_length)
        padded[:n] = filtered * np.hanning(n)

        # FFT
        spectrum = np.abs(fft(padded))
        freqs = fftfreq(pad_length, 1.0 / self.fps)

        # Only look at positive frequencies in HR range
        mask = (freqs >= self._hr_low) & (freqs <= self._hr_high)
        if not np.any(mask):
            return 0.0

        hr_spectrum = spectrum[mask]
        hr_freqs = freqs[mask]

        # Find dominant frequency
        peak_idx = np.argmax(hr_spectrum)
        peak_freq = hr_freqs[peak_idx]
        
        heart_rate = peak_freq * 60.0  # Convert Hz to BPM
        return round(float(np.clip(heart_rate, 45, 200)), 1)

    def _compute_respiration_rate(self) -> float:
        """Compute respiration rate from low-frequency rPPG components."""
        if len(self._rppg_signal) < 90:
            return 0.0

        sig = np.array(list(self._rppg_signal))
        filtered = self._bandpass_filter(sig, self._rr_low, self._rr_high)

        n = len(filtered)
        pad_length = max(512, 2 ** int(np.ceil(np.log2(n)) + 1))
        padded = np.zeros(pad_length)
        padded[:n] = filtered * np.hanning(n)

        spectrum = np.abs(fft(padded))
        freqs = fftfreq(pad_length, 1.0 / self.fps)

        mask = (freqs >= self._rr_low) & (freqs <= self._rr_high)
        if not np.any(mask):
            return 0.0

        rr_spectrum = spectrum[mask]
        rr_freqs = freqs[mask]

        peak_idx = np.argmax(rr_spectrum)
        peak_freq = rr_freqs[peak_idx]

        resp_rate = peak_freq * 60.0  # Convert Hz to breaths/min
        return round(float(np.clip(resp_rate, 8, 40)), 1)

    def _compute_spo2(self) -> float:
        """
        Estimate SpO2 using the ratio of ratios (R/IR approximation).
        Uses red and blue channels as proxies for red and infrared light.
        """
        if len(self._r_buffer) < 60:
            return 0.0

        r = np.array(list(self._r_buffer))
        b = np.array(list(self._b_buffer))

        # AC/DC ratio for each channel
        r_ac = np.std(r)
        r_dc = np.mean(r) + 1e-8
        b_ac = np.std(b)
        b_dc = np.mean(b) + 1e-8

        ratio = (r_ac / r_dc) / (b_ac / b_dc + 1e-8)

        # Empirical calibration: SpO2 ≈ 110 - 25 * ratio
        spo2 = 110.0 - 25.0 * ratio
        return round(float(np.clip(spo2, 85, 100)), 1)

    def _compute_blood_pressure(self, heart_rate: float) -> Tuple[float, float]:
        """
        Estimate blood pressure using Pulse Transit Time (PTT) approximation.
        Uses rPPG waveform morphology and heart rate correlation.
        """
        if heart_rate <= 0 or len(self._rppg_signal) < 90:
            return (0.0, 0.0)

        sig = np.array(list(self._rppg_signal))
        filtered = self._bandpass_filter(sig, self._hr_low, self._hr_high)

        # Peak detection for pulse wave analysis
        peaks, properties = signal.find_peaks(
            filtered, distance=int(self.fps * 0.4),
            prominence=0.01
        )

        if len(peaks) < 3:
            return (0.0, 0.0)

        # Analyze pulse wave intervals
        intervals = np.diff(peaks) / self.fps
        mean_interval = np.mean(intervals)

        # PTT-based BP estimation (simplified model)
        # Systolic: correlates with 1/PTT and HR
        # Diastolic: correlates with vascular resistance proxy
        ptt_proxy = mean_interval  # Simplified PTT
        
        systolic = 100.0 + (heart_rate - 70) * 0.3 + (1.0 / (ptt_proxy + 0.1)) * 5.0
        diastolic = 65.0 + (heart_rate - 70) * 0.15 + (1.0 / (ptt_proxy + 0.1)) * 2.5

        systolic = float(np.clip(systolic, 85, 180))
        diastolic = float(np.clip(diastolic, 50, 110))

        return (round(systolic, 0), round(diastolic, 0))

    def _compute_temperature(self) -> float:
        """
        Estimate skin temperature from color channel ratios.
        Uses the ratio of red to blue channel intensities as a thermal proxy.
        """
        if len(self._r_buffer) < 60:
            return 0.0

        r = np.array(list(self._r_buffer))
        g = np.array(list(self._g_buffer))
        b = np.array(list(self._b_buffer))

        # Color temperature proxy
        r_mean = np.mean(r)
        b_mean = np.mean(b) + 1e-8
        g_mean = np.mean(g)

        # Warm-to-cool ratio
        warmth_ratio = r_mean / b_mean

        # Empirical model for skin temperature estimation
        # Normal skin temp ~36.5°C, range 35-38°C
        temp = 35.0 + warmth_ratio * 0.8 + (g_mean / 255.0) * 0.5
        temp = float(np.clip(temp, 35.0, 38.5))

        return round(temp, 1)

    def _compute_hrv(self, heart_rate: float) -> Dict:
        """
        Compute Heart Rate Variability metrics.
        Returns SDNN, RMSSD, and pNN50.
        """
        if len(self._rppg_signal) < 120:
            return {"sdnn": 0.0, "rmssd": 0.0, "pnn50": 0.0}

        sig = np.array(list(self._rppg_signal))
        filtered = self._bandpass_filter(sig, self._hr_low, self._hr_high)

        peaks, _ = signal.find_peaks(
            filtered, distance=int(self.fps * 0.4),
            prominence=0.01
        )

        if len(peaks) < 5:
            return {"sdnn": 0.0, "rmssd": 0.0, "pnn50": 0.0}

        # RR intervals in milliseconds
        rr_intervals = np.diff(peaks) / self.fps * 1000.0

        # SDNN: Standard deviation of NN intervals
        sdnn = float(np.std(rr_intervals))

        # RMSSD: Root mean square of successive differences
        successive_diffs = np.diff(rr_intervals)
        rmssd = float(np.sqrt(np.mean(successive_diffs ** 2)))

        # pNN50: Percentage of successive intervals differing by more than 50ms
        pnn50 = float(np.sum(np.abs(successive_diffs) > 50) / len(successive_diffs) * 100)

        return {
            "sdnn": round(sdnn, 1),
            "rmssd": round(rmssd, 1),
            "pnn50": round(pnn50, 1),
        }

    def _compute_stress_index(self, hrv: Dict, heart_rate: float) -> float:
        """Compute stress index from HRV metrics and heart rate."""
        if hrv["sdnn"] <= 0 or heart_rate <= 0:
            return 0.0

        # Baevsky's stress index approximation
        # Higher HR + Lower HRV = Higher stress
        stress = (heart_rate / 70.0) * (50.0 / (hrv["sdnn"] + 1.0)) * 50.0
        return round(float(np.clip(stress, 0, 100)), 1)

    def _compute_perfusion_index(self) -> float:
        """Compute Perfusion Index from pulsatile/non-pulsatile ratio."""
        if len(self._g_buffer) < 60:
            return 0.0

        g = np.array(list(self._g_buffer))
        ac = np.std(g)
        dc = np.mean(g) + 1e-8
        pi = (ac / dc) * 100.0
        return round(float(np.clip(pi, 0.1, 20.0)), 2)

    def get_vitals(self) -> Dict:
        """
        Compute and return all vital signs.
        Returns a comprehensive dictionary of all estimated physiological parameters.
        """
        min_frames = 60  # Minimum frames needed for any measurement

        if self._frame_count < min_frames:
            progress = min(100, int((self._frame_count / min_frames) * 100))
            return {
                "status": "measuring",
                "progress": progress,
                "frame_count": self._frame_count,
                "buffer_fullness": len(self._rppg_signal),
                "heart_rate": 0,
                "blood_pressure": {"systolic": 0, "diastolic": 0},
                "spo2": 0,
                "respiration_rate": 0,
                "temperature": 0,
                "hrv": {"sdnn": 0, "rmssd": 0, "pnn50": 0},
                "stress_index": 0,
                "perfusion_index": 0,
                "measurement_time": 0,
            }

        # Compute all vitals
        heart_rate = self._compute_heart_rate()
        respiration_rate = self._compute_respiration_rate()
        spo2 = self._compute_spo2()
        bp_sys, bp_dia = self._compute_blood_pressure(heart_rate)
        temperature = self._compute_temperature()
        hrv = self._compute_hrv(heart_rate)
        stress_index = self._compute_stress_index(hrv, heart_rate)
        perfusion_index = self._compute_perfusion_index()

        elapsed = 0.0
        if self._measurement_start_time and self._timestamps:
            elapsed = self._timestamps[-1] - self._measurement_start_time

        self._last_vitals = {
            "status": "active",
            "progress": 100,
            "frame_count": self._frame_count,
            "buffer_fullness": len(self._rppg_signal),
            "heart_rate": heart_rate,
            "blood_pressure": {"systolic": bp_sys, "diastolic": bp_dia},
            "spo2": spo2,
            "respiration_rate": respiration_rate,
            "temperature": temperature,
            "hrv": hrv,
            "stress_index": stress_index,
            "perfusion_index": perfusion_index,
            "measurement_time": round(elapsed, 1),
        }

        return self._last_vitals

    def get_signal_quality(self) -> float:
        """
        Assess the quality of the current rPPG signal (0-1).
        Based on signal-to-noise ratio and periodicity.
        """
        if len(self._rppg_signal) < 60:
            return 0.0

        sig = np.array(list(self._rppg_signal))
        filtered = self._bandpass_filter(sig, self._hr_low, self._hr_high)

        # SNR estimation
        signal_power = np.var(filtered)
        noise = sig - filtered
        noise_power = np.var(noise) + 1e-8

        snr = signal_power / noise_power
        quality = min(1.0, snr / 5.0)  # Normalize, SNR of 5 = perfect quality

        return round(quality, 3)
