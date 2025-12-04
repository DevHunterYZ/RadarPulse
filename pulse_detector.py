import numpy as np
from scipy import signal as sp_signal

class PulseDetector:
    #Radar pulse tespit ve analiz sınıfı
    
    def __init__(self, fs=1e6):
        """
        Args:
            fs: Örnekleme frekansı (Hz)
        """
        self.fs = fs
        
    def calculate_snr(self, signal_segment, noise_segment):
        """
        Signal-to-Noise Ratio hesaplar
        
        Args:
            signal_segment: Sinyal içeren bölüm
            noise_segment: Sadece gürültü içeren bölüm
            
        Returns:
            snr_db: SNR değeri (dB cinsinden)
        """
        signal_power = np.mean(np.abs(signal_segment)**2)
        noise_power = np.mean(np.abs(noise_segment)**2)
        
        if noise_power == 0:
            return float('inf')
        
        snr_linear = signal_power / noise_power
        snr_db = 10 * np.log10(snr_linear)
        
        return snr_db
    
    def detect_pulses(self, signal, threshold_factor=3.0, min_pulse_samples=10):
        """
        Sinyalde pulse'ları tespit eder
        
        Args:
            signal: Giriş sinyali (kompleks)
            threshold_factor: Eşik çarpanı (standart sapmanın kaç katı)
            min_pulse_samples: Minimum pulse uzunluğu (örnek sayısı)
            
        Returns:
            detections: Tespit edilen pulse'ların listesi
                Her bir tespit: {
                    'start_idx': başlangıç indeksi,
                    'end_idx': bitiş indeksi,
                    'duration': süre (saniye),
                    'peak_amplitude': maksimum genlik,
                    'snr_db': tahmini SNR
                }
        """
        # Zarfı hesapla (magnitude)
        envelope = np.abs(signal)
        
        # Gürültü seviyesini tahmin et (sinyal olmayan bölgelerden)
        # Basit yöntem: en düşük %30'luk bölümün ortalaması
        sorted_envelope = np.sort(envelope)
        noise_level = np.mean(sorted_envelope[:int(len(sorted_envelope) * 0.3)])
        noise_std = np.std(sorted_envelope[:int(len(sorted_envelope) * 0.3)])
        
        # Eşik değeri
        threshold = noise_level + threshold_factor * noise_std
        
        # Eşik üzerindeki bölgeleri bul
        above_threshold = envelope > threshold
        
        # Pulse başlangıç ve bitişlerini bul
        diff = np.diff(above_threshold.astype(int))
        pulse_starts = np.where(diff == 1)[0] + 1
        pulse_ends = np.where(diff == -1)[0] + 1
        
        # Eğer sinyal eşiğin üzerinde başlıyorsa
        if above_threshold[0]:
            pulse_starts = np.insert(pulse_starts, 0, 0)
        
        # Eğer sinyal eşiğin üzerinde bitiyorsa
        if above_threshold[-1]:
            pulse_ends = np.append(pulse_ends, len(signal))
        
        # Pulse'ları analiz et
        detections = []
        for start, end in zip(pulse_starts, pulse_ends):
            pulse_length = end - start
            
            # Çok kısa pulse'ları filtrele
            if pulse_length < min_pulse_samples:
                continue
            
            # Pulse özellikleri
            pulse_segment = signal[start:end]
            peak_amplitude = np.max(np.abs(pulse_segment))
            duration = pulse_length / self.fs
            
            # SNR tahmini
            # Pulse öncesi ve sonrası gürültü bölgelerinden tahmin
            noise_before = max(0, start - 50)
            noise_after = min(len(signal), end + 50)
            
            if noise_before < start and noise_after > end:
                noise_samples = np.concatenate([
                    signal[noise_before:start],
                    signal[end:noise_after]
                ])
                snr_db = self.calculate_snr(pulse_segment, noise_samples)
            else:
                snr_db = None
            
            detection = {
                'start_idx': start,
                'end_idx': end,
                'duration': duration,
                'peak_amplitude': peak_amplitude,
                'snr_db': snr_db,
                'center_idx': (start + end) // 2
            }
            
            detections.append(detection)
        
        return detections, threshold, noise_level
    
    def calculate_prf(self, detections):
        """
        Pulse Repetition Frequency hesaplar
        
        Args:
            detections: Tespit edilen pulse listesi
            
        Returns:
            prf: Ortalama PRF (Hz)
            pri: Ortalama PRI (saniye)
        """
        if len(detections) < 2:
            return None, None
        
        # Pulse merkezleri arasındaki zamanları hesapla
        centers = [d['center_idx'] for d in detections]
        time_diffs = np.diff(centers) / self.fs
        
        # Ortalama PRI
        pri = np.mean(time_diffs)
        
        # PRF
        prf = 1.0 / pri if pri > 0 else None
        
        return prf, pri
    
    def analyze_signal(self, signal, threshold_factor=3.0):
        """
        Sinyali tam analiz eder
        
        Args:
            signal: Giriş sinyali
            threshold_factor: Tespit eşiği
            
        Returns:
            results: Analiz sonuçları sözlüğü
        """
        # Pulse tespiti
        detections, threshold, noise_level = self.detect_pulses(
            signal, 
            threshold_factor=threshold_factor
        )
        
        # PRF hesapla
        prf, pri = self.calculate_prf(detections)
        
        # Ortalama SNR
        snr_values = [d['snr_db'] for d in detections if d['snr_db'] is not None]
        avg_snr = np.mean(snr_values) if snr_values else None
        
        # Ortalama pulse genişliği
        avg_pulse_width = np.mean([d['duration'] for d in detections]) if detections else None
        
        results = {
            'num_pulses': len(detections),
            'detections': detections,
            'prf': prf,
            'pri': pri,
            'avg_snr_db': avg_snr,
            'avg_pulse_width': avg_pulse_width,
            'threshold': threshold,
            'noise_level': noise_level
        }
        
        return results
    
    def generate_report(self, results):
        """
        Analiz sonuçlarını metin raporu olarak döner
        
        Args:
            results: analyze_signal() çıktısı
            
        Returns:
            report: Metin raporu
        """
        report = "=" * 50 + "\n"
        report += "RADAR PULSE ANALİZ RAPORU\n"
        report += "=" * 50 + "\n\n"
        
        report += f"Tespit Edilen Pulse Sayısı: {results['num_pulses']}\n\n"
        
        if results['num_pulses'] > 0:
            if results['prf'] is not None:
                report += f"Pulse Repetition Frequency (PRF): {results['prf']:.2f} Hz\n"
                report += f"Pulse Repetition Interval (PRI): {results['pri']*1e3:.3f} ms\n\n"
            
            if results['avg_pulse_width'] is not None:
                report += f"Ortalama Pulse Genişliği: {results['avg_pulse_width']*1e6:.2f} µs\n\n"
            
            if results['avg_snr_db'] is not None:
                report += f"Ortalama SNR: {results['avg_snr_db']:.2f} dB\n\n"
            
            report += f"Gürültü Seviyesi: {results['noise_level']:.4f}\n"
            report += f"Tespit Eşiği: {results['threshold']:.4f}\n\n"
            
            report += "Tespit Edilen Pulse'lar:\n"
            report += "-" * 50 + "\n"
            for i, det in enumerate(results['detections'], 1):
                report += f"Pulse #{i}:\n"
                report += f"  Süre: {det['duration']*1e6:.2f} µs\n"
                report += f"  Genlik: {det['peak_amplitude']:.4f}\n"
                if det['snr_db'] is not None:
                    report += f"  SNR: {det['snr_db']:.2f} dB\n"
                report += "\n"
        else:
            report += "⚠️  Hiç pulse tespit edilemedi.\n"
            report += "   - Eşik değerini düşürmeyi deneyin\n"
            report += "   - Sinyal gücünü kontrol edin\n"
        
        return report


# Test kodu
if __name__ == "__main__":
    from radar_simulator import RadarSimulator
    
    # Sinyal üret
    sim = RadarSimulator(fs=1e6)
    t, clean_signal = sim.generate_pulse_train(num_pulses=5, prf=1000)
    noisy_signal, _ = sim.add_noise(clean_signal, snr_db=10)
    
    # Tespit sistemi
    detector = PulseDetector(fs=1e6)
    results = detector.analyze_signal(noisy_signal, threshold_factor=3.0)
    
    # Rapor
    report = detector.generate_report(results)
    print(report)
    
    print("✅ Pulse tespit sistemi test edildi!")
