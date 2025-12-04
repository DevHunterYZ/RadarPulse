import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class RadarSimulator:
    #Gerçekçi radar pulse sinyalleri üretir.
    
    def __init__(self, fs=1e6):
        """
        Args:
            fs: Örnekleme frekansı (Hz) - 1 MHz varsayılan
        """
        self.fs = fs
        
    def generate_pulse(self, duration=1e-6, amplitude=1.0, carrier_freq=10e6):
        """
        Tek bir radar pulse üretir
        
        Args:
            duration: Pulse süresi (saniye) - varsayılan 1 mikrosaniye
            amplitude: Genlik
            carrier_freq: Taşıyıcı frekans (Hz) - 10 MHz varsayılan
            
        Returns:
            t: Zaman vektörü
            pulse: Pulse sinyali
        """
        num_samples = int(duration * self.fs)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        
        # Dikdörtgen zarf
        envelope = amplitude * np.ones(num_samples)
        
        # Taşıyıcı sinyal (RF carrier)
        carrier = np.exp(1j * 2 * np.pi * carrier_freq * t)
        
        # Pulse = zarf * taşıyıcı
        pulse = envelope * carrier
        
        return t, pulse
    
    def generate_pulse_train(self, num_pulses=10, prf=1e3, pulse_width=1e-6, 
                            amplitude=1.0, carrier_freq=10e6):
        """
        Pulse treni (birden fazla pulse) üretir.
        
        Args:
            num_pulses: Pulse sayısı
            prf: Pulse Repetition Frequency (Hz) - 1 kHz varsayılan
            pulse_width: Her pulse'un genişliği (saniye)
            amplitude: Genlik
            carrier_freq: Taşıyıcı frekans
            
        Returns:
            t: Zaman vektörü
            signal: Pulse treni sinyali
        """
        pri = 1.0 / prf  # Pulse Repetition Interval
        total_duration = num_pulses * pri
        num_samples = int(total_duration * self.fs)
        
        t = np.linspace(0, total_duration, num_samples, endpoint=False)
        radar_signal = np.zeros(num_samples, dtype=complex)
        
        for i in range(num_pulses):
            pulse_start = int(i * pri * self.fs)
            pulse_samples = int(pulse_width * self.fs)
            pulse_end = pulse_start + pulse_samples
            
            if pulse_end <= num_samples:
                t_pulse = t[pulse_start:pulse_end] - t[pulse_start]
                carrier = np.exp(1j * 2 * np.pi * carrier_freq * t_pulse)
                radar_signal[pulse_start:pulse_end] = amplitude * carrier
        
        return t, radar_signal
    
    def add_noise(self, signal, snr_db=10):
        """
        Sinyale gürültü ekler
        
        Args:
            signal: Temiz sinyal
            snr_db: Signal-to-Noise Ratio (dB)
            
        Returns:
            noisy_signal: Gürültülü sinyal
            noise_power: Gürültü gücü
        """
        # Sinyal gücü
        signal_power = np.mean(np.abs(signal)**2)
        
        # Gürültü gücü hesapla
        snr_linear = 10**(snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # Kompleks Gaussian gürültü
        noise_real = np.random.normal(0, np.sqrt(noise_power/2), len(signal))
        noise_imag = np.random.normal(0, np.sqrt(noise_power/2), len(signal))
        noise = noise_real + 1j * noise_imag
        
        noisy_signal = signal + noise
        
        return noisy_signal, noise_power
    
    def generate_scenario(self, scenario_type='clean'):
        """
        Farklı senaryolar için sinyal üretir
        
        Args:
            scenario_type: 'clean', 'noisy', 'weak', 'multiple_targets'
            
        Returns:
            t: Zaman
            signal: Sinyal
            params: Parametreler
        """
        if scenario_type == 'clean':
            # Temiz, güçlü sinyal
            t, sig = self.generate_pulse_train(
                num_pulses=8, prf=1000, pulse_width=1e-6, 
                amplitude=1.0, carrier_freq=10e6
            )
            params = {'snr_db': None, 'description': 'Temiz radar sinyali'}
            
        elif scenario_type == 'noisy':
            # Gürültülü sinyal
            t, sig = self.generate_pulse_train(
                num_pulses=8, prf=1000, pulse_width=1e-6,
                amplitude=1.0, carrier_freq=10e6
            )
            sig, _ = self.add_noise(sig, snr_db=5)
            params = {'snr_db': 5, 'description': 'Gürültülü sinyal (SNR=5dB)'}
            
        elif scenario_type == 'weak':
            # Zayıf sinyal, yüksek gürültü
            t, sig = self.generate_pulse_train(
                num_pulses=8, prf=1000, pulse_width=0.5e-6,
                amplitude=0.5, carrier_freq=10e6
            )
            sig, _ = self.add_noise(sig, snr_db=0)
            params = {'snr_db': 0, 'description': 'Zayıf sinyal (SNR=0dB)'}
            
        elif scenario_type == 'multiple_targets':
            # İki farklı hedef (farklı PRF)
            t1, sig1 = self.generate_pulse_train(
                num_pulses=5, prf=1000, pulse_width=1e-6,
                amplitude=1.0, carrier_freq=10e6
            )
            t2, sig2 = self.generate_pulse_train(
                num_pulses=6, prf=1200, pulse_width=0.8e-6,
                amplitude=0.7, carrier_freq=10.5e6
            )
            # Daha kısa olanı al
            min_len = min(len(sig1), len(sig2))
            sig = sig1[:min_len] + sig2[:min_len]
            t = t1[:min_len]
            sig, _ = self.add_noise(sig, snr_db=8)
            params = {'snr_db': 8, 'description': 'Çoklu hedef'}
            
        return t, sig, params


# Test kodu
if __name__ == "__main__":
    # Simülatörü başlat
    sim = RadarSimulator(fs=1e6)
    
    # Temiz sinyal üret
    t, signal = sim.generate_pulse_train(num_pulses=5, prf=1000)
    
    # Gürültü ekle
    noisy_signal, _ = sim.add_noise(signal, snr_db=10)
    
    # Görselleştir
    plt.figure(figsize=(12, 4))
    plt.plot(t * 1e3, np.abs(noisy_signal))
    plt.xlabel('Zaman (ms)')
    plt.ylabel('Genlik')
    plt.title('Radar Pulse Treni (SNR=10dB)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('radar_signal_example.png', dpi=150)
    print("✅ Örnek sinyal oluşturuldu: radar_signal_example.png")
