import math
import numpy as np

class MovingAverageFilter:
    def __init__(self, window_size):
        self.window_size = window_size
        self.buffer = [0.0] * window_size
        self.index = 0
        self.sum = 0.0
        self.mean = 0.0
    
    def update(self, value):
        """Atualiza o filtro com novo valor"""
        # Remove valor antigo da soma
        self.sum -= self.buffer[self.index]
        # Adiciona novo valor
        self.buffer[self.index] = value
        self.sum += value
        # Avança índice circular
        self.index = (self.index + 1) % self.window_size
        # Calcula média
        self.mean = self.sum / self.window_size
        return self.mean

class RMSCalculator:
    def __init__(self, window_size):
        self.window_size = window_size
        self.buffer = [0.0] * window_size
        self.index = 0
        self.sum_of_squares = 0.0
        self.rms = 0.0
    
    def update(self, value):
        """Atualiza RMS com novo valor"""
        squared_value = value * value
        # Remove valor antigo
        self.sum_of_squares -= self.buffer[self.index]
        # Adiciona novo valor
        self.buffer[self.index] = squared_value
        self.sum_of_squares += squared_value
        # Avança índice
        self.index = (self.index + 1) % self.window_size
        # Calcula RMS
        self.sum_of_squares = max(self.sum_of_squares, 0.0001)  # Evita divisão por zero
        self.rms = math.sqrt(self.sum_of_squares / self.window_size)
        return self.rms

class UnbiasedIntegrator:
    def __init__(self, dt, window_size):
        self.dt = dt
        self.integral = 0.0
        self.previous_input = 0.0
        self.maf = MovingAverageFilter(window_size)
        self.output = 0.0
    
    def update(self, value):
        """Atualiza integral com novo valor (regra do trapézio)"""
        # Integração por trapézio
        self.integral += (self.dt / 2) * (value + self.previous_input)
        self.previous_input = value
        # Remove offset médio
        mean = self.maf.update(self.integral)
        self.output = self.integral - mean
        return self.output

class CPT_Decomposition:
    def __init__(self, data_obj):
        self.data = data_obj
        self.points_per_cycle = data_obj.get_points_per_cycle()
        self.dt = data_obj.get_time_step()
        
        # initialize filters and calculators
        self._init_filters()
        
        # Pré-alocar arrays numpy (
        signal_length = len(data_obj.current_segment)
        self.i_active = np.zeros(signal_length, dtype=np.float32)
        self.i_reactive = np.zeros(signal_length, dtype=np.float32)
        self.i_void = np.zeros(signal_length, dtype=np.float32)
        self.i_void_rms_history = np.zeros(signal_length, dtype=np.float32)
        self._sample_index = 0  # índice atual para preencher arrays
    
    def _init_filters(self):
        self.voltage_integrator = UnbiasedIntegrator(self.dt, self.points_per_cycle)
        self.voltage_rms = RMSCalculator(self.points_per_cycle)
        self.current_rms = RMSCalculator(self.points_per_cycle)
        self.integrated_voltage_rms = RMSCalculator(self.points_per_cycle)
        self.active_power_filter = MovingAverageFilter(self.points_per_cycle)
        self.reactive_power_filter = MovingAverageFilter(self.points_per_cycle)
    
    def process_sample(self, voltage, current):
        """Processa uma amostra de tensão e corrente"""
        # 1. Processar tensão
        integrated_voltage = self.voltage_integrator.update(voltage)
        v_rms = self.voltage_rms.update(voltage)
        i_rms = self.current_rms.update(current)
        ui_rms = self.integrated_voltage_rms.update(integrated_voltage)
        
        # 2. Calcular potências
        active_power = self.active_power_filter.update(voltage * current)
        active_power = max(active_power, 0.000001)  # Evita divisão por zero
        
        reactive_power = self.reactive_power_filter.update(integrated_voltage * current)
        
        # 3. Calcular componentes de corrente
        # Corrente ativa
        i_a = (active_power / (v_rms ** 2)) * voltage if v_rms != 0 else 0.0
        
        # Corrente reativa
        i_r = (reactive_power / (ui_rms ** 2)) * integrated_voltage if ui_rms != 0 else 0.0
        
        # Corrente void (resíduo)
        i_v = current - i_a - i_r
        
        # Calcular RMS da corrente void
        term1 = (active_power / v_rms) ** 2 if v_rms != 0 else 0
        term2 = (reactive_power / ui_rms) ** 2 if ui_rms != 0 else 0
        i_void_rms = math.sqrt(max(i_rms ** 2 - term1 - term2, 0))
        
        # Armazenar resultados diretamente no array (sem append)
        idx = self._sample_index
        self.i_active[idx] = i_a
        self.i_reactive[idx] = i_r
        self.i_void[idx] = i_v
        self.i_void_rms_history[idx] = i_void_rms
        self._sample_index += 1
    
    def decompose(self):
        """Executa decomposição CPT completa"""
        voltage_signal = self.data.voltage_segment
        current_signal = self.data.current_segment
        
        # Resetar índice para reutilização da instância
        self._sample_index = 0
        
        # Processar todas as amostras
        for v, i in zip(voltage_signal, current_signal):
            self.process_sample(v, i)
        
        # remover transitório
        #i_active = self._remove_transient(self.i_active)
        #i_reactive = self._remove_transient(self.i_reactive)
        #i_void = self._remove_transient(self.i_void)
        i_active = self.i_active
        i_reactive = self.i_reactive
        i_void = self.i_void

        return i_active, i_reactive, i_void
    
    def _remove_transient(self, signal, threshold=0.01):
        """Remove estado transitório do sinal baseado no RMS da corrente void"""
        window_size = self.points_per_cycle
        signal_length = len(signal)
        
        if signal_length < 2 * window_size: 
            return signal
        
        num_windows = signal_length // window_size
        if num_windows < 2:
            return signal
        
        # Calcular média de cada janela
        i_void_rms = np.array(self.i_void_rms_history)
        window_means = []
        
        for window_idx in range(num_windows):
            start = window_idx * window_size
            end = (window_idx + 1) * window_size
            window_mean = np.mean(i_void_rms[start:end])
            window_means.append(window_mean)
        
        # Detectar mudança significativa entre janelas
        diffs = np.abs(np.diff(window_means))
        
        if len(diffs) == 0 or max(diffs) < threshold:
            # Sem mudança significativa, remover primeiros 2 ciclos
            cut_index = 4 * window_size
        else:
            # Cortar após maior mudança detectada
            cut_index = 3 * window_size * (1 + np.argmax(diffs))
        
        if cut_index >= signal_length:
            return signal
        
        return signal[cut_index:]

def CPT(data_obj):
    processor = CPT_Decomposition(data_obj)
    return processor.decompose()