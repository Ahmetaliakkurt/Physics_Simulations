import numpy as np
import matplotlib.pyplot as plt
import time
import fast_ensemble  # C++ Modülümüz!

# --- PARAMETRELER ---
M = 100000        
N = 500           
L = 10.0          
m = 1.0           
dt = 0.005        
radius = 0.15     
STEPS = 200       

K_PAIRS = N // 2  
TARGET_T = 300.0
TARGET_KE = N * TARGET_T   
P_theo = TARGET_KE / (L**2)

print(f"Başlangıç matrisleri hazırlanıyor... ({M} Evren, {M*N} Atom)")

# C++ tarafına geçecek RAM dostu Matrisler (NumPy)
pos = np.random.uniform(radius, L - radius, size=(M, N, 2))
angles = np.random.uniform(0, 2 * np.pi, size=(M, N))
speeds = np.random.uniform(1.0, 5.0, size=(M, N))

vel = np.zeros((M, N, 2))
vel[:, :, 0] = speeds * np.cos(angles)
vel[:, :, 1] = speeds * np.sin(angles)

# Başlangıç Momentum ve Enerji Ölçeklemesi (Python'da yapıp C++'a hazır veriyoruz)
cm_vel = vel.mean(axis=1, keepdims=True)
vel -= cm_vel
current_ke = np.sum(0.5 * m * (vel[:, :, 0]**2 + vel[:, :, 1]**2), axis=1)
scale = np.sqrt(TARGET_KE / current_ke)
vel[:, :, 0] *= scale[:, np.newaxis]
vel[:, :, 1] *= scale[:, np.newaxis]
vel -= vel.mean(axis=1, keepdims=True)

print("C++ / OPENMP Fizik Motoru Tetiklendi! Kemerlerinizi bağlayın...")
start_time = time.time()

# SIHİRLİ SATIR: 100.000 Evreni C++'a Fırlatıyoruz
results = fast_ensemble.run_simulation(
    pos, vel, M, N, L, m, dt, radius, STEPS, K_PAIRS, TARGET_KE
)

calc_time = time.time() - start_time
print(f"Hesaplama bitti! Toplam C++ süresi: {calc_time:.4f} saniye.")

# Sonuçları C++ Sözlüğünden Çekme
time_data = np.array(results["time_data"])
ensemble_pressure_data = np.array(results["ensemble_pressure"])
ke_data = np.array(results["ke_data"])
mean_abs_px_data = np.array(results["mean_abs_px"])
mean_abs_py_data = np.array(results["mean_abs_py"])

overall_mean_p = np.mean(ensemble_pressure_data)
relative_error = abs(overall_mean_p - P_theo) / P_theo * 100

print(f"Teorik Beklenen Basınç : {P_theo:.6f} Pa")
print(f"Ölçülen Ansamble Ort.  : {overall_mean_p:.6f} Pa")
print(f"Bağıl Hata             : % {relative_error:.6f}")

# (Bundan sonraki kısıma senin yazdığın o muazzam Matplotlib Dark Mode çizim kodlarını ekleyebilirsin)
# fig = plt.figure(...) vb.
