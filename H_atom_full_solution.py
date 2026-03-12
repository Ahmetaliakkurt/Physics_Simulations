import numpy as np
import matplotlib.pyplot as plt

# 8 adet n=2 durumunun fiziksel özelliklerini tanımlıyoruz
# g: Landé g-faktörü, mj: manyetik kuantum sayısı
states = [
    {"name": "$2s_{1/2}$", "mj": 0.5, "g": 2.0, "color": "#1f77b4", "ls": "-"},    # Mavi, düz çizgi
    {"name": "$2s_{1/2}$", "mj": -0.5, "g": 2.0, "color": "#1f77b4", "ls": "--"},   # Mavi, kesik çizgi
    {"name": "$2p_{1/2}$", "mj": 0.5, "g": 2/3, "color": "#2ca02c", "ls": "-"},    # Yeşil, düz çizgi
    {"name": "$2p_{1/2}$", "mj": -0.5, "g": 2/3, "color": "#2ca02c", "ls": "--"},   # Yeşil, kesik çizgi
    {"name": "$2p_{3/2}$", "mj": 1.5, "g": 4/3, "color": "#d62728", "ls": "-"},    # Kırmızı, düz çizgi
    {"name": "$2p_{3/2}$", "mj": 0.5, "g": 4/3, "color": "#d62728", "ls": "--"},   # Kırmızı, kesik çizgi
    {"name": "$2p_{3/2}$", "mj": -0.5, "g": 4/3, "color": "#d62728", "ls": "-."},  # Kırmızı, nokta-kesik
    {"name": "$2p_{3/2}$", "mj": -1.5, "g": 4/3, "color": "#d62728", "ls": ":"}    # Kırmızı, noktalı
]

# Hamiltonyen eklenme aşamaları (X ekseni)
stages = [
    "1. $\hat{H}_0$\n(Bohr/Schrödinger)", 
    "2. + $\hat{H}_{fs}$\n(İnce Yapı)", 
    "3. + $\hat{H}_{Lamb}$\n(QED / Vakum)", 
    "4. + $\hat{H}_{Zeeman}$\n(Manyetik Alan)"
]
num_stages = len(stages)
num_states = len(states)

# Enerji matrisi (Satırlar: Durumlar, Sütunlar: Aşamalar)
energies = np.zeros((num_states, num_stages))

# 1. Aşama: H0 (Referans 0 kabul ediliyor, tüm durumlar dejenere)
# Zaten 0 ile başlatıldı.

# 2. Aşama: İnce Yapı (Fine Structure) eklendi
for i, s in enumerate(states):
    if "3/2" in s["name"]:
        energies[i, 1] = 45.0  # 2p_3/2 seviyesi yaklaşık 45 ueV yukarı çıkar
    else:
        energies[i, 1] = 0.0   # 2s_1/2 ve 2p_1/2 aşağıda kalır

# 3. Aşama: Lamb Kayması (QED) eklendi
for i, s in enumerate(states):
    energies[i, 2] = energies[i, 1] # Önceki enerjiyi devral
    if "2s" in s["name"]:
        energies[i, 2] += 4.3  # Sadece s orbitalleri vakumdan etkilenir ve hafifçe yukarı çıkar

# 4. Aşama: Zeeman Etkisi (Dış Manyetik Alan B_z) eklendi
B_field = 8.0 # Görselleştirmeyi netleştirmek için ölçeklenmiş manyetik alan şiddeti
for i, s in enumerate(states):
    zeeman_shift = s["g"] * s["mj"] * B_field
    energies[i, 3] = energies[i, 2] + zeeman_shift

# --- GRAFİK ÇİZİMİ ---
plt.figure(figsize=(12, 7))

# Her bir durumu aşama aşama çizdiriyoruz
for i in range(num_states):
    label = f'{states[i]["name"]} ($m_j={states[i]["mj"]}$)'
    plt.plot(range(num_stages), energies[i, :], marker='o', markersize=6, 
             color=states[i]["color"], linestyle=states[i]["ls"], linewidth=2.5, label=label)

# Grafik formatlama
plt.xticks(range(num_stages), stages, fontsize=12)
plt.yticks(fontsize=11)
plt.ylabel(r"Enerji Kayması ($\mu$eV)", fontsize=14, fontweight='bold')
plt.title(r"Hidrojen Atomu $n=2$ Seviyesi: Hamiltonyen Pertürbasyonlarıyla Dejenerelik Kırılımı", 
          fontsize=16, fontweight='bold', pad=20)

# Izgara ve Lejant ayarları
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=11, frameon=True, shadow=True)

# Düzen ve Gösterim
plt.tight_layout()
plt.show()