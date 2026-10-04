import numpy as np
import matplotlib.pyplot as plt

# Manifold (Yüzey) tanımı: z = -x^2 - y^2 (Aşağı Bakan Paraboloid)
def f(x, y):
    return -x**2 - y**2

# Kısmi türevler
def df_dx(x, y):
    return -2 * x

def df_dy(x, y):
    return -2 * y

# Kutupsal koordinat parametreleri
theta = np.linspace(0, 2 * np.pi, 100) # Açısal çözünürlük
r_norm = np.linspace(0, 1, 50)         # Normalize edilmiş yarıçap
Theta, R_norm = np.meshgrid(theta, r_norm)

# Sınırı sinüsoidal dalga ile pürüzsüzleştirme
R0 = 3.0   # Ortalama yarıçap
A = 0.3    # Sinüsoidal dalganın genliği
k = 4      # Tepe noktası sayısı

# Sınır yarıçapının açıya bağlı değişimi
R_max = R0 + A * np.sin(k * Theta)
R = R_norm * R_max

# Kutupsal koordinatlardan Kartezyen koordinatlara geçiş
X = R * np.cos(Theta)
Y = R * np.sin(Theta)
Z = f(X, Y)

# Manifold üzerindeki P noktası
x0, y0 = 0.0, 0.0
z0 = f(x0, y0)

# P noktasındaki türev değerleri
fx = df_dx(x0, y0)
fy = df_dy(x0, y0)

# Teğet vektörler (S1 ve S2)
S1 = np.array([1, 0, fx])
S2 = np.array([0, 1, fy])

# Görselleştirme Ayarları
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# ================= KUTUYU VE TÜM UZAY ÇİZGİLERİNİ SİL =================
ax.axis('off')  # Temel eksenleri ve yazıları kapatır

# 3D arka plan panellerini tamamen görünmez yapıyoruz
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

# Panellerin etrafındaki çerçeve çizgilerinin kalınlığını sıfırlıyoruz
ax.xaxis.line.set_linewidth(0)
ax.yaxis.line.set_linewidth(0)
ax.zaxis.line.set_linewidth(0)

# Pencere arka planını şeffaf yapıyoruz
fig.patch.set_facecolor('none')
ax.set_facecolor('none')
# ======================================================================

# 1. Manifoldu (Yüzeyi) Çiz - Kenar çizgileri yok
ax.plot_surface(X, Y, Z, alpha=0.8, cmap='magma', edgecolor='none', antialiased=True)

# 2. Teğet Düzlemini Çiz - Kenar çizgileri yok
X_t = np.linspace(x0 - 1.5, x0 + 1.5, 10)
Y_t = np.linspace(y0 - 1.5, y0 + 1.5, 10)
X_t, Y_t = np.meshgrid(X_t, Y_t)
Z_t = z0 + fx * (X_t - x0) + fy * (Y_t - y0)
ax.plot_surface(X_t, Y_t, Z_t, alpha=0.5, color='limegreen', edgecolor='none', antialiased=True)

# 3. P noktası
ax.scatter([x0], [y0], [z0], color='black', s=80, label='P Noktası', zorder=5)

# 4. Teğet vektörleri çiz
scale = 1.2
ax.quiver(x0, y0, z0, S1[0]*scale, S1[1]*scale, S1[2]*scale, 
          color='mediumblue', linewidth=3, label='$S_1$ Vektörü', arrow_length_ratio=0.15)
ax.quiver(x0, y0, z0, S2[0]*scale, S2[1]*scale, S2[2]*scale, 
          color='darkorange', linewidth=3, label='$S_2$ Vektörü', arrow_length_ratio=0.15)

# Bakış açısı
ax.view_init(elev=25, azim=55)

# Başlık ve lejant (Çizgilerden arındırılmış saf grafik)
ax.set_title('Çizgisiz Boşlukta Manifold $M$ ve Teğet Düzlemi $T_pM$')
ax.legend()

plt.show()