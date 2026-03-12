import tkinter as tk
from tkinter import ttk
import math

def dalga_boyundan_renk(dalga_boyu):
    try:
        wl = float(dalga_boyu)
        
        if wl < 450: return "#8B00FF"
        elif wl < 495: return "#0000FF"
        elif wl < 520: return "#00FFFF"
        elif wl < 565: return "#00FF00"
        elif wl < 590: return "#FFFF00"
        elif wl < 625: return "#FF8C00"
        elif wl < 750: return "#FF0000"
        else: return "#8B0000"
    except:
        return "#FF0000"

def hesapla_f_gereken(s1, s2_prime):
    if s1 <= 0 or s2_prime <= 0:
        return None
    return 1 / (1/s1 + 1/s2_prime)

def tahmin_kirma_indisi(dalga_boyu):
    A = 1.50
    B = 50000.0
    return A + B / (dalga_boyu ** 2)

def hesapla_mercek_yaricapi(f, n, d):
    try:
        if f is None or f <= 0:
            return None, "A positive focal length (f) is required."
        
        a = 1 / f
        b = -2 * (n - 1)
        c = (n - 1)**2 * d / n
        
        diskriminant = b**2 - 4 * a * c
        
        if diskriminant < 0:
            return None, "No real surface radius (R) found (Complex Root)."
        
        R1 = (-b + math.sqrt(diskriminant)) / (2 * a)
        R2 = (-b - math.sqrt(diskriminant)) / (2 * a)
        
        R_gereken = max(R1, R2)
        
        if R_gereken <= 0:
             return None, "No positive surface radius found."
        
        return R_gereken, None

    except Exception as e:
        return None, f"Mathematical Error: {e}"

def cizim_setup_guncelle(event=None):
    cizim_yap(guncelle_sonuc=False)

def hesapla_ve_ciz(event=None):
    cizim_yap(guncelle_sonuc=True)

def cizim_yap(guncelle_sonuc=False):

    try:
        dalga_boyu = float(dalga_boyu_var.get())
        kalinlik_d = float(kalinlik_var.get())
    except ValueError:
        sonuc_etiketi.config(text="⚠️ ERROR: Inputs must be numerical (use '.' for decimal).", foreground="#E74C3C")
        canvas.delete("all")
        mercek_canvas.delete("all")
        return

    isin_rengi = dalga_boyundan_renk(dalga_boyu)
    
    mercek_konumu_mm = mercek_slider.get()
    hedef_konumu_mm = hedef_slider.get()

    s1 = mercek_konumu_mm
    s2_prime = hedef_konumu_mm - s1

    canvas.delete("all")
    canvas_genislik = canvas.winfo_width()
    canvas_yukseklik = 280
    orta_cizgi_y = canvas_yukseklik / 2
    
    MAX_MM = 500
    PADDING = 80
    olcekleme_faktoru = (canvas_genislik - PADDING * 2) / MAX_MM

    lazer_x = PADDING + 20
    mercek_x = lazer_x + mercek_konumu_mm * olcekleme_faktoru
    hedef_x = lazer_x + hedef_konumu_mm * olcekleme_faktoru

    for i in range(0, MAX_MM + 1, 50):
        x = lazer_x + i * olcekleme_faktoru
        canvas.create_line(x, 10, x, canvas_yukseklik - 10, fill="#E8E8E8", width=1)
    
    canvas.create_line(lazer_x, orta_cizgi_y, canvas_genislik - PADDING, orta_cizgi_y, 
                       fill="#34495E", width=2, dash=(8, 4))
    canvas.create_text(canvas_genislik - PADDING + 5, orta_cizgi_y + 15, 
                       text="Principal Axis", fill="#34495E", font=("Segoe UI", 8, "italic"), anchor="w")

    canvas.create_oval(lazer_x - 12, orta_cizgi_y - 12, lazer_x + 12, orta_cizgi_y + 12, 
                       fill=isin_rengi, outline=isin_rengi, width=3)
    canvas.create_text(lazer_x - 18, orta_cizgi_y, 
                       text="Light Source", fill=isin_rengi, font=("Segoe UI", 10, "bold"), anchor="e")
    
    offsets = [0, 25, 45, -25, -45]
    for offset in offsets:
         canvas.create_line(lazer_x, orta_cizgi_y + offset, mercek_x, orta_cizgi_y + offset, 
                            fill=isin_rengi, width=2, arrow=tk.LAST, arrowshape=(10, 12, 4))

    mercek_kalinlik_piksel = 10
    mercek_yukseklik = 100
    
    canvas.create_rectangle(mercek_x - mercek_kalinlik_piksel/2 + 3, orta_cizgi_y - mercek_yukseklik/2 + 3,
                            mercek_x + mercek_kalinlik_piksel/2 + 3, orta_cizgi_y + mercek_yukseklik/2 + 3,
                            fill="#95A5A6", outline="")
    
    canvas.create_rectangle(mercek_x - mercek_kalinlik_piksel/2, orta_cizgi_y - mercek_yukseklik/2,
                            mercek_x + mercek_kalinlik_piksel/2, orta_cizgi_y + mercek_yukseklik/2,
                            fill="#AED6F1", outline="#2980B9", width=2)
    
    kavis_r = 20
    canvas.create_arc(mercek_x - kavis_r, orta_cizgi_y - mercek_yukseklik/2, 
                      mercek_x + kavis_r, orta_cizgi_y + mercek_yukseklik/2,
                      start=270, extent=180, style=tk.ARC, outline="#2980B9", width=3)
    canvas.create_arc(mercek_x - kavis_r, orta_cizgi_y - mercek_yukseklik/2, 
                      mercek_x + kavis_r, orta_cizgi_y + mercek_yukseklik/2,
                      start=90, extent=180, style=tk.ARC, outline="#2980B9", width=3)
    
    canvas.create_text(mercek_x, orta_cizgi_y - mercek_yukseklik/2 - 18, 
                       text="Lens", fill="#2980B9", font=("Segoe UI", 10, "bold"))

    if hedef_x > mercek_x:
        for offset in offsets:
            canvas.create_line(mercek_x, orta_cizgi_y + offset, hedef_x, orta_cizgi_y, 
                              fill=isin_rengi, width=2, arrow=tk.LAST, arrowshape=(10, 12, 4))
        
        for r in [12, 8, 5]:
            canvas.create_oval(hedef_x - r, orta_cizgi_y - r, hedef_x + r, orta_cizgi_y + r, 
                              outline=isin_rengi, width=2, fill="" if r > 5 else isin_rengi)
        
        canvas.create_text(hedef_x, orta_cizgi_y + 25, 
                          text="Focus", fill=isin_rengi, font=("Segoe UI", 10, "bold"))
        
        canvas.create_line(lazer_x, orta_cizgi_y + 70, mercek_x, orta_cizgi_y + 70, 
                          fill="#E67E22", width=2, arrow=tk.BOTH)
        canvas.create_text((lazer_x + mercek_x) / 2, orta_cizgi_y + 85, 
                          text=f"s₁ = {s1:.1f} mm", fill="#E67E22", font=("Segoe UI", 9, "bold"))
        
        canvas.create_line(mercek_x, orta_cizgi_y + 70, hedef_x, orta_cizgi_y + 70, 
                          fill="#9B59B6", width=2, arrow=tk.BOTH)
        canvas.create_text((mercek_x + hedef_x) / 2, orta_cizgi_y + 85, 
                          text=f"s₂' = {s2_prime:.1f} mm", fill="#9B59B6", font=("Segoe UI", 9, "bold"))
    else:
        canvas.create_text(canvas_genislik / 2, orta_cizgi_y, 
                          text="⚠️ Target must be positioned after the lens!", 
                          fill="#E74C3C", font=("Segoe UI", 14, "bold"))
    
    if guncelle_sonuc:
        f_gereken = hesapla_f_gereken(s1, s2_prime)
        n_tahmini = tahmin_kirma_indisi(dalga_boyu)
        R_hesaplanan, hata_mesaji = hesapla_mercek_yaricapi(f_gereken, n_tahmini, kalinlik_d)

        if s1 <= 0 or s2_prime <= 0:
            sonuc_etiketi.config(
                text="⚠️ Lens and target positions must be positive!", 
                foreground="#E74C3C")
        elif R_hesaplanan is None:
            sonuc_etiketi.config(
                text=f"⚠️ {hata_mesaji}\nAdjust parameters.", 
                foreground="#E74C3C")
        else:
            sonuc_metni = f"""✅ CALCULATION RESULTS
            
Focal Length (f):  {f_gereken:.2f} mm
Refractive Index (n):   {n_tahmini:.4f}  [λ = {dalga_boyu} nm]
Lens Thickness (d):   {kalinlik_d:.2f} mm

═══════════════════════════════
🎯 Required Surface Radius (R):  {R_hesaplanan:.2f} mm
═══════════════════════════════

Symmetrical Biconvex Lens Design
R₁ = +{R_hesaplanan:.2f} mm  |  R₂ = -{R_hesaplanan:.2f} mm"""
            
            sonuc_etiketi.config(text=sonuc_metni, foreground="#27AE60")
            
            mercek_canvas.update_idletasks()
            ciz_mercek_sekli(mercek_canvas, R_hesaplanan, kalinlik_d)

def ciz_mercek_sekli(canvas, R, d):
    
    canvas.delete("all")
    
    W = canvas.winfo_width()
    H = canvas.winfo_height()
    merkez_x = W / 2
    merkez_y = H / 2
    
    lens_height = 120
    y_ust = merkez_y - lens_height / 2
    y_alt = merkez_y + lens_height / 2
    
    d_piksel = min(80, max(30, 1000 / R))
    
    canvas.create_text(merkez_x, H - 20, 
                      text=f"Symmetrical Design  |  R₁ = +{R:.2f} mm  |  d = {d:.2f} mm  |  R₂ = -{R:.2f} mm", 
                      font=("Segoe UI", 10, "bold"), fill="#2980B9")
    
    canvas.create_line(100, merkez_y, W - 100, merkez_y, fill="#BDC3C7", width=1, dash=(5, 3))
    
    x_sol = merkez_x - d_piksel / 2
    x_sag = merkez_x + d_piksel / 2
    
    kavis_genisligi = min(60, max(20, 800 / R))
    
    fill_color = "#EBF5FB"
    
    canvas.create_arc(
        x_sol - kavis_genisligi, y_ust, 
        x_sol + kavis_genisligi, y_alt,
        start=270, extent=180, 
        style=tk.CHORD, fill=fill_color, outline="", width=0
    )
    
    canvas.create_arc(
        x_sag - kavis_genisligi, y_ust, 
        x_sag + kavis_genisligi, y_alt,
        start=90, extent=180, 
        style=tk.CHORD, fill=fill_color, outline="", width=0
    )
    
    outline_color = "#2980B9"
    outline_width = 3
    
    canvas.create_arc(
        x_sol - kavis_genisligi, y_ust, 
        x_sol + kavis_genisligi, y_alt,
        start=270, extent=180, 
        style=tk.ARC, outline=outline_color, width=outline_width
    )
    
    canvas.create_arc(
        x_sag - kavis_genisligi, y_ust, 
        x_sag + kavis_genisligi, y_alt,
        start=90, extent=180, 
        style=tk.ARC, outline=outline_color, width=outline_width
    )

    canvas.create_line(x_sol, y_ust - 20, x_sag, y_ust - 20, 
                       fill="#E67E22", width=2, arrow=tk.BOTH)
    canvas.create_text(merkez_x, y_ust - 32, 
                       text=f"d = {d:.2f} mm", fill="#E67E22", font=("Segoe UI", 10, "bold"))
    
    r_start_x = x_sol - kavis_genisligi
    canvas.create_line(r_start_x, merkez_y, x_sol, y_ust, 
                       fill="#9B59B6", width=2, dash=(4, 2))
    canvas.create_text(r_start_x - 45, merkez_y - 25, 
                       text=f"R₁ = {R:.1f} mm", fill="#9B59B6", font=("Segoe UI", 9, "bold"))
    
    r_end_x = x_sag + kavis_genisligi
    canvas.create_line(r_end_x, merkez_y, x_sag, y_ust, 
                       fill="#9B59B6", width=2, dash=(4, 2))
    canvas.create_text(r_end_x + 45, merkez_y - 25, 
                       text=f"R₂ = {-R:.1f} mm", fill="#9B59B6", font=("Segoe UI", 9, "bold"))

kok = tk.Tk()
kok.title("🔬 Light Focusing and Lens Design")
kok.geometry("1600x900")
kok.configure(bg="#ECF0F1")

style = ttk.Style()
style.theme_use('clam')
style.configure('TFrame', background='#ECF0F1')
style.configure('TLabelframe', background='#ECF0F1', foreground='#2C3E50')
style.configure('TLabelframe.Label', font=('Segoe UI', 10, 'bold'))
style.configure('TLabel', background='#ECF0F1', foreground='#2C3E50')
style.configure('TButton', font=('Segoe UI', 10, 'bold'))

ana_cerceve = ttk.Frame(kok, padding="15")
ana_cerceve.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
kok.columnconfigure(0, weight=1)
ana_cerceve.columnconfigure(0, weight=1)
ana_cerceve.columnconfigure(1, weight=1)

baslik = tk.Label(ana_cerceve, text="🔬 LIGHT FOCUSING SYSTEM DESIGNER", 
                font=("Segoe UI", 16, "bold"), bg="#ECF0F1", fg="#2C3E50")
baslik.grid(row=0, column=0, columnspan=2, pady=(0, 10))

sol_kolon = ttk.Frame(ana_cerceve)
sol_kolon.grid(row=1, column=0, sticky=(tk.N, tk.W, tk.E), padx=(0, 15))

input_frame = ttk.LabelFrame(sol_kolon, text="⚙️  Lens Parameters", padding="15")
input_frame.grid(row=0, column=0, pady=(0, 15), sticky=(tk.W, tk.E))
input_frame.columnconfigure(1, weight=1)

ttk.Label(input_frame, text="Wavelength (λ):").grid(row=0, column=0, sticky=tk.W, pady=8, padx=(0, 10))
dalga_boyu_var = tk.StringVar(value="632.8")
dalga_boyu_giris = ttk.Entry(input_frame, textvariable=dalga_boyu_var, width=15, font=("Segoe UI", 10))
dalga_boyu_giris.grid(row=0, column=1, sticky=tk.W, pady=8)
ttk.Label(input_frame, text="nm").grid(row=0, column=2, sticky=tk.W, padx=(5, 0))
dalga_boyu_giris.bind("<Return>", hesapla_ve_ciz)

ttk.Label(input_frame, text="Lens Thickness (d):").grid(row=1, column=0, sticky=tk.W, pady=8, padx=(0, 10))
kalinlik_var = tk.StringVar(value="5.0")
kalinlik_giris = ttk.Entry(input_frame, textvariable=kalinlik_var, width=15, font=("Segoe UI", 10))
kalinlik_giris.grid(row=1, column=1, sticky=tk.W, pady=8)
ttk.Label(input_frame, text="mm").grid(row=1, column=2, sticky=tk.W, padx=(5, 0))
kalinlik_giris.bind("<Return>", hesapla_ve_ciz)

slider_frame = ttk.LabelFrame(sol_kolon, text="📍  Position Settings", padding="15")
slider_frame.grid(row=1, column=0, pady=(0, 15), sticky=(tk.W, tk.E))
slider_frame.columnconfigure(0, weight=1)

ttk.Label(slider_frame, text="Lens Position (s₁):").grid(row=0, column=0, sticky=tk.W, pady=8)
mercek_slider = tk.Scale(slider_frame, from_=1, to=499, orient=tk.HORIZONTAL, 
                         length=350, resolution=0.1, command=cizim_setup_guncelle,
                         bg="#ECF0F1", fg="#2C3E50", highlightthickness=0, 
                         troughcolor="#BDC3C7", font=("Segoe UI", 9))
mercek_slider.set(100.0)
mercek_slider.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 15))

ttk.Label(slider_frame, text="Target Position (s₁ + s₂'):").grid(row=2, column=0, sticky=tk.W, pady=8)
hedef_slider = tk.Scale(slider_frame, from_=2, to=500, orient=tk.HORIZONTAL, 
                        length=350, resolution=0.1, command=cizim_setup_guncelle,
                        bg="#ECF0F1", fg="#2C3E50", highlightthickness=0, 
                        troughcolor="#BDC3C7", font=("Segoe UI", 9))
hedef_slider.set(200.0)
hedef_slider.grid(row=3, column=0, sticky=(tk.W, tk.E))

hesapla_butonu = tk.Button(sol_kolon, text="⚡ CALCULATE & DRAW", command=hesapla_ve_ciz,
                           font=("Segoe UI", 12, "bold"), bg="#27AE60", fg="white",
                           padx=25, pady=12, relief="raised", cursor="hand2")
hesapla_butonu.grid(row=2, column=0, pady=(0, 15))

sonuc_frame = tk.Frame(sol_kolon, bg="white", relief="solid", borderwidth=2)
sonuc_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

sonuc_etiketi = tk.Label(sonuc_frame, text="Set parameters and press\n'Calculate & Draw'.", 
                         font=("Consolas", 10), bg="white", fg="#7F8C8D", 
                         justify=tk.LEFT, padx=20, pady=20)
sonuc_etiketi.grid(row=0, column=0, sticky=(tk.W, tk.N))

sag_kolon = ttk.Frame(ana_cerceve)
sag_kolon.grid(row=1, column=1, sticky=(tk.N, tk.W, tk.E, tk.S))

canvas_frame = ttk.LabelFrame(sag_kolon, text="🔭  Optical Setup", padding="10")
canvas_frame.grid(row=0, column=0, pady=(0, 15), sticky=(tk.W, tk.E))

canvas = tk.Canvas(canvas_frame, width=1050, height=280, bg="white", 
                  borderwidth=2, relief="solid", highlightthickness=0)
canvas.grid(row=0, column=0)

mercek_canvas_frame = ttk.LabelFrame(sag_kolon, text="🔍  Lens Design", padding="10")
mercek_canvas_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))

mercek_canvas = tk.Canvas(mercek_canvas_frame, width=1050, height=220, bg="white", 
                          borderwidth=0, highlightthickness=0)
mercek_canvas.grid(row=0, column=0)

kok.update_idletasks()
cizim_setup_guncelle()

kok.mainloop()