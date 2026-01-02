---
name: EnSim Futuristic UI
overview: EnSim uygulamasinin UI'ini ultra-futuristik sci-fi estetigine donusturecegiz. Neon glow efektleri, holografik gorunumler, glassmorphism ve zengin animasyonlarla tam bir visual overhaul yapacagiz.
todos:
  - id: style-system
    content: Yeni futuristik QSS tema dosyasi olustur (neon, glow, glassmorphism)
    status: completed
  - id: custom-fonts
    content: Futuristik fontlari yukle (Orbitron, Rajdhani, JetBrains Mono)
    status: completed
  - id: kpi-cards
    content: KPI kartlarini holografik/neon tasarima donustur
    status: completed
  - id: input-panel
    content: Input paneli cyberpunk stiline cevir
    status: completed
  - id: animated-button
    content: RUN butonuna animated gradient ve glow efekti ekle
    status: completed
  - id: tab-widget
    content: Tab widget'i neon underline ve glassmorphism ile modernize et
    status: completed
  - id: graphs-theme
    content: Matplotlib grafiklerine neon cyberpunk tema uygula
    status: completed
  - id: splash-screen
    content: Splash ekrani cinematic animasyonlu hale getir
    status: completed
  - id: micro-interactions
    content: Hover, focus ve value-change animasyonlari ekle
    status: completed
  - id: 3d-enhancements
    content: 3D gorunume neon wireframe ve holografik efektler ekle
    status: completed
---

# EnSim Futuristic UI Overhaul

## Mevcut Durum Analizi

**Teknoloji Stack:**

- PyQt6 + QSS styling
- Matplotlib (2D grafikler)
- PyVista (3D nozzle visualization)
- Mevcut dark tema (~600 satir QSS)

**Ana UI Dosyalari:**

- [`src/ui/windows/main_window.py`](src/ui/windows/main_window.py) - Ana pencere ve layout
- [`assets/styles/pro.qss`](assets/styles/pro.qss) - Mevcut tema
- [`src/ui/widgets/input_panel.py`](src/ui/widgets/input_panel.py) - Input panel
- [`src/ui/widgets/graph_widget.py`](src/ui/widgets/graph_widget.py) - Matplotlib grafikleri
- [`src/ui/splash_screen.py`](src/ui/splash_screen.py) - Splash ekrani

---

## Yeni Futuristik Tasarim Sistemi

### Renk Paleti (Cyberpunk/Sci-Fi)

| Kullanim | Renk | Kod ||----------|------|-----|| Ana Arka Plan | Ultra koyu mor-siyah | `#0a0a0f` || Sekonder BG | Koyu mor | `#12121a` || Kart BG | Yarim saydam mor | `rgba(20, 15, 35, 0.85)` || Neon Cyan | Parlak cyan | `#00ffff` || Neon Magenta | Parlak pembe | `#ff00ff` || Neon Yesil | Matrix yesili | `#00ff88` || Neon Turuncu | Uyari rengi | `#ff6b00` || Glow efekti | Cyan glow | `0 0 20px #00ffff` |

### Tipografi

- **Baslik Font**: "Orbitron" veya "Audiowide" (futuristik)
- **Body Font**: "Rajdhani" veya "Exo 2" (okunabilir sci-fi)
- **Mono Font**: "JetBrains Mono" (log output)

---

## Uygulama Plani

### Faz 1: Temel Stil Sistemi

Yeni futuristik QSS dosyasi olusturma:

- Neon border-glow efektleri
- Glassmorphism background-blur efektleri
- Animated hover states (QSS transitions yerine Python animasyonlari)
- Custom font yuklemesi

### Faz 2: KPI Kartlari Yeniden Tasarimi

Mevcut `KPICard` sinifini futuristik hale getirme:

- Holografik kart cercevesi
- Neon glow value animasyonu
- Pulsing border efekti
- Gradient mesh arkaplan

### Faz 3: Input Panel Modernizasyonu

- Neon-bordered input grupları
- Animated card reveal efekti
- Futuristik spinbox ve combo tasarimi
- Cyberpunk-style RUN butonu (animated gradient + glow)

### Faz 4: Tab Widget ve Layout

- Neon underline tab indicator
- Glassmorphism tab bar
- Animated tab switch transitions
- Holografik tab ikonlari

### Faz 5: Grafik Temalari

Matplotlib temalarini gunceleme:

- Neon grid lines
- Glow efektli plot cizgileri
- Animated data points
- Cyberpunk color scheme

### Faz 6: 3D View Gelistirmeleri

- Neon wireframe mode
- Holografik plume efekti
- Grid floor efekti
- Ambient glow lighting

### Faz 7: Splash Screen ve Animasyonlar

- Cinematic loading animasyonu
- Neon logo reveal
- Particle/starfield background
- Progress bar glow efekti

### Faz 8: Micro-Interactions

- Button hover pulse
- Input focus glow animation
- Value change flash effect
- Tab switch slide animation

---

## Teknik Notlar

1. **QSS Limitleri**: PyQt6 QSS, CSS animasyonlarini desteklemez. Animasyonlar icin `QPropertyAnimation` ve custom paint kullanacagiz.
2. **Performans**: Glow efektleri icin `QGraphicsDropShadowEffect` dikkatli kullanilmali, cok fazla effect performansi etkiler.
3. **Font Yukleme**: Custom fontlar `QFontDatabase.addApplicationFont()` ile yuklenecek.
4. **Glassmorphism**: Qt'de gercek blur efekti zorlayici, `QGraphicsBlurEffect` veya yarim saydam overlay kullanilacak.

---

## Ongorülen Sonuc

- Modern, ultra-futuristik aerospace kontrol paneli gorunumu