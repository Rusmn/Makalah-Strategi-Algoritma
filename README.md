# A* Lotka-Volterra Parameter Optimizer

Implementasi algoritma A* untuk estimasi parameter model Lotka-Volterra dalam pemodelan dinamika populasi predator-mangsa.

## Deskripsi

Proyek ini mengadaptasi algoritma A* untuk menyelesaikan masalah estimasi parameter dalam model Lotka-Volterra. Model ini menggambarkan interaksi dinamis antara populasi predator dan mangsa menggunakan sistem persamaan diferensial dengan empat parameter kunci (α, β, δ, γ).

## Fitur Utama

- **Implementasi A* untuk optimasi parameter** dengan fungsi heuristik admissible
- **Simulasi model Lotka-Volterra** dengan solver persamaan diferensial
- **Generator data sintetis** dengan kontrol noise untuk testing
- **Framework evaluasi komprehensif** untuk analisis performa
- **Visualisasi hasil** dengan grafik konvergensi dan phase portrait

## Struktur File

```
├── Lotka_Volterra_Astar_Optimizer.py  
├── testing.py                         
└── README.md                          
```

## Instalasi

```bash
# Clone repository
git clone https://github.com/Rusmn/Makalah-Strategi-Algoritma.git
cd Makalah-Strategi-Algoritma

# Install dependencies
pip install numpy matplotlib scipy pandas
```

## Penggunaan

### Eksperimen Dasar
```python
from testing import Testing

# Jalankan semua eksperimen
tester = Testing()
tester.run_all_tests()
```

### Penggunaan Manual
```python
from Lotka_Volterra_Astar_Optimizer import AStarOpt, DataGen

# Generate data sintetis
true_params = (1.0, 1.5, 0.75, 1.25)
init_state = [10, 5]
time_pts = np.linspace(0, 15, 100)
_, obs_data = DataGen.gen_data(true_params, init_state, time_pts, noise_lvl=0.3)

# Setup optimizer
bounds = {
    'alpha': (0.2, 2.0), 'beta': (0.5, 2.5),
    'delta': (0.2, 1.5), 'gamma': (0.5, 2.0)
}
optimizer = AStarOpt(obs_data, time_pts, init_state, bounds, step_size=0.1)

# Optimasi parameter
best_params, best_error, results = optimizer.optimize(max_iter=300)
```

## Eksperimen yang Tersedia

1. **Eksperimen Dasar** - Validasi konsep dan fitting parameter
2. **Test Robustness** - Analisis pengaruh noise terhadap performa
3. **Test Granularity** - Trade-off antara akurasi dan efisiensi
4. **Test Ruang Pencarian** - Analisis skalabilitas algoritma
5. **Multiple Runs** - Analisis konsistensi dan reliabilitas

## Parameter Model Lotka-Volterra

- **α (alpha)**: Laju pertumbuhan mangsa tanpa predator
- **β (beta)**: Tingkat predasi/kematian mangsa
- **δ (delta)**: Efisiensi konversi predator
- **γ (gamma)**: Laju kematian predator tanpa mangsa

## Hasil Utama

- Algoritma A* berhasil mengestimasi parameter dengan akurasi yang memuaskan
- Trade-off signifikan antara granularity pencarian dan akurasi
- Robustness menurun seiring peningkatan noise dalam data
- Performa optimal pada ruang pencarian dengan ukuran sedang


## Kontak

**Muh. Rusmin Nurwadin**  
NIM: 13523068  
Email: rusmn17@gmail.com
