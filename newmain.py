import random

# Data dummy
toko = ['TEH', 'GP', 'TPE', 'ECT', 'TUCR', 'SSK', 'ECR', 'EC', 'SEJ', 'SE']
permintaan = {'TEH': 50, 'GP': 60, 'TPE': 70, 'ECT': 80, 'TUCR': 55, 'SSK': 65,
              'ECR': 45, 'EC': 90, 'SEJ': 50, 'SE': 40}
matriks_jarak = {
    'DC': {'TEH': 10, 'GP': 12, 'TPE': 15, 'ECT': 18, 'TUCR': 20, 'SSK': 25, 'ECR': 30, 'EC': 35, 'SEJ': 40, 'SE': 45},
    'TEH': {'GP': 5, 'TPE': 8, 'ECT': 12, 'TUCR': 18, 'SSK': 22, 'ECR': 27, 'EC': 32, 'SEJ': 38, 'SE': 42, 'DC': 10},
    # Lengkapi matriks dari gambar
}

kapasitas_truk = 271  # Kapasitas truk

# Fungsi untuk inisialisasi populasi
def inisialisasi_populasi(jumlah_individu):
    populasi = []
    for _ in range(jumlah_individu):
        rute = list(permintaan.keys())
        random.shuffle(rute)  # Acak urutan toko
        rute = pisahkan_rute(rute)
        populasi.append(rute)
    return populasi

# Fungsi untuk memisahkan rute berdasarkan kapasitas
def pisahkan_rute(rute):
    rute_terpisah = []
    rute_sementara = ['DC']
    kapasitas_sisa = kapasitas_truk

    for toko in rute:
        if permintaan[toko] <= kapasitas_sisa:
            rute_sementara.append(toko)
            kapasitas_sisa -= permintaan[toko]
        else:
            rute_sementara.append('DC')
            rute_terpisah.append(rute_sementara)
            rute_sementara = ['DC', toko]
            kapasitas_sisa = kapasitas_truk - permintaan[toko]

    rute_sementara.append('DC')
    rute_terpisah.append(rute_sementara)
    return rute_terpisah

# Fungsi untuk menghitung jarak total
def hitung_jarak(rute):
    jarak_total = 0
    for sub_rute in rute:
        for i in range(len(sub_rute) - 1):
            jarak_total += matriks_jarak[sub_rute[i]][sub_rute[i + 1]]
    return jarak_total

# Fungsi fitness
def fitness(rute):
    jarak = hitung_jarak(rute)
    return 1 / jarak if jarak > 0 else 0  # Fitness lebih tinggi untuk jarak lebih pendek

# Mutasi
def mutasi(rute):
    rute_flat = [toko for sub_rute in rute for toko in sub_rute if toko != 'DC']
    i, j = random.sample(range(len(rute_flat)), 2)
    rute_flat[i], rute_flat[j] = rute_flat[j], rute_flat[i]
    return pisahkan_rute(rute_flat)

# Contoh penggunaan
populasi = inisialisasi_populasi(10)
for individu in populasi:
    print("Rute:", individu)
    print("Jarak:", hitung_jarak(individu))
