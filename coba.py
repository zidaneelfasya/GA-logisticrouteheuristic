import pandas as pd
import numpy as np

# Membaca file CSV ke dalam DataFrame pandas
file_path = "data_route.csv"
data = pd.read_csv(file_path, header=None)  # header=None karena file hanya berisi angka

# Mengonversi DataFrame menjadi matriks numpy
distance_matrix = data.values

print("Matriks Jarak dari CSV:")
print(distance_matrix)


print(round(np.random.uniform(0,0.9), 1))
