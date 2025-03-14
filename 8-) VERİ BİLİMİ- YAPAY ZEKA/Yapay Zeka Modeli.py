import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 📌 1. Sahte Veri Oluşturma (Yapay Zeka için)
np.random.seed(42)
X = np.random.rand(100, 1) * 10  # 0-10 arasında rastgele sayılar
y = 3 * X + np.random.randn(100, 1) * 2  # Doğrusal bir fonksiyon

# 📌 2. Veriyi Görselleştirme
plt.scatter(X, y, color="blue", alpha=0.5)
plt.xlabel("Bağımsız Değişken (X)")
plt.ylabel("Bağımlı Değişken (y)")
plt.title("Yapay Zeka İçin Örnek Veri")
plt.show()

# 📌 3. Veriyi Eğitim ve Test Olarak Ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 4. Makine Öğrenmesi Modeli (Basit Lineer Regresyon)
model = LinearRegression()
model.fit(X_train, y_train)  # Modeli eğit

# 📌 5. Tahmin Yapma
y_pred = model.predict(X_test)

# 📌 6. Modelin Başarısını Ölçme
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"📌 Ortalama Hata Karesi (MSE): {mse:.2f}")
print(f"📌 R-Kare Skoru (R²): {r2:.2f}")

# 📌 7. Sonuçları Görselleştirme
plt.scatter(X_test, y_test, color="blue", label="Gerçek Değerler")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Tahminler")
plt.xlabel("Bağımsız Değişken (X)")
plt.ylabel("Bağımlı Değişken (y)")
plt.title("Yapay Zeka Modeli Tahminleri")
plt.legend()
plt.show()
