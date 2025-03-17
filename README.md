# 8--YAPAY-ZEKA

# 🤖 Yapay Zeka (Artificial Intelligence - AI) README

## 📌 Giriş (Introduction)
Yapay Zeka (AI), bilgisayar sistemlerinin insan benzeri düşünme ve karar verme yeteneğine sahip olmasını sağlayan bir bilim dalıdır.
Makine öğrenmesi, derin öğrenme, sinir ağları ve doğal dil işleme gibi birçok alt dalı kapsar. 🚀🧠

Bu repo, yapay zeka alanındaki temel konuları ve Python kullanarak AI modelleri geliştirmenin yollarını ele almaktadır. 📊

---

## 🚀 Kurulum (Installation)
Yapay zeka projelerinde kullanılan temel kütüphaneleri yüklemek için:

```bash
pip install numpy pandas scikit-learn tensorflow keras torch torchvision transformers
```

---

## 🔥 Kullanılan Kütüphaneler (Libraries Used)

1. **NumPy** 🔢 - Sayısal hesaplamalar.
2. **Pandas** 📊 - Veri manipülasyonu ve analizi.
3. **Scikit-Learn** 🤖 - Makine öğrenmesi algoritmaları.
4. **TensorFlow** 🧠 - Derin öğrenme ve yapay sinir ağları.
5. **Keras** 🔬 - Kolay kullanımlı derin öğrenme API'si.
6. **PyTorch** 🔥 - Dinamik hesaplama grafikleri ve düşük seviyeli derin öğrenme.
7. **Transformers (Hugging Face)** 🤗 - Doğal dil işleme ve büyük dil modelleri.

---

## 🏗️ Örnek Kullanım (Examples)

### 📌 Basit Bir Yapay Sinir Ağı (TensorFlow/Keras Kullanarak)
```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Basit bir sinir ağı oluşturma
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(10,)),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Rastgele veri ile modelin çalışması
X = np.random.rand(100, 10)
y = np.random.randint(0, 2, size=(100,))
model.fit(X, y, epochs=10, batch_size=8)
```

### 🔥 PyTorch ile Basit Bir Model
```python
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

model = SimpleNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()
```

---

## 📚 Ek Kaynaklar (Additional Resources)
- [TensorFlow Resmi Dokümanı](https://www.tensorflow.org/)
- [PyTorch Resmi Dokümanı](https://pytorch.org/)
- [Scikit-Learn Dokümanı](https://scikit-learn.org/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

---

## 📌 Katkı Yapma (Contributing)
Projeye katkıda bulunmak ister misiniz? Forklayın, geliştirin ve bir PR gönderin! 🚀

---

## 📜 Lisans (License)
Bu proje MIT lisansı altında sunulmaktadır. Serbestçe kullanabilirsiniz! 😊

