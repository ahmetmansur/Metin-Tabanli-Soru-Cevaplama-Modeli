
# 📘 Metin Tabanlı Soru-Cevaplama Sistemi

Bu proje, bir metin belgesine dayanarak çoktan seçmeli sorulara otomatik yanıt verebilen bir **Soru-Cevaplama (QA)** sistemidir. Proje kapsamında Python programlama dili ve klasik NLP teknikleri kullanılarak metin analizi gerçekleştirilmiştir.

## 🚀 Projenin Amacı

Metin içerisinden anlamlı bilgiyi çıkararak kullanıcı tarafından yöneltilen sorulara en uygun cevabı verebilen, sade ve etkili bir model geliştirmek.

## 🧠 Kullanılan Yöntemler ve Teknolojiler

- **TF-IDF (Term Frequency–Inverse Document Frequency)** ile vektörleştirme
- **Cosine Similarity** ile anlamsal yakınlık ölçümü
- **NLTK** ile metin ön işleme (stopwords, tokenization)
- `scikit-learn`, `numpy` kütüphaneleri

## ⚙️ Sistem Çalışma Adımları

1. Metin dosyası sisteme yüklenir (`Whispers of Adventure.txt`).
2. Cümlelere bölünüp temizlenir, TF-IDF ile vektörleştirilir.
3. Soru ve cevap seçenekleri alınır.
4. Soru vektörü ile metindeki cümleler karşılaştırılır.
5. En uygun şık cosine similarity skoruna göre seçilir.

## 💻 Kurulum

```bash
pip install nltk scikit-learn numpy
```

Python içinden çalıştırmadan önce:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## 📂 Dosya Yapısı

```
.
├── LLM_v4.py                  # Ana uygulama kodu
├── Whispers of Adventure.txt # Girdi metni (hikaye)
```

## 📝 Örnek Kullanım

```python
question = "What is the name of the young explorer in the story?"
choices = ["Lila", "Malgor", "Zephyr", "Willow"]
```

Çıktı:
```
Predicted Answer: Lila (Score: 0.91)
```

## 👤 Geliştirici

**Ahmet Mansur Özer**  
İskenderun Teknik Üniversitesi – Bilgisayar Mühendisliği  
E-posta: ozerahmetmansur@gmail.com



