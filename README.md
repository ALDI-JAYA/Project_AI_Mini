# AI Jobs Enthusiast 2030

Project Python untuk menganalisis dataset `AI_Impact_on_Jobs_2030.csv` dari sudut pandang AI enthusiast: pekerjaan mana yang berisiko otomatisasi, pekerjaan mana yang punya peluang tinggi untuk dibantu AI, dan skill profile seperti apa yang menarik untuk diprioritaskan.

## Fitur

- Membaca dataset AI Impact on Jobs 2030.
- Membersihkan tipe data numerik dan validasi kolom penting.
- Membuat metrik turunan:
  - `Average_Salary_IDR`
  - `Skill_Readiness`
  - `Automation_Risk_Score`
  - `AI_Opportunity_Score`
  - `AI_Enthusiast_Segment`
- Menampilkan insight terminal lewat `main.py`.
- Menyediakan dashboard interaktif lewat Streamlit di `app.py`.
- Memberi rekomendasi karier personal berdasarkan profil user.
- Menganalisis gap skill menuju pekerjaan target.
- Mengonversi salary dari USD ke Rupiah dengan kurs yang bisa diubah.
- Melatih model machine learning untuk memprediksi `Risk_Category`.
- Menampilkan akurasi, macro F1, confusion matrix, classification report, dan feature importance.
- Membandingkan beberapa model ML dalam mode full features dan no-leakage.

## Struktur

```text
ai-jobs-enthusiast-2030/
  ai_jobs_enthusiast/
    analysis.py
    config.py
    data.py
    ml_model.py
    recommender.py
    skill_gap.py
  app.py
  main.py
  requirements.txt
  README.md
```

## Instalasi

```powershell
cd "C:\Users\Aldi Tri Wijaya\Documents\Codex\2026-05-07\files-mentioned-by-the-user-ai\ai-jobs-enthusiast-2030"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Menjalankan Analisis Terminal

```powershell
python main.py
```

Secara default project akan membaca file ini:

```text
C:\Users\Aldi Tri Wijaya\Downloads\archive\AI_Impact_on_Jobs_2030.csv
```

Jika ingin memakai path CSV lain:

```powershell
python main.py --data "D:\folder\AI_Impact_on_Jobs_2030.csv"
```

## Menjalankan Dashboard

```powershell
streamlit run app.py
```

Dashboard akan membuka halaman lokal di browser. Di sana kamu bisa filter berdasarkan kategori risiko dan edukasi, melihat pekerjaan dengan peluang AI tertinggi, membandingkan risiko otomatisasi dengan opportunity score, memakai AI Career Recommender, menganalisis skill gap untuk pekerjaan target, dan mengevaluasi model machine learning.

Default kurs yang dipakai adalah `1 USD = Rp16.000`. Nilai ini bisa diubah dari sidebar dashboard.

## Fitur AI Career Recommender

Masukkan profil seperti education level, pengalaman kerja, target gaji, toleransi risiko otomatisasi, dan level skill saat ini. Aplikasi akan menghasilkan `Personal_Fit_Score` untuk membantu memilih pekerjaan yang paling cocok.

## Fitur Skill Gap Analyzer

Pilih pekerjaan target dan masukkan level skill saat ini. Aplikasi akan membandingkan skill kamu dengan rata-rata profil skill pekerjaan target, lalu memberi prioritas `High`, `Medium`, atau `Low` beserta rekomendasi latihan.

## Fitur Machine Learning Model

Project membandingkan beberapa model untuk memprediksi `Risk_Category`:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

Dashboard menyediakan dua mode evaluasi:

- `Full Features`: memakai semua fitur, termasuk `Automation_Probability_2030`.
- `No-Leakage`: menghapus `Automation_Probability_2030` agar evaluasi tidak terlalu bergantung pada fitur yang sangat dekat dengan label.

Output evaluasi tersedia di terminal dan dashboard:

- Accuracy
- Macro F1
- Training time
- Confusion matrix
- Classification report
- Feature importance

Catatan portofolio: mode no-leakage penting karena `Automation_Probability_2030` kemungkinan sangat berkaitan dengan `Risk_Category`. Membandingkan dua mode ini menunjukkan pemahaman tentang data leakage dan evaluasi model yang lebih profesional.

## Ide Pengembangan

- Tambahkan model prediksi kategori risiko.
- Buat rekomendasi skill yang lebih spesifik berdasarkan jenis pekerjaan.
- Ekspor laporan otomatis ke Excel atau PDF.
- Tambahkan clustering pekerjaan untuk mencari kelompok karier yang mirip.
