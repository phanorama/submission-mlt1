# Memprediksi Harga Gas Alam Global Menggunakan Deep Learning Berbasis Long Short Term Memory

## 1. Domain Proyek
### Latar Belakang
Gas alam memegang peran strategis dalam sistem energi dunia. Konsumsi global gas alam terus meningkat dan gas alam menyumbang bagian signifikan dari campuran energi dunia [1], [2]. IEA melaporkan bahwa pada 2024 permintaan gas alam naik sebesar 2,8% dan gas alam memenuhi sekitar 40% dari kenaikan permintaan energi global – proporsi terbesar dibandingkan bahan bakar lainnya [2]. Karena peranan penting ini, peramalan harga gas alam yang akurat menjadi sangat penting untuk perencanaan investasi energi, penetapan kebijakan, diversifikasi portofolio, dan strategi lindung nilai [1]. Fluktuasi harga gas alam yang tinggi dapat menimbulkan beban signifikan bagi berbagai sektor. Misalnya, IEA mencatat bahwa pasar gas Eropa yang sangat volatil pada awal 2025 mendorong harga ke tingkat tertinggi dalam dua tahun terakhir, sehingga menambah tekanan pada daya saing industri dan biaya hidup masyarakat [3]. Ketidakpastian harga yang tinggi juga meningkatkan risiko finansial bagi produsen dan konsumen energi [4]. Oleh karena itu, peramalan harga gas alam yang andal menjadi semakin dibutuhkan untuk mendukung pengambilan keputusan strategis dan pengelolaan risiko di sektor energi global.
### Masalah dan Urgensi
Meskipun penting, peramalan harga gas alam global menghadapi tantangan besar. Harga gas alam dipengaruhi oleh beragam faktor eksternal (cuaca ekstrem, dinamika geopolitik, kebijakan energi) yang menimbulkan fluktuasi signifikan dan sulit diprediksi. Livieris et al. mencatat bahwa karena sifat pasar gas yang chaotik, peramalan harga gas alam merupakan tugas yang sangat kompleks dan menantang [1]. Akibatnya, model statistik tradisional seperti ARIMA sering kesulitan mengikuti dinamika non-linear dan korelasi jangka panjang dalam data harga gas alam. Kejadian global tak terduga (misalnya pandemi COVID-19 atau konflik geopolitik) sering memicu lonjakan harga gas alam secara tiba-tiba. Sebagai contoh, EIA mencatat harga gas alam amblas pada awal 2020 dan kemudian melonjak drastis selama periode 2021–2022 [5]. Variasi musiman dan kondisi cuaca ekstrem juga berdampak besar, misalnya lonjakan permintaan pada musim dingin atau panas yang tidak biasa, sehingga menyebabkan fluktuasi konsumsi dan penyimpanan gas yang sulit diprediksi
[5]. Model peramalan konvensional (seperti ARIMA/GARCH) berasumsi linearitas dan stasioneritas, sehingga sering gagal menangkap pola dinamis non-linear maupun hubungan memori panjang pada data harga gas [1]. Jaringan LSTM merupakan model pembelajaran mendalam yang mampu mempelajari dependensi jangka panjang pada deret waktu. Lapisan LSTM dapat menangkap pola temporal kompleks dalam data yang berisik [1], sehingga pendekatan ini berpotensi meningkatkan akurasi prediksi harga gas alam dibandingkan teknik tradisional. Dengan latar tersebut, penggunaan model LSTM untuk memprediksi harga gas alam ke depan menjadi sangat penting. Livieris et al. menyatakan bahwa peramalan harga gas alam dianggap esensial karena hasilnya digunakan dalam perencanaan energi, perdagangan komoditas, dan pengambilan keputusan [1]. Oleh karena itu, model prediksi berbasis LSTM diharapkan dapat menghasilkan prakiraan harga hingga Januari 2028 yang lebih akurat, sehingga mendukung pengambilan keputusan strategis dan pengelolaan risiko di sektor energi global. 

## 2. Business Understanding

### Problem Statement

Harga gas alam global bersifat sangat fluktuatif dan sulit diprediksi. Model peramalan konvensional seperti ARIMA seringkali tidak cukup efektif untuk menangkap pola dinamis dan kompleks pada data harga energi.

### Goals

Mengembangkan model deep learning berbasis LSTM dan untuk memprediksi harga gas alam ke depan, dan membandingkan performa kedua model tersebut.

### Solution Statement

1. Membangun model baseline menggunakan algoritma LSTM untuk memprediksi harga gas alam.
3. Melakukan evaluasi dan perbandingan model menggunakan metrik MSE, RMSE dan MAE.
4. Menyediakan prediksi harga gas alam hingga Januari 2028 kedepan.

## 3. Data Understanding

* **Sumber Data**: U.S. Energy Information Administration (EIA)
  [Link Dataset](https://datahub.io/core/natural-gas#readme)

* **Periode**: Januari 1997 hingga tahun terkini

* **Frekuensi**: Harian

* **Fitur utama**: `Date`, `Price`

* **Jumlah data**: 7128 baris (bervariasi tergantung preprocessing)

### Exploratory Data Analysis

* Visualisasi tren harga gas alam sepanjang waktu
* Deteksi missing value
* Plot seasonal dan rolling mean untuk melihat fluktuasi dan tren jangka panjang
![download](https://github.com/user-attachments/assets/c256883b-e1cf-4c50-8b36-079a492ea278)

## 4. Data Preparation

* Konversi data tanggal menjadi datetime
* Penyesuaian frekuensi menjadi bulanan
* Normalisasi data menggunakan MinMaxScaler
* Pembuatan window input dengan sequence length panjang 60 hari
* Split data menjadi training dan testing (80:20)

## 5. Modeling

### Model 1: LSTM

* Layer: 1 LSTM layers + Dense output
* Loss: MSE
* Optimizer: Adam
* Epoch: 50–100 (Improvement)
* Output: Prediksi harga gas dari test set dan extended prediction hingga Januari 2028

### Model Improvement

* EarlyStopping

## 6. Evaluation

* **Metrik**: RMSE, MAE, MSE
* Visualisasi: plot aktual vs prediksi
* Analisis performa pada data test dan data prediksi ke masa depan

### Rumus RMSE:

$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

### Rumus MAE:

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

### Rumus MSE

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

### Hasil Uji Test
```
Test Loss (MSE): 0.0010
Test MSE       : 0.0010
Test RMSE      : 0.0161
Test MAE       : 0.0093
```
![download](https://github.com/user-attachments/assets/ffd1ffe8-d0d4-4217-b2ce-dbbf4ebbd34d)

### Hasil Sementara:
![download](https://github.com/user-attachments/assets/ef23c8cd-4a00-4183-a485-295b9885b6d0)
* Prediksi harga gas alam per 6 Januari 2028: \$3.32/MMBtu (berdasarkan model LSTM)

## 7. Kesimpulan dan Rekomendasi

* Model LSTM berhasil mempelajari pola harga gas dengan akurasi yang cukup baik.
* Diperlukan evaluasi lanjutan menggunakan data eksternal dan integrasi fitur tambahan (misalnya suhu, permintaan industri).
* Bisa lakukan studi komperatif dengan beberapa model dan metode statistik seperti Monte-Carlo
---

## Referensi
[1] Ioannis E. Livieris, Emmanuel Pintelas, Niki Kiriakidou, Stavros Stavroyiannis. An Advanced Deep Learning Model for Short-Term Forecasting U.S. Natural Gas Price and Movement. 16th IFIP International Conference on Artificial Intelligence Applications and Innovations (AIAI), Jun 2020, Neos Marmaras, Greece. pp.165-176, ⟨10.1007/978-3-030-49190-1_15⟩. ⟨hal-03677622⟩ \
[2] International Energy Agency, Gas Market Report, Q1-2025. Paris: IEA, 2025. [Online]. Available: https://iea.blob.core.windows.net/assets/23968aa1-73c7-4f29-86e8-38d9818fadfc/GasMarketReport,Q1-2025.pdf \
[3] International Energy Agency, “European gas market volatility puts continued pressure on competitiveness and cost of living,” IEA, Paris, 2025. [Online]. Available: https://www.iea.org/commentaries/european-gas-market-volatility-puts-continued-pressure-on-competitiveness-and-cost-of-living \
[4] S. Hosseinipoor, “Consumption of natural gas as an energy source and the role of natural gas futures market to hedge the financial risk,” Student Journal of Economics, Univ. of Oklahoma, pp. 1–14, 2020. [Online]. Available: https://ou.edu/content/dam/cas/economics/Student%20Journal%20of%20Economics%20publications/Saied%20Hosseinipoor_AppJOE.pdf \
[5] U.S. Energy Information Administration, “Natural gas prices are increasingly driven by global fundamentals,” Short-Term Energy Outlook Special Supplement, U.S. EIA, Washington, DC, Jan. 2023. [Online]. Available: https://www.eia.gov/outlooks/steo/special/supplements/2023/2023_sp_01.pdf
