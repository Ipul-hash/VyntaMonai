# Technical Documentation — Level 1: Anatomical Segmentation & Volumetric Quantification (CADq)

**Project Name**: VyntaMonai AI Medical Imaging Automation Pipeline  
**Role & Subsystem**: Level 1 — Anatomical Semantic Segmentation & Standard DICOM-SEG Volumetric Quantification  
**Target Environment**: 100% Local CPU Execution (Zero GPU/CUDA Overhead, Zero External PACS Dependency)  
**Version**: 1.0.0  
**Date**: September 2026  

---

## 1. Executive Summary & Clinical Intent

Dalam alur kerja radiologi dan integrasi kecerdasan buatan (*health-tech AI integration*), **Level 1 (CADq - Computer-Aided Quantification)** merupakan fondasi kritis sebelum model diagnosa patologi dapat dijalankan. 

Modul Level 1 ini bertugas untuk:
1. **Mengonsumsi citra tomografi medis (CT Scan)** secara otomatis langsung dari sistem penyimpanan data lokal atau PACS.
2. **Melakukan penapisan spasial medis** (validasi orientasi, jarak irisan, dan konversi ke skala Hounsfield Units terstandarisasi).
3. **Mengisolasi batas anatomi organ secara 3D piksel per piksel** (*voxel-wise delineation*) menggunakan deep learning MONAI / PyTorch pada arsitektur CPU.
4. **Menghitung volumetri organ** secara kuantitatif ($mL / cm^3$) sebagai parameter fisiologis klinis.
5. **Membungkus output AI ke dalam standar internasional DICOM-SEG (Segmentation Object)** yang ditautkan langsung ke `SOPInstanceUID` irisan original pasien.
6. **Menyajikan hasil visualisasi overlay interaktif** melalui Web DICOM Viewer tanpa ketergantungan software eksternal berat.

---

## 2. Arsitektur Pipeline & Data Flow

```text
[Direktori CT DICOM Pasien / PACS Local Storage]
                     │
                     ▼
       [1. DICOM Series Ingest & Spatial Sort]
         - Sortir urutan Z via proyeksi normal vektor: d = P · (R × C)
         - Ekstraksi RescaleSlope & RescaleIntercept ke Hounsfield Units (HU)
         - Validasi integritas FrameOfReferenceUID & ImageOrientationPatient
                     │
                     ▼
       [2. CPU Preprocessing & Spatial Transform]
         - Clipping window intensitas anatomis (Soft Tissue vs Lung)
         - Konversi koordinat: DICOM LPS -> MONAI Canonical RAS
         - Anisotropic spacing resampling ke grid target CPU
                     │
                     ▼
       [3. MONAI 3D Inference Engine (CPU-Optimized)]
         - Sliding Window Inference (patch 16-divisible)
         - Thread allocation: torch.set_num_threads(4)
         - Zero autograd memory footprint: torch.inference_mode()
                     │
                     ▼
       [4. Postprocessing & Spatial Restoration]
         - Inversi RAS -> DICOM LPS
         - Nearest-neighbor interpolation kembali ke matriks CT original (512x512)
         - Largest Connected Component (LCC) morphological cleaning
                     │
                     ▼
       [5. highdicom DICOM-SEG Encapsulation]
         - Pemetaan SOPClassUID: 1.2.840.10008.5.1.4.1.1.66.4
         - Binding metadata SourceImageSequence & PerFrameFunctionalGroupsSequence
         - Terminologi Medis SNOMED-CT (Spleen: 78961009, Lung: 39607008)
                     │
                     ▼
       [6. Custom HTML5 Web DICOM Viewer]
         - Real-time windowing (Lung Window, Soft Tissue, Bone)
         - Dynamic Canvas overlay blending dengan opacity slider
         - Cine player loop & 4-corner radiology HUD
```

---

## 3. Kepatuhan Standar Medis & Spesifikasi DICOM

Untuk memastikan interoperabilitas penuh dengan PACS rumah sakit (seperti Orthanc, dcm4chee, GE, Siemens, Philips) dan Web Viewer (OHIF, Horos, Radiant), output disimpan sesuai standar DICOM Part 3 & Part 16:

| Atribut DICOM | Tag DICOM | Nilai Implementasi | Penjelasan Medis / Teknis |
| :--- | :--- | :--- | :--- |
| **SOP Class UID** | `(0008,0016)` | `1.2.840.10008.5.1.4.1.1.66.4` | **Segmentation Storage** (Standar resmi multi-frame DICOM-SEG). |
| **Modality** | `(0008,0060)` | `SEG` | Modalitas segmentasi sekunder yang dapat di-overlay. |
| **Source Image Sequence** | `(0008,2112)` | Ditautkan ke tiap `SOPInstanceUID` CT original | Memastikan keterlacakan (*lineage*) frame ke irisan CT asli. |
| **Segmentation Type** | `(0062,0001)` | `BINARY` | 0 = Background, 1 = Jaringan organ tersegmentasi. |
| **Segment Algorithm Type** | `(0062,0008)` | `AUTOMATIC` | Dihasilkan secara komputasi otomatis oleh AI. |
| **Algorithm Family** | `(0066,002F)` | `CID 7162: 113690 (Artificial Intelligence)` | Pengelompokan algoritma berstandar DICOM. |
| **Anatomy Category** | CID 7150 | `SCT: 123037004 (Anatomical Structure)` | Kategori ontologi SNOMED-CT. |
| **Anatomy Type (Lung)** | Target Code | `SCT: 39607008 (Lung)` | Kode ontologi organ Paru-paru. |
| **Anatomy Type (Spleen)** | Target Code | `SCT: 78961009 (Spleen)` | Kode ontologi organ Limpa. |

---

## 4. Landasan Matematika Spasial & Rekonstruksi Geometri 3D

### A. Proyeksi Vektor Normal untuk Pengurutan Irisan
File DICOM CT sering kali disimpan dengan penamaan acak (*unsorted filenames*). Untuk menjamin akurasi 3D, setiap irisan diurutkan berdasarkan proyeksi skalar posisi pasien terhadap vektor normal bidang irisan:

$$\vec{R} = [IOP_0, IOP_1, IOP_2] \quad (\text{Direction Cosine Baris})$$
$$\vec{C} = [IOP_3, IOP_4, IOP_5] \quad (\text{Direction Cosine Kolom})$$
$$\vec{N} = \vec{R} \times \vec{C} \quad (\text{Vektor Normal Bidang Irisan})$$
$$d_i = \vec{IPP}_i \cdot \vec{N} \quad (\text{Jarak Skalar terhadap Normal})$$

Seluruh dataset irisan diurutkan secara menaik (*ascending*) berdasarkan nilai $d_i$.

### B. Konversi Hounsfield Units (HU)
Nilai mentah pada `PixelData` dikonversi ke skala fisik densitas radiologis terstandarisasi:

$$HU = (\text{PixelValue} \times \text{RescaleSlope}) + \text{RescaleIntercept}$$

- Udara (*Air*): $\approx -1000\text{ HU}$
- Parenkim Paru (*Lung tissue*): $-950\text{ s/d } -400\text{ HU}$
- Lemak (*Fat*): $-100\text{ s/d } -50\text{ HU}$
- Jaringan Lunak / Organ (*Soft tissue*): $+30\text{ s/d } +60\text{ HU}$
- Tulang Tulang Kortikal (*Bone*): $+400\text{ s/d } > +1000\text{ HU}$

### C. Transformasi Sistem Koordinat: DICOM LPS $\longleftrightarrow$ MONAI RAS
- Standar koordinat spasial pasien pada DICOM menggunakan **LPS** (*Left, Posterior, Superior*).
- Standar kanonikal model deep learning MONAI menggunakan **RAS** (*Right, Anterior, Superior*).
- Konversi dilakukan dengan membalik (*flip*) sumbu transversal (X) dan sumbu sagital (Y):

$$X_{\text{RAS}} = -X_{\text{LPS}}, \quad Y_{\text{RAS}} = -Y_{\text{LPS}}, \quad Z_{\text{RAS}} = Z_{\text{LPS}}$$

Setelah inferensi selesai, transformasi invers diterapkan kembali agar mask biner presisi secara geometris saat dibungkus ke DICOM-SEG.

---

## 5. Strategi Optimasi CPU (Zero-GPU Constraint)

Agar model 3D dapat berjalan dengan mulus pada CPU laptop/server tanpa lag:

1. **Sliding Window Patching dengan Kelipatan Divisible 16**:
   - Arsitektur UNet 3D dengan 4 level downsampling mensyaratkan setiap dimensi patch habis dibagi $2^4 = 16$.
   - Digunakan ukuran patch `roi_size = (64, 96, 96)` dengan `overlap = 0.25` dan `sw_batch_size = 1`.
2. **Thread Contention Management**:
   - Pembatasan worker thread PyTorch (`torch.set_num_threads(4)`) guna mencegah saturasi core CPU (mencegah *freezing* pada OS pengguna).
3. **Memory Footprint Reduction**:
   - Eksekusi berada di dalam blok konteks `with torch.inference_mode():` yang meniadakan pembuatan graf autograd dan menghemat alokasi memori RAM hingga ~40%.
4. **Anisotropic Spacing Resampling**:
   - Volume CT original di-resample ke resolusi spasial target `(1.5, 1.5, 2.0) mm` sebelum masuk model, kemudian dikembalikan ke grid matriks asli $512 \times 512$ via *nearest-neighbor interpolation* untuk menjaga integritas batas biner tanpa *label bleeding*.

---

## 6. Kuantifikasi Volumetri Klinis

Kapasitas volume organ dihitung secara otomatis menggunakan formula integrasi voxel spasial 3D:

$$V_{\text{organ}} = \sum_{z=1}^{D} \sum_{y=1}^{H} \sum_{x=1}^{W} M(z, y, x) \times \frac{\Delta x \cdot \Delta y \cdot \Delta z}{1000} \quad (\text{dalam satuan mL atau cm}^3)$$

*Di mana:*
- $M(z, y, x) \in \{0, 1\}$ adalah nilai mask biner pada koordinat voxel tersebut.
- $\Delta x, \Delta y$ adalah nilai `PixelSpacing` (mm).
- $\Delta z$ adalah nilai `SliceThickness` atau interval antar irisan (mm).
- Faktor pembagi $1000$ mengonversi satuan dari $mm^3$ ke $mL$ ($1\text{ mL} = 1000\text{ mm}^3 = 1\text{ cm}^3$).

---

## 7. Hasil Benchmark & Verifikasi Data Pasien Asli

Pipeline diuji secara komparatif pada dua dataset:

| Metrik Evaluasi | Dataset Pasien Asli (`pasien_01`) | Dataset Kontrol (`sample_ct_abdomen`) |
| :--- | :--- | :--- |
| **Deskripsi Klinis** | Thorax / Chest CT (LIDC-IDRI-0580) | Phantom Abdomen Lengkap |
| **Jumlah Irisan (Slices)** | **250 Slices** ($512 \times 512$) | 36 Slices ($256 \times 256$) |
| **Voxel Spacing (mm)** | $0.70 \times 0.70 \times 1.25\text{ mm}$ | $1.20 \times 1.20 \times 2.50\text{ mm}$ |
| **Target Organ** | **Bilateral Lungs (Paru Kanan & Kiri)** | **Spleen (Limpa)** |
| **Durasi Ingest & Sort** | $1.15\text{ detik}$ | $0.07\text{ detik}$ |
| **Durasi Resampling Spasial** | $1.03\text{ detik}$ | $0.03\text{ detik}$ |
| **Durasi CPU Inference** | $6.56\text{ detik}$ | $1.87\text{ detik}$ |
| **Durasi DICOM-SEG Export** | $0.74\text{ detik}$ | $0.35\text{ detik}$ |
| **Total Waktu Pemrosesan** | **`11.18 detik`** *(250 slice penuh!)* | **`2.35 detik`** |
| **Voxel Positif Terdeteksi** | $8,199,558\text{ voxels}$ | $25,740\text{ voxels}$ |
| **Volume Organ Terukur** | **`5,067.17 mL (~5.06 L)`** | **`92.66 mL`** |
| **Kesesuaian Fisiologis** | ✅ **Sesuai Total Lung Capacity (TLC)** | ✅ **Sesuai rentang limpa normal** |

---

## 8. Spesifikasi Custom HTML5 Web DICOM Viewer

Aplikasi web viewer dibangun dengan arsitektur SPA (*Single Page Application*) zero-dependency:

- **Frontend Engine**: HTML5 Canvas compositing, CSS Grid Dark Mode berstandar radiologi, Vanilla ES6 JavaScript.
- **Backend API Server**: FastAPI + Uvicorn (port 8000).
- **Fitur Interaksi**:
  1. *Mouse Wheel & Keyboard Scrolling*: Navigasi irisan instan dengan *caching* memory in-RAM (< 5ms per render).
  2. *Radiological Window Presets*:
     - **Lung Window** ($WW: 1500, WL: -600\text{ HU}$)
     - **Soft Tissue Window** ($WW: 350, WL: 40\text{ HU}$)
     - **Bone Window** ($WW: 1800, WL: 400\text{ HU}$)
     - **Brain Window** ($WW: 80, WL: 40\text{ HU}$)
  3. *AI Overlay Control*: Pengaturan transparansi dinamis (0–100%) dan kustomisasi warna mask (Cyan, Red, Green, Yellow).
  4. *Cine Mode*: Pemutaran loop otomatis seluruh irisan CT dengan kecepatan 20 FPS.
  5. *Radiology 4-Corner HUD*: Menampilkan data pasien, resolusi matriks, koordinat Z spasial, dan parameter windowing secara *real-time*.

---

## 9. Struktur Modul Kode

```text
VyntaMonai/
├── src/
│   ├── app/
│   │   ├── main.py                   # FastAPI entrypoint, CORS, static mounts
│   │   ├── config.py                 # Konfigurasi CPU threads, path, target spacing
│   │   ├── api/v1/
│   │   │   ├── endpoints.py          # Endpoints sistem (/health, /infer/local-directory)
│   │   │   └── viewer.py             # Rendering irisan on-the-fly & API metadata viewer
│   │   ├── core/
│   │   │   ├── logger.py             # Structured medical logging
│   │   │   └── exceptions.py         # Exception handling spesifik medis
│   │   └── services/
│   │       ├── dicom_reader.py       # Reader standar medis & sortir vektor normal Z
│   │       ├── preprocessor.py       # Transformasi spasial RAS/LPS & resampling
│   │       ├── inference.py          # Sliding-window CPU inference engine
│   │       ├── seg_builder.py        # Pembungkus highdicom DICOM-SEG
│   │       └── pipeline.py           # Orchestrator alur end-to-end
│   └── viewer/
│       ├── local_viewer.py           # Desktop viewer script (Matplotlib)
│       └── static/index.html         # Custom HTML5 Web DICOM Viewer interface
├── data/
│   ├── samples/                      # Direktori input series CT pasien
│   └── output_seg/                   # Output file DICOM-SEG (.dcm)
└── tests/
    └── test_dicom_pipeline.py        # Automated test suite (100% PASS)
```

---

## 10. Panduan Menjalankan & Verifikasi

### Menjalankan Server Web Viewer:
```powershell
.venv\Scripts\python.exe -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload
```
Akses di browser: **[http://localhost:8000/viewer](http://localhost:8000/viewer)**

### Menjalankan Automated Test Suite:
```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```
