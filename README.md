# VyntaMonai: Local CPU Medical AI Automation Pipeline

Pipeline otomatisasi AI Medical Imaging lokal berbasis **MONAI**, **PyTorch (CPU-mode)**, **pydicom**, **highdicom**, dan **FastAPI**.
Dirancang khusus untuk kebutuhan portofolio engineering berstandar industri dengan kemampuan inferensi tanpa GPU dan enkapsulasi output ke format standar medis **DICOM-SEG**.

> 📖 **Dokumentasi Teknis Lengkap**:
> - **Level 1 (CADq - Quantification)**: Baca [docs/LEVEL_1_DOCUMENTATION.md](file:///c:/Users/User/Documents/VyntaCAD/VyntaMonai/docs/LEVEL_1_DOCUMENTATION.md) (Segmentasi 3D organ paru & limpa, DICOM-SEG standard `1.2.840.10008.5.1.4.1.1.66.4`, volumetri).
> - **Level 2 (CADe - Detection)**: Baca [docs/LEVEL_2_DOCUMENTATION.md](file:///c:/Users/User/Documents/VyntaCAD/VyntaMonai/docs/LEVEL_2_DOCUMENTATION.md) (Deteksi nodul paru 3D Cascade AI, kategorisasi risiko Fleischner Society 2017, export standar DICOM-SR `1.2.840.10008.5.1.4.1.1.88.22`).
> - **Level 3 (CADx - Diagnosis)**: Baca [docs/LEVEL_3_DOCUMENTATION.md](file:///c:/Users/User/Documents/VyntaCAD/VyntaMonai/docs/LEVEL_3_DOCUMENTATION.md) (Analisis radiomik 3D, spikulasi tepi corona radiata, probabilitas keganasan kanker, klasifikasi resmi ACR Lung-RADS v2022).

---

## 🏗️ Arsitektur Sistem (Level 1 CADq + Level 2 CADe + Level 3 CADx)

```text
[Raw CT DICOM Series (e.g. pasien_01 250 slices)]
                      │
                      ▼
[DICOM Reader] (Z-projection slice sorting, Hounsfield Unit extraction)
                      │
       ┌──────────────┴──────────────┐
       ▼                             ▼
[Level 1: CADq Pipeline]      [Level 2: CADe Pipeline]
  - 3D UNet Organ Mask          - Cascade AI (Constrained to Lung Mask)
  - Volumetri (mL / cm³)        - 3D Connected Components & Vessel Rejection
  - highdicom DICOM-SEG         - Fleischner Society 2017 Risk Categorization
  - output_seg/*.dcm                 │
       │                             ▼
       │                      [Level 3: CADx Diagnostic Engine]
       │                        - 3D Radiomics (Spiculation / Calcification)
       │                        - Calibrated Malignancy Risk Score (0-100%)
       │                        - ACR Lung-RADS v2022 Assessment
       │                        - DICOM Enhanced Structured Report (DICOM-SR)
       │                             │
       └──────────────┬──────────────┘
                      ▼
[Interactive HTML5 Web DICOM Workstation (http://localhost:8000/viewer)]
  - CT Grayscale with Window/Level Presets (Lung, Soft Tissue, Bone)
  - AI DICOM-SEG Multi-color Overlay Layer (Adjustable Opacity)
  - AI CADe Target Bounding Boxes (Color-coded by Lung-RADS Severity)
  - Level 3 Lung-RADS Triage Filter Buttons (All, Cat 4, Cat 3, Cat 2)
  - Interactive Diagnostic Profile Inspector (Malignancy Gauge & Clinical Action)
```

---

## 🚀 Quickstart Guide

### 1. Menjalankan Microservice & Web Workstation
```bash
.venv\Scripts\python.exe -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000 --reload
```
Buka browser di: **`http://localhost:8000/viewer`**

### 2. Fitur Web Viewer Workstation
- **Active Study Selector**: Pilih antara `pasien_01` (Chest CT real) atau `sample_ct_abdomen`.
- **Level 1 (CADq)**: Aktifkan/nonaktifkan mask segmentasi organ dan atur opacity serta warna overlay.
- **Level 2 (CADe)**: Bounding box neon target nodul paru otomatis muncul di atas slice yang mengandung lesi. Klik kartu nodul pada sidebar (`Nodule #1`, `Nodule #3`, dll.) untuk otomatis lompat (*auto-jump*) ke slice lesi tersebut.
- **Cine Loop**: Tekan tombol `Cine Play` atau spasi untuk pemutaran otomatis animasi slice.

### 3. Menjalankan Automated Unit Tests (100% Passing)
```bash
# Menjalankan seluruh test suite (12 tests)
.venv\Scripts\python.exe -m unittest discover tests

# Atau jalankan per modul:
.venv\Scripts\python.exe -m unittest tests/test_dicom_pipeline.py  # Level 1 (DICOM-SEG)
.venv\Scripts\python.exe -m unittest tests/test_detection.py       # Level 2 (CADe & SR)
.venv\Scripts\python.exe -m unittest tests/test_cadx.py            # Level 3 (CADx & Lung-RADS)
```

---

## 🩺 Standar Medis & Kepatuhan DICOM
- **Level 1 DICOM-SEG**: SOP Class UID `1.2.840.10008.5.1.4.1.1.66.4` (Segmentation Storage) dengan standard terminology SNOMED-CT (`39607008` Bilateral Lungs, `78961009` Spleen).
- **Level 2 & 3 DICOM-SR**: SOP Class UID `1.2.840.10008.5.1.4.1.1.88.22` (Enhanced SR Storage) yang memuat koordinat 3D temuan nodul, pedoman klinis Fleischner Society 2017, kategorisasi resmi ACR Lung-RADS v2022, estimasi probabilitas keganasan, dan rekomendasi tatalaksana medis.
