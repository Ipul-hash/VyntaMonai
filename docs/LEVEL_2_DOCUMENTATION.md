# VyntaMonai Level 2 Documentation: 3D Pulmonary Nodule Detection (CADe) & DICOM-SR

## 1. Executive Summary & Clinical Taxonomy

| Dimension | Level 1: Quantification (CADq) | Level 2: Detection (CADe) |
| :--- | :--- | :--- |
| **Objective** | Anatomical organ boundary segmentation | Identification & localization of abnormal focal lesions |
| **Target Anatomy** | Bilateral Lungs / Spleen | Pulmonary Nodules (Solid, Subsolid, Ground-Glass) |
| **Output DICOM Format**| **DICOM-SEG** (1.2.840.10008.5.1.4.1.1.66.4)| **DICOM-SR** (1.2.840.10008.5.1.4.1.1.88.22) |
| **Clinical Value** | Organ volume quantification (e.g. 5.06 L lung volume) | Early lung cancer screening, nodule burden tracking |
| **Clinical Standard** | SNOMED-CT / RadLex Organ Codes | Fleischner Society 2017 Lung Nodule Guidelines |

In medical imaging artificial intelligence:
- **CADq (Computer-Aided Quantification)**: Segments known macro-anatomical volumes (completed in Level 1).
- **CADe (Computer-Aided Detection)**: Automatically flags and alerts clinicians to abnormal lesions that could easily be overlooked during rapid radiologist reading rounds.
- **CADx (Computer-Aided Diagnosis)**: Predicts lesion malignancy or staging (Level 3).

---

## 2. Cascade AI Architecture

To maximize sensitivity while minimizing false-positive detections on a purely CPU-driven environment, VyntaMonai implements a **Cascade AI Pipeline**:

### Why Cascade AI is Crucial:
1. **Search Space Reduction**: Aerated lungs constitute only ~20-30% of the whole thoracic volume. Searching exclusively within the Level 1 lung mask cuts unnecessary voxel analysis by >70%.
2. **Elimination of Extrapulmonary False Positives**: Muscle layers, subcutaneous fat, mediastinal vessels, and vertebral trabeculae have overlapping Hounsfield Unit densities with nodules. The Level 1 anatomical mask guarantees zero false positives outside pulmonary parenchyma.
3. **Pleural Interface Guarding**: A 3x3 2D erosion kernel is applied to the lung boundaries to prevent costal cartilage, ribs, and pleural thickening from masquerading as peripheral nodules.

---

## 3. Algorithmic Implementation Details

### 3.1 3D Density & Connectivity Analysis
1. **Voxel Selection**:
   Mask_candidate = (V_HU >= -450) and (V_HU <= +150) and ErodedLung
2. **3D 18-Connectivity Graph**:
   scipy.ndimage.label groups adjacent voxels in 3 dimensions into contiguous lesion volumes V_cluster.
3. **Micro-Noise Rejection**:
   Clusters smaller than 8 voxels (~0.5 mm³) are filtered out to suppress reconstruction noise.

### 3.2 Feature Extraction & Tubular Vessel Rejection
Pulmonary vessels run through the parenchyma. While nodules are spherical or lobular, vessels are tubular and elongated.
- **Equivalent Spherical Diameter**:
  d = 2 * (3 * V / (4 * pi))^(1/3)
- **Vessel Elongation Heuristic**:
  Elongation = max(dx, dy, dz) / min(dx, dy)
  If Elongation > 3.8, the candidate is classified as a blood vessel branch and rejected.

### 3.3 Density Sub-classification
- **Solid Nodule**: Mean HU > -100 HU
- **Part-Solid (Subsolid) Nodule**: -350 HU <= Mean HU <= -100 HU
- **Ground-Glass Nodule (GGN)**: Mean HU < -350 HU

---

## 4. Fleischner Society 2017 Guidelines Integration

Each detected nodule is automatically categorized according to the international Fleischner Society 2017 criteria:

| Category | Diameter Range | Clinical Interpretation & Follow-up |
| :--- | :--- | :--- |
| **Low Risk** | < 6.0 mm | Subcentimeter incidental nodule; routine CT follow-up optional. |
| **Intermediate Risk** | 6.0 - 8.0 mm | Elevated suspicion; follow-up low-dose CT recommended at 6 to 12 months. |
| **High Risk** | > 8.0 mm | Clinically actionable; consider CT at 3 months, PET/CT scan, or tissue biopsy. |

---

## 5. DICOM Enhanced Structured Report (DICOM-SR) Specification

The findings are encapsulated into a native DICOM file adhering to the **Enhanced SR Storage SOP Class** (1.2.840.10008.5.1.4.1.1.88.22):
- **Study / Patient Continuity**: Inherits PatientID, StudyInstanceUID, and clinical header information.
- **Series Identification**: Modality SR and unique SeriesInstanceUID.
- **Content Sequence Structure**:
  - DOCUMENT TITLE: Diagnostic Imaging Report (LOINC 11528-7)
  - CONTAINER: Pulmonary Nodule Detection Findings (DCM 121070)
  - Finding Items: Nodule #X, Fleischner recommendation, Diameter (mm), Volume (mm³), Mean HU, and 3D Coordinates (SCOORD3D).

---

## 6. Web Viewer Workstation Features

The HTML5 DICOM Viewer (http://localhost:8000/viewer) includes dedicated Level 2 workstation tools:
1. **Target Bounding Box Overlay**: Neon amber bounding box (#ffaa00) + green reticle crosshair (#00e676) + HUD label.
2. **Interactive Findings List**: Sidebar cards sorted by confidence and risk.
3. **One-Click Slice Jump**: Clicking any nodule card in the sidebar instantly jumps the viewport to that slice.
4. **Independent Toggles**: Independent visibility toggles for Level 1 SEG mask and Level 2 CADe boxes.

---

## 7. API Reference

- GET /api/v1/viewer/detections/{study_name}: Retrieve all detected nodules in JSON.
- POST /api/v1/viewer/detect/{study_name}: Run CADe detection, generate DICOM-SR, and return findings.
