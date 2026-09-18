# VyntaMonai Level 3 Documentation: CADx Malignancy Prediction & ACR Lung-RADS v2022

## 1. Executive Summary & The Medical AI Trinity

With Level 3, VyntaMonai completes the **Gold Standard Medical Imaging AI Pipeline**:

$$\text{Level 1 (CADq: Quantification)} \longrightarrow \text{Level 2 (CADe: Detection)} \longrightarrow \text{Level 3 (CADx: Diagnosis)}$$

| Dimension | Level 1: CADq | Level 2: CADe | Level 3: CADx |
| :--- | :--- | :--- | :--- |
| **Clinical Question** | *"Where are the organs & what is their volume?"* | *"Where are the focal lesions / nodules?"* | *"Is this nodule Benign or Malignant (Cancer)?"* |
| **Target Output** | Multi-organ 3D Segmentation | Spatial Coordinates & Bounding Boxes | Malignancy Risk Score ($P_{\text{mal}}$) & Lung-RADS Category |
| **DICOM Standard** | **DICOM-SEG** (`1.2.840.10008.5.1.4.1.1.66.4`) | **DICOM-SR** (`1.2.840.10008.5.1.4.1.1.88.22`) | **Enhanced DICOM-SR** (TID 1500 / Malignancy Assessment) |
| **Clinical Guideline** | SNOMED-CT Anatomy Codes | Fleischner Society 2017 | **ACR Lung-RADS v2022 Guidelines** |

---

## 2. 3D Radiomics & Texture Feature Extraction

Level 3 CADx analyzes both macro-morphological and intra-lesional textural features to differentiate indolent benign granulomas from aggressive malignancies.

### 2.1 Margin Spiculation Index (Corona Radiata Analysis)
Malignant bronchogenic carcinomas typically present with irregular, spiculated radiating borders (*corona radiata*) due to invasive infiltration into adjacent parenchymal septa, whereas benign lesions (granulomas, hamartomas, intrapulmonary lymph nodes) exhibit smooth, circumscribed contours.

1. **Boundary Extraction**:
   $$\text{Boundary} = \text{NoduleMask} \land (\neg \text{ErodedMask})$$
2. **Radial Distance Dispersion**:
   For all boundary voxels $\vec{x}_i$, distance from centroid $\vec{x}_c$:
   $$r_i = \|\vec{x}_i - \vec{x}_c\|_2$$
3. **Coefficient of Radial Variation ($CV_r$)**:
   $$CV_r = \frac{\sigma(r_i)}{\mu(r_i) + \epsilon}$$
4. **Size-Damped Spiculation Index**:
   $$\text{SpiculationIndex} = \min\left(1.0, \max\left(0.05, CV_r \cdot \text{Damping}(d) \cdot 2.0\right)\right)$$
   - **Smooth / Circumscribed**: $\text{Index} < 0.22$
   - **Lobulated / Irregular**: $0.22 \le \text{Index} < 0.45$
   - **Spiculated (*Corona Radiata*)**: $\text{Index} \ge 0.45$

### 2.2 Intra-Nodular Density & Calcification Profiling
- **Benign Calcification Pattern**: Popcorn, central, or concentric dense calcifications ($>200\text{ HU}$) occupying $>35\%$ of the nodule sphere represent healed histoplasmosis/tuberculosis granulomas or chondroid hamartomas (strictly classified as **Lung-RADS 2**).
- **Subsolid Attenuation Ratio**: Part-solid nodules with ground-glass halos carry significantly elevated risk for adenocarcinoma in situ (AIS) or minimally invasive adenocarcinoma (MIA).

---

## 3. Calibrated Bayesian Malignancy Probability Model

VyntaMonai implements a calibrated multivariate logistic regression model derived from established thoracic oncology risk models (Mayo Clinic Model / Brock University PanCan Risk Calculator):

$$\text{Logit}(P_{\text{mal}}) = \beta_0 + \beta_{\text{diam}} \cdot (d - 5.0) + \beta_{\text{spic}} \cdot S + \beta_{\text{density}} + \beta_{\text{upper}} - \beta_{\text{calc}}$$

Where:
- $\beta_0 = -3.40$ (Base population screening prior)
- $\beta_{\text{diam}} = +0.24\text{ per mm above } 5.0\text{ mm}$
- $\beta_{\text{spic}} = +1.65 \cdot \text{SpiculationIndex}$
- $\beta_{\text{density}} = +0.55$ for Part-Solid lesions; $-0.30$ for pure GGN
- $\beta_{\text{upper}} = +0.35$ for Upper Lobe anatomical location (predilection for bronchogenic carcinoma)
- $\beta_{\text{calc}} = +3.80$ for verified benign calcification

Probability of Malignancy:
$$P_{\text{malignancy}} = \frac{1}{1 + e^{-\text{Logit}}}$$

---

## 4. Official ACR Lung-RADS v2022 Decision Tree

Every nodule candidate is automatically staged according to the **American College of Radiology (ACR) Lung CT Screening Reporting & Data System (Lung-RADS v2022)**:

| Category | Clinical Meaning | Criteria | Malignancy Risk | Clinical Recommendation | UI Color |
| :--- | :--- | :--- | :---: | :--- | :---: |
| **Category 2** | **Benign** | Solid $<6\text{ mm}$, or pure GGN $<30\text{ mm}$, or benign calcification | $<1\%$ | Routine annual LDCT screening in 12 months | `#00e676` (Green) |
| **Category 3** | **Probably Benign** | Solid $6\text{--}8\text{ mm}$, or non-solid $\ge 30\text{ mm}$ | $1\text{--}2\%$ | Short-interval follow-up LDCT in 6 months | `#ffaa00` (Amber) |
| **Category 4A** | **Suspicious** | Solid $8\text{--}15\text{ mm}$, or part-solid with solid core $6\text{--}8\text{ mm}$ | $5\text{--}15\%$ | Low-dose CT in 3 months; PET/CT if solid core $\ge 8\text{ mm}$ | `#ff8000` (Orange) |
| **Category 4B** | **Very Suspicious** | Solid $\ge 15\text{ mm}$, or solid core $\ge 8\text{ mm}$ | $>15\%$ | Diagnostic chest CT with IV contrast, PET/CT, or biopsy | `#ff2d55` (Crimson) |
| **Category 4X** | **High Suspicion with Malignant Features**| Category 3 or 4A with marked spiculation, lymphadenopathy, or retraction | $>15\%$ | Immediate diagnostic CT, PET/CT, and urgent biopsy/thoracic surgery consultation | `#ff2d55` (Crimson) |

---

## 5. Enhanced DICOM-SR Standard Integration

The findings are encapsulated into [`data/output_sr/{study_name}_sr.dcm`](file:///c:/Users/User/Documents/VyntaCAD/VyntaMonai/data/output_sr/pasien_01_sr.dcm) conforming to **Enhanced SR Storage** (`1.2.840.10008.5.1.4.1.1.88.22`):
- `NUM`: Equivalent Diameter (UCUM: `mm`)
- `NUM`: Probability of Malignancy (Code: `111024`, UCUM: `%`)
- `TEXT`: ACR Lung-RADS Assessment Category (`Category 2`, `Category 3`, `Category 4X`)
- `TEXT`: Margin Texture & Spiculation Index
- `TEXT`: Official Clinical Management Recommendation

---

## 6. Web Workstation Manual (CADx Tools)

1. **Lung-RADS Triage Filter Bar**:
   - Quick buttons: `[ All (109) ]`, `[ 🚨 Cat 4 (12) ]`, `[ ⚠️ Cat 3 (3) ]`, `[ 🟢 Cat 2 (94) ]`.
   - Clicking a filter immediately filters both the sidebar card list and the canvas bounding boxes.
2. **Severity Color-Coded Target Overlays**:
   - High-suspicion lesions display with **Neon Crimson (`#ff2d55`)**.
   - Intermediate lesions display with **Amber Yellow (`#ffaa00`)**.
   - Benign lesions display with **Emerald Green (`#00e676`)**.
   - Floating HUD labels show: `NODULE #X | Ø...mm | Cat 4X | Mal: ...%`.
3. **Interactive Diagnostic Profile Inspector**:
   - Clicking any nodule card expands the **🔬 CADx Diagnostic Profile**:
     - Visual Malignancy Risk Gauge ($0\%\text{--}100\%$).
     - Official ACR Lung-RADS Assessment badge.
     - Spiculation index & Calcification pattern.
     - Actionable Clinical Recommendation.
