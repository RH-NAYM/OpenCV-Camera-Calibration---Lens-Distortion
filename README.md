# 💡 Camera Calibration & Lens Distortion Correction in OpenCV (Python Tutorial)
---
[![main branch](https://img.shields.io/badge/branch-main-red?style=flat&logo=git&logoColor=white)](https://github.com/RH-NAYM/OpenCV-Camera-Calibration---Lens-Distortion/tree/main)
#

<p align="center">
  <a href="https://opencv.org/" target="_blank">
    <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv&logoColor=white">
  </a>
  <a href="https://www.python.org/" target="_blank">
    <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white">
  </a>
  <a href="https://jupyter.org/" target="_blank">
    <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white">
  </a>
  <a href="https://numpy.org/" target="_blank">
    <img src="https://img.shields.io/badge/Numpy-Numerical-lightblue?logo=numpy&logoColor=white">
  </a>
  <a href="https://matplotlib.org/" target="_blank">
    <img src="https://img.shields.io/badge/Matplotlib-Visualization-orange?logo=matplotlib&logoColor=white">
  </a>
</p>

# 📌 Overview
This repository provides a comprehensive guide to Camera Calibration and Lens Distortion Correction using OpenCV in Python. Camera calibration is crucial to:

Correct radial and tangential lens distortions

Accurately map 3D world coordinates to 2D image points

Stabilize augmented reality overlays

Enable precise stereo vision and 3D measurements

The notebook covers from fundamental theory to advanced calibration workflows, including fisheye and stereo calibration scenarios.
---

**Key Features:**
Camera Intrinsics & Extrinsics: Learn how to obtain the intrinsic matrix, distortion coefficients, and camera poses.

Lens Distortion Correction: Remove barrel and pincushion distortions using OpenCV.

Subpixel Corner Detection: Achieve high-precision chessboard corner detection for calibration.

Reprojection Error Analysis: Quantify calibration accuracy.

Advanced Use Cases: Fisheye lens and stereo camera calibration.

This repository is ideal for mid-level OpenCV users up to advanced practitioners who want to master camera calibration.


# 📁 Project Structure
.
├── calib_images/                                  # Sample chessboard images for calibration
│   ├── left01.jpg
│   ├── left02.jpg
│   ├── left03.jpg
│   ├── left04.jpg
│   ├── left05.jpg
│   ├── left06.jpg
│   ├── left07.jpg
│   ├── left08.jpg
│   └── left09.jpg
├── camera_calib.npz                               # Saved calibration parameters (camera matrix, distortion coefficients)
├── OpenCV-Camera-Calibration-and-Lens-Distortion.ipynb  # Full tutorial notebook
├── README.md                                      # Project documentation
└── tools/                                        # Utility module
    └── tools.py                                   # Helper functions for image loading & visualization

# 📋 Table of Contents (Notebook Sections)
---
```bash
1. Introduction to Camera Calibration
2. Understanding Camera Intrinsics and Distortion
3. Preparing Chessboard Object and Image Points
4. Detecting Chessboard Corners with cv2.findChessboardCorners
5. Performing Camera Calibration with cv2.calibrateCamera
6. Undistorting Images using cv2.getOptimalNewCameraMatrix
7. Computing Reprojection Error
8. Saving and Loading Calibration Parameters
9. Advanced Use Cases: Fisheye and Stereo Calibration
10. Practical Applications in AR and 3D Measurement
```

# 🧠 What You’ll Learn in Camera Calibration

---

### Mathematical Mapping – The Core of Camera Projection

#### 1. Pinhole Camera Projection Model

The fundamental equation that maps a 3D world point to a 2D image point (before distortion):

$$
s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
=
\mathbf{K}
[\mathbf{R} \mid \mathbf{t}]
\begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}
$$

Where:
- $\mathbf{K}$ → **camera intrinsic matrix** (focal lengths + principal point)
- $[\mathbf{R} \mid \mathbf{t}]$ → **extrinsic parameters** (rotation matrix $\mathbf{R}$ + translation vector $\mathbf{t}$)
- $s$ → arbitrary positive **scale factor** (homogeneous coordinate normalization)
- $(X, Y, Z)$ → 3D point in world coordinates
- $(u, v)$ → corresponding 2D image coordinates (in pixels)

#### 2. Lens Distortion Model (OpenCV / Brown–Conrady)

Real cameras introduce non-linear distortions. The most widely used model is:

**Radial distortion** (barrel or pincushion effect):

$$
x_{\text{dist}} = x \left(1 + k_1 r^2 + k_2 r^4 + k_3 r^6 \right)
$$
$$
y_{\text{dist}} = y \left(1 + k_1 r^2 + k_2 r^4 + k_3 r^6 \right)
$$

**Tangential distortion** (decentering / lens misalignment):

$$
x_{\text{dist}} = x + \left[ 2p_1\,xy + p_2 \left(r^2 + 2x^2\right) \right]
$$
$$
y_{\text{dist}} = y + \left[ p_1 \left(r^2 + 2y^2\right) + 2p_2\,xy \right]
$$

Where:
- $r^2 = x^2 + y^2$ (normalized coordinates)
- $k_1, k_2, k_3$ → radial distortion coefficients
- $p_1, p_2$ → tangential distortion coefficients

*(Note: OpenCV supports up to $k_4,k_5,k_6$ when `CALIB_RATIONAL_MODEL` is used.)*

### Practical Applications After Calibration

Once you have accurate $\mathbf{K}$ and distortion coefficients $\mathbf{D}$, you can:

- **Correct distorted images** → remove barrel/pincushion effects
- **Improve 3D reconstructions** → accurate triangulation and stereo depth
- **Enable stable AR/VR** → precise overlay of virtual objects
- **Perform metric measurements** from images (with known scale)
- **Enhance** feature matching, homography estimation, and PnP solutions

### How to Evaluate Calibration Quality

The gold standard metric is the **mean reprojection error**:

$$
\text{Mean Reprojection Error} = \frac{1}{N} \sum_{i=1}^{N} \left\| \mathbf{m}_i - \hat{\mathbf{m}}_i \right\|_2
$$

- $\mathbf{m}_i$ → observed (detected) image point
- $\hat{\mathbf{m}}_i$ → projected point using estimated $\mathbf{K}$, $\mathbf{D}$, $\mathbf{R}_i$, $\mathbf{t}_i$
- **Excellent** calibration → **< 0.5 pixels**
- **Good** → **0.5–0.8 pixels**
- **> 1 pixel** → usually indicates problems (poor images, inaccurate corners, insufficient views, or unmodeled distortion)

Mastering these concepts allows you to build robust, real-world computer vision systems!

# 🛠️ Technologies Used
---
- `Python 3.x`
- `OpenCV` for morphological image processing
- `NumPy` for array operations
- `Matplotlib` for visualization
- `Jupyter Notebook` for interactive experimentation


# 📦 Installation
---
## 1️⃣ Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate    # Linux / macOS
venv\Scripts\activate       # Windows
```
## 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
# 🚀 How to Run
---
**Option 1: Jupyter Notebook (Local)**
- Install Jupyter if needed: `pip install notebook`.
- Launch Jupyter: `jupyter notebook`.
- Open `OpenCV-Camera-Calibration-and-Lens-Distortion.ipynb` and run cells sequentially.
    - Notebook will automatically download a placeholder image if testImage.jpg is missing.


**Option 2: Google Colab**
- Upload `OpenCV-Camera-Calibration-and-Lens-Distortion.ipynb` to Colab.
- Install dependencies: `!pip install -r requirements.txt`.
- Run all cells for interactive demonstrations.


# ✅ Summary
---
Camera calibration is essential to remove lens distortions and obtain accurate measurements.

Distortion correction stabilizes AR overlays and stereo vision pipelines.

Visualizing object points, image points, and reprojection error is critical to understanding calibration quality.

Advanced calibration (fisheye, stereo) enables wide-angle and multi-camera applications.

# 🍴 Real-World Applications
---
Augmented Reality: Accurately overlay 3D objects onto real-world scenes.

3D Measurement: Compute precise distances and dimensions.

Robotics: Enable reliable vision-based navigation.

Stereo Vision: Generate depth maps and 3D reconstructions.

Industrial Automation: Calibrate cameras for inspection and measurement tasks.

# 📝 Contribution
Feel free to open an issue or submit a pull request to add advanced calibration examples or additional lens distortion scenarios.
