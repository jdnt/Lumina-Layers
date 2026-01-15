# Lumina-Layers

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://gradio.app/)

**An experimental FDM engine exploring layered optical color mixing. Starting with CMYK pixel art, evolving into a universal multi-material photo processor.**

> 🚧 **Current Status**: Phase 1 (CMYK Pixel Art) is stable.
> * **Active**: Integer-geometry generation for standard CMYK filaments.
> * **Next Up**: Manga Mode (High-contrast B&W layering).

---

## 🗺️ Development Roadmap (技术路线图)

Our goal is to build a complete ecosystem for multi-color FDM printing, moving from fixed logic to AI-assisted calibration.

### Phase 1: The Foundation (Current) ✅
* **Target**: Pixel Art & Simple Graphics.
* **Logic**: Fixed CMYK + White + Black (5 colors).
* **Tech**: Integer-based "Slab" geometry to solve slicer overlapping; Basic TD optical mixing.
* **Status**: *Completed & Testing.*

### Phase 2: Manga Mode (Monochrome) 🚧
* **Target**: Manga panels, Ink drawings, High-contrast illustrations.
* **Logic**: Black & White (2-color) layering.
* **Tech**: Using layer thickness to generate grayscale gradients (Lithophane logic but flat) to simulate screen tones (Ben-Day dots).

### Phase 3: Full-Color Photo Engine
* **Target**: Photographs, Anime illustrations, Complex gradients.
* **Logic**: Dynamic Palette Support (2, 4, 6, 8+ colors).
* **Tech**:
    * Advanced Dithering algorithms (Floyd-Steinberg/Atkinson).
    * Weighted color solvers to find the best filament combination from a user's library.

### Phase 4: Web 3D Preview
* **Target**: WYSIWYG (What You See Is What You Get).
* **Tech**: Real-time WebGL/Three.js rendering in the browser.
* **Feature**: Visualize the transmission effect and layer stacking before generating the 3MF file.

### Phase 5: AI Color Calibration (The "Closed Loop")
* **Target**: Perfect color matching for personal filaments.
* **Workflow**:
    1.  User prints a generated "Calibration Test Model" with their own filaments (4/6/8 colors).
    2.  User uploads a photo of the printed model.
    3.  **Engine reverse-engineers the actual TD and RGB values** from the photo.
    4.  Software automatically corrects the mixing algorithm to match the user's physical hardware.

---

## ✨ Key Features (v1.0)

* **Pixel Art Optimized Geometry**:
    * Utilizes a specialized **Integer-based Slab Generation** algorithm.
    * Merges continuous pixels into solid bars along the X-axis.
    * **Solves the "internal wall overlapping" issue** common in voxel-based slicers.
* **CMYK Optical Mixing**:
    * Simulates physical color blending based on **Transmission Distance (TD)**.
* **Sandwich Structure**:
    * Automatically generates a **Face-Down + Spacer + Face-Up** structure.
    * Ideal for double-sided keychains.
* **Smart Processing**:
    * **Auto Background Removal**: Detects and removes solid backgrounds.
    * **Pixel Perfect Scaling**: Forces Nearest-Neighbor resampling.

---

## 🛠️ Installation

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/YourUsername/Lumina-Layers.git](https://github.com/YourUsername/Lumina-Layers.git)
    cd Lumina-Layers
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🚀 Usage

1.  **Run the Engine**
    ```bash
    python app.py
    ```
2.  **Configure**: Upload image, set background removal, input TD values.
3.  **Generate**: Download `.3mf` and import to Bambu Studio (Load as single object).
4.  **Map Filaments**: 0=White, 1=Cyan, 2=Magenta, 3=Yellow, 4=Black.

---

## 📄 License

This project is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)**.

* **Attribution**: You must give appropriate credit.
* **NonCommercial**: You may not use this for commercial purposes.
* **ShareAlike**: If you modify it, you must distribute it under the same license.

[View Full License](LICENSE)
