# Dehesa Tree Crown Segmentation (YOLOv11)

An automated deep learning-based tool for forest inventory and tree crown segmentation in Mediterranean "Dehesa" ecosystems. This project leverages **YOLOv11** and geo-spatial libraries to transform aerial imagery into actionable forestry data.

### Project Status: Beta & Preliminary Approach
This repository represents a **preliminary approximation** to automated crown detection. Please note:
* The current version is in **Beta**.
* Results and model precision are subject to ongoing refinement.
* The workflow is designed for research and testing purposes, not yet for production environments.

### ontributions & Feedback
I am actively looking to improve this tool! **Feedback, bug reports, and contributions are highly encouraged.** If you have suggestions regarding model optimization, geospatial logic, or edge-case handling in complex forest structures, please feel free to open an issue or submit a pull request.

---

## Key Features
* **Tiling & Overlap Logic**: Efficiently processes large high-resolution orthophotos by splitting them into manageable windows.
* **Geospatial Accuracy**: Uses Affine transformations to map pixel-level detections to real-world CRS coordinates.
* **Biometric Calculations**: Automatically computes:
    * Tree density (stems/ha).
    * Canopy Cover (CC %).
    * Individual and mean crown diameters (m).
* **GIS Integration**: Exports results directly to **GeoPackage (.gpkg)**, compatible with QGIS and ArcGIS.

---

## Sample Inventory Statistics
The following results were generated from a sample analysis of a **75.22 hectare** study area:

| Metric | Result |
| :--- | :--- |
| **Detected Stems** | 3,643 trees |
| **Tree Density** | 48.43 stems/ha |
| **Canopy Cover (CC)** | 18.08 % |
| **Mean Crown Diameter** | 6.68 m |
| **Max Crown Diameter** | 13.44 m |

---

## Usage Example
The core logic is structured into a modular `ForestSegmenter` class for easy integration:

```python
from detector import ForestSegmenter

# Initialize the segmenter
segmenter = ForestSegmenter(model_path="path/to/weights.pt", tile_size=960)

# Run detection and get a GeoDataFrame
results_gdf = segmenter.predict_crowns(image_path="dehesa_ortho.tif")

# Results include calculated crown diameters using:
# Diameter = 2 * sqrt(Area / pi)
