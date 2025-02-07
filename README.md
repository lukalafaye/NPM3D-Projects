# NPM3D-Projects

This repository contains practical exercises (TPs) related to point cloud processing, 3D reconstruction, and classification. Each lab contains specific tasks related to manipulating and analyzing point clouds.

## Table of Contents

- [Lab 1: Point Cloud Structures and Neighborhoods](#lab-1-point-cloud-structures-and-neighborhoods)
- [Lab 2: Iterative Closest Point (ICP) Algorithm](#lab-2-iterative-closest-point-icp-algorithm)
- [Lab 3: Neighborhood Descriptors](#lab-3-neighborhood-descriptors)
- [Lab 4: 3D Reconstruction](#lab-4-3d-reconstruction)

## Lab 1: Point Cloud Structures and Neighborhoods
[Lab 1 Report](Lab1/TPX_LAFAYE_DE_MICHEAUX_KOUVATSEAS.pdf)  
[Lab 1 PDF](Lab1/TP1_Structures_Neighborhoods.pdf)

**Objective:**
- Load and visualize point clouds using Cloud Compare.
- Perform point cloud transformations in Python.
- Implement subsampling methods (e.g., decimation).
- Compute point neighborhoods using brute-force search and KD-trees.

### Files:
- **[`Lab1/code/neighborhoods.py`](Lab1/code/neighborhoods.py)** – Implements neighborhood search algorithms.
- **[`Lab1/code/ply.py`](Lab1/code/ply.py)** – Handles PLY file I/O.
- **[`Lab1/code/subsampling.py`](Lab1/code/subsampling.py)** – Implements point cloud decimation.
- **[`Lab1/data/indoor_scan.ply`](Lab1/data/indoor_scan.ply)**, **[`Lab1/data/bunny.ply`](Lab1/data/bunny.ply)** – Example point clouds for testing.

---

## Lab 2: Iterative Closest Point (ICP) Algorithm
[Lab 2 Report](Lab2/TPX_LAFAYE_DE_MICHEAUX_KOUVATSEAS.pdf)  
[Lab 2 PDF](Lab2/TP2_Recalage_ICP.pdf)

**Objective:**
- Understand and implement the Iterative Closest Point (ICP) algorithm.
- Align point clouds using rigid transformations.
- Evaluate ICP convergence using RMS error.

### Files:
- **[`Lab2/code/ICP.py`](Lab2/code/ICP.py)** – Implements the ICP algorithm.
- **[`Lab2/code/ply.py`](Lab2/code/ply.py)** – Handles PLY file I/O.
- **[`Lab2/code/visu.py`](Lab2/code/visu.py)** – Visualization tools for ICP results.
- **[`Lab2/data/bunny_original.ply`](Lab2/data/bunny_original.ply)**, **[`Lab2/data/bunny_perturbed.ply`](Lab2/data/bunny_perturbed.ply)** – 3D models used for alignment.
- **[`Lab2/data/ref2D.ply`](Lab2/data/ref2D.ply)**, **[`Lab2/data/data2D.ply`](Lab2/data/data2D.ply)** – 2D point clouds for testing ICP.

---

## Lab 3: Neighborhood Descriptors
[Lab 3 Report](Lab3/TPX_LAFAYE_DE_MICHEAUX_KOUVATSEAS.pdf)  
[Lab 3 PDF](Lab3/TP3_Descripteurs.pdf)

**Objective:**
- Compute normal vectors for point clouds.
- Use Principal Component Analysis (PCA) for feature extraction.
- Compare different neighborhood definitions (radius vs. k-nearest neighbors).

### Files:
- **[`Lab3/code/descriptors.py`](Lab3/code/descriptors.py)** – Implements PCA-based feature extraction.
- **[`Lab3/code/classification.py`](Lab3/code/classification.py)** – Classifies points based on extracted descriptors.
- **[`Lab3/data/Lille_street_small.ply`](Lab3/data/Lille_street_small.ply)** – Urban point cloud dataset.

---

## Lab 4: 3D Reconstruction
[Lab 4 PDF](Lab4/TP4_Reconstruction.pdf)

**Objective:**
- Implement methods for reconstructing 3D surfaces from point clouds.
- Explore different meshing and normal estimation techniques.

### Files:
- **[`Lab4/code/reconstruction.py`](Lab4/code/reconstruction.py)** – Implements 3D surface reconstruction.
- **[`Lab4/data/bunny_normals.ply`](Lab4/data/bunny_normals.ply)** – Point cloud with computed normals.

---

## How to Run the Code
1. Install dependencies: `pip install numpy scipy scikit-learn matplotlib`
2. Run individual scripts within each `code/` directory to execute the respective lab tasks.

For any questions or contributions, feel free to reach out!
