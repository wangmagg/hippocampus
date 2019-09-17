# Analysis of Hippocampus Thickness
### Summary
  Construct thickness map of CA1, CA2, CA3, and subiculum region using segmented high resolution ex vivo MRI images
  
### Method
  1. Determine midsurface through manual point selection and B-spline interpolation (Midsurface.py)
  2. Create triangulated meshes for the midsurface and target surface (mesh.py)
  3. Perform current-based optimization to optimize position of midsurface
  4. Perform current-based optimization to optimize thickness
  5. Construct thickness map on rectangular grid with global curvature considerations
    
  Note: For fast optimization, GPU is required
  
#### See pipeline.py for full pipeline code.
