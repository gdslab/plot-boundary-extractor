# plot-boundary-extractor
plot boundary extractor using segment anything model


# Install virtual environment
```bash
conda create -n pbe -c conda-forge gdal python=3.10 -y
conda activate pbe
```

# Install plot boundary extractor
```bash
pip install plot-boundary-extractor
```

# Install torch
```bash
(CPU)
pip install torch torchvision
(GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

# Install SAM and download checkpoint file
```bash
pip install git+https://github.com/facebookresearch/segment-anything.git

wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```