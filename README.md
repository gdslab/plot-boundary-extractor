# `plot-boundary-extractor` module

This module is to extract field experiment plot boundaries for agricultural applications. This module was initially developed for wheat breeding trials, but it can also extract plot boundaries for other row crops. More tests will be done and shared in the future. 

This repository is associated with our research on **Developing a segment anything model-based framework for automated plot extraction**, which has been submitted to *Precision Agriculture* journal and is currently under review. Citation information will become available when the paper is published. 

---

*This module was developed by [Geospatial Data Science Lab](https://gdsl.org) at Purdue University. If you have any questions, you can create an issue in this repository or contact Hansae Kim (kim4012@purdue.edu) or Jinha Jung (jinha@purdue.edu).*

## `plot-boundary-extractor` installation

### Set up a virtual environment

First, we need to install `plot-boundary-extractor` on the machine you want to run this example tutorial. We will set up a new Python Virtual Environment called `pbe`. 

```bash
$ conda create -n pbe -c conda-forge gdal python=3.10 -y
```

We need to activate the newly created virtual environment before we start.

```bash
$ conda activate pbe
```

### Install `plot-boundary-extractor` module

Now we need to install the `plot-boundary-extractor` module using `pip`.

```bash
$ pip install plot-boundary-extractor
```

### Install torch

The `plot-boundary-extractor` module uses `torch` module, so we need to install it on the machine you would like to run this example tutorial. 

#### CPU version

If your machine is not equipped with NVIDIA GPU, then you will have to install the `torch` without GPU support.

```bash
$ pip install torch torchvision
```

#### GPU version

If your machine is equipped with NVIDIA GPU, then you will have to install the `torch` with GPU support.

```bash
$ pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Install `SAM`

The `plot-boundary-extractor` uses `SAM` to detect plot boundaries automatically, so you must install `SAM` using the below command.

```bash
$ pip install git+https://github.com/facebookresearch/segment-anything.git
```

You also have to download a pre-trained model using the below command. Please make a note of where you're downloading the pre-trained model. We will need this path information in the later tutorial.

```bash
$ wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Now, the installation is done, and you're ready to extract plot boundaries using the `plot-boundary-extractor` module. An example [Jupyter Notebook](./plot_boundary_extractor_demo.ipynb) is available.
