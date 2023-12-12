# RHINO: Regularizing the Hash-based Implicit Neural Representation

PyTorch implementation of RHINO.

## Pipeline

## Setup

We provide a conda environment setup file including all of the above dependencies. Create the conda environment RHINO by running:

    conda env create -f Rhino.yaml

## Task

#### Image Representation

- Training

For tasks like fitting a single image, we prepare a test image in the `data` folder.

To train image representations, use the config files in the `config` folder. For example, to train on the provided image, run the following

    cd img
    python train_img_interp.py --config ./config/img.ini

After the image representation has been trained, the results of the image will appear in the `img/log/<experiment_name>` folder, where `<experiment_name>` is the subdirectory in the `log` folder corresponding to the particular training run.

#### 3D Shape Representation

- Datasets

  &#x20;Datasets can be downloaded using the `sdf/download_datasets.py` script.

- Training

<!---->

    cd sdf/experiments
    python train_sdf_RHINO.py --config ./config/sdf/bacon_armadillo.ini

After the 3D shape representation is trained, the results will appear in the `sdf/log/<experiment_name>` folder

- Rendering

<!---->

    cd sdf/experiments
    python render_sdf_RHINO.py

You can obtain the rendering results of different data by modifying `bacon_names`.

## Citation

    @article{zhu2023rhino,
          title={RHINO: Regularizing the Hash-based Implicit Neural Representation},
          author={Zhu, Hao and Liu, Fengyi and Zhang, Qi and Cao, Xun and Ma, Zhan},
          journal={arXiv preprint arXiv:2309.12642},
          year={2023}
        }
