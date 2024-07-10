## Manubot Project

This repository contains the analysis code for the Manubot project [_Multi-Dataset Integration and Residual Connections Improve Proteome Prediction from Transcriptomics Using Deep Learning_](https://github.com/xomicsdatascience/transcriptome-proteome-nas-manubot).

### Scripts

Files used for running our code are found under the `scripts` directory. The files under `scripts/NAS` are based on [the Multi-Objective NAS with Ax](https://pytorch.org/tutorials/intermediate/ax_multiobjective_nas_tutorial.html) tutorial. The files under `scripts/5x2` were for generating the boxplot inside figure 1. `scripts/SHAP` contains preliminary code for running SHAP on the model. Note that the branch `input_residual_connection` contains a more complete setup for the SHAP runs used in the paper.

### src

The two main files to consider are `src/RnaToProteinDataModule/NasModel.py`, which contains the deep learning model used, and `src/RnaToProteinDataModule/RnaToProteinDataModule.py`, which performs the bulk of the data processing steps. 

## M1 Mac Issues
`pytorch` often gives Mac users trouble. Below is a script that bypasses most errors when installing the necessary packages for a Mac.

```
conda create -n <env name>
conda activate <env name>
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 -c pytorch
pip3 install ax-platform
pip3 install pytorch-lightning
pip3 install -U 'tensorboardX'
pip3 install tensorboard
pip3 install torchx
pip install -e <path to RnaToProteinDataModule>
```

In a nutshell: Running a GPU on M1 Macs doesn't always work perfectly well. Interestingly, the most recent pytorch versions sometimes run with the mps GPU without a hitch, but other times will not. I don't know why the inconsistency exists. However, installing packages with the above script has it working every time. I do not recommend trying this in a pre-made environment, as things like the python version might skew the install into not working again.
