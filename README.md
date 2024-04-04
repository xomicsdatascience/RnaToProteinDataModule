# RnaToProteinDataModule

My classes for using CPTAC and AD data reproducibly. My scripts for using those classes are also included.

## M1 Mac Issues
I should note that pytorch gives me trouble fairly often. I need to run the following to set up the environment appropriately. I'm putting it here so I don't lose any more hours reinventing this wheel.

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
