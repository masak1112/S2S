# Introduction to subseasonal to seasonal forecats

This project aims to develop an AI model for probabilistic forecasting at the subseasonal to seasonal timescale.

# Prepare your datasets
The experiments rely on the ERA5 dataset. We have stored and processed the data in `.h5` format on the Midway, Derecho, and Stampede3 systems.

 - For the user of Midway from RCC of the university of Chicago: the corresponding path to the data is: `/project/pedramh/h5data/h5data`

 - For the users of Derecho from NCAR: the corresponding path to the data is: `/glade/campaign/univ/uchi0014/yqsun/pangu_s2s/h5data`

 - For the users of stampede3 system from TACC: the corresponding path to the data is: `/scratch/08198/tg874973/pangu-s2s/h5data`


# Installation

## Access to the code 
Clone this repository by running the following command in your personal target directory:

```
git clone git@github.com:masak1112/S2S.git
```

To the source code by running the command:

```
cd src
```

## Change branch

Once you access to the code repo, switch to the branch `bing_issue#004_vae_crps_v2_fix_dsi` by the following command

```
git checkout --track origin/bing_issue#011_add_evaluation_metric_jupiter
```


# Set up virtual enviornment

The code can be set-up on different operating systems. The related virtual environment can be set up with the help of the `conda` command. The enviornment request is listed in the `src/enviornment.yml` file. You can simply use the following command:

```
conda env create -f environment.yml --prefix /path/to/myenv
```

If you are on the Midway, Derecho and stampede3 system you can also use the virtual enviornment we already established and activite your env by conda: 

- Midway: 
```
conda activate /project/pedramh/bing/env
```

- Derecho:
```
conda activate  /glade/work/zand/anaconda/py311
```

- Stampede3: 
```
conda activate /home1/10786/bgong1/stampede3/env
```

# Setup wandb account

If beginning a training run for the first time, log in to your weights and biases account first. This can be done by activating the environment using the information above, then running `wandb login`. You'll be prompted to open a link to login to your account, then will receive an access code to enter in the command line.

# Getting Started

## Run the workflow using HPC script templates

To help you submit the jobs to different systems, we prepare the HPC job submission templates under `src/HPC_scripts` for training and inference.

For each template, you need to change your working directory path and configuration file path


## Configure your yaml file for each experiment. 

1. Before beginning a training or inference run, you'll first need to create a configuration file. These should be stored in the `src/config` directory. The naming convention I've been using is `exp${id}.yaml`.  You can use `exp1.yaml` as an example to run on Midway system.

2. Edit your configuration file to set the parameters you'd like to use for the run. Remember to set the `data_dir` to point to the data location for the cluster you're using.


4. To start a training/inference:

- On Midway and Stampede3

 run `sbatch ${cluster}_training.sh ` 


- on Derecho:

 run `qsub ncar_training.sh`


