# LILIAN

## What is this all about?

This repo contains all of the updated code for our project Reinforcement Learning for longitudinal control.
You can find more documentation here: 
https://porsche-my.sharepoint.com/personal/justin_hartenstein_porsche_de/_layouts/OneNote.aspx?id=%2Fpersonal%2Fjustin_hartenstein_porsche_de%2FDocuments%2FJustin%20%40%20Dr.%20Ing.%20h.c.%20F.%20Porsche%20AG \
All of my saved drivers can be found under savedDrivers in the folders addedCurrentSpeed, steering, and simpleCar.

## Setup

Git clone this repository, create a new virtual environment and pip install the requirements.txt file and you are good to go.

Within Anaconda Powershell prompt:
```python
cd path/to/reinforcement-learning
conda create -n env_thesis python=3.9.15
conda activate env_thesis
pip install setuptools==65.5.0
pip install -r requirements.txt
```
An in-depth setup guide can be found in ```setup.md```.

## Getting Started

To start, we suggest going through the notebooks folder. We recommend starting with the following notebooks:
- *notebooks/plot_track*: Plot a track to get a feeling for the task at hand
- *notebooks/train_driver*: Train an agent using an environment from the folder drivers
- *notebooks/plot_policy*: Plot a learned policy

## TensorBoard

Within Anaconda Powershell prompt: 
```python
conda activate env_thesis
cd path/to/reinforcement-learning
tensorboard --logdir logDirs/SAC/
```

We also include a modified version of sac.py from the Stable-Baselines3 library under SB3-modified/sac_additional_logging_gpu.py (works only for GPU). This file additionally tracks the Q-Values and the entropy for TensorBoard. You need to incorporate the changes in this file into your local SAC.py file which is saved in your virtual environment folder.

## Structure

```
LILIAN
|
|__drivers
|  |         
|  |__A collection of gym environments with different reward functions, action and observation spaces, normalizations,...
|  |__Playground: A simpler Gym environment for experimentation.
|
|__experiments
|  |         
|  |__len_outlook: How many future values to use
|  |__make_movie: Generates a mp4 file that shows how the agent learns over time       
|  |__number_cpu: Does vectorization lead to faster training?  
|
|__logDirs
|  |         
|  |__tfevents files for TensorBoard
|
|__notebooks: Collection of useful notebooks for first time users
|  |         
|  |__behavior_cloning_example: How to pretrain agent using BC
|  |__plot_mat_OUT: Plot Simulink data       
|  |__plot_tensorboard: Collection of functions to plot reward from tfevents files in logDirs         
|  |__train_driver: Train an agent from drivers
|  |__plot_policy: Plot learned policy of an existing driver saved in savedDrivers
|  |__plot_track: Plot a track
|
|__savedDrivers
|  |         
|  |__Collection of saved drivers. We always save the highest reward driver of each run
|
|__savedDrivers
|  |         
|  |__Collection of exported Simulink models. Each folder must consist of the tracks dll file *model_win64.dll* and        optionally the tracks data. Ask my supervisor Jannes Schilling on details for the dll export or see my bachelor thesis (Chapter: Technical Implementation --> From Simulink to Python).
|
|__BA_Nitschke_final: My bachelor thesis, read the final chapter for documentation on experiments.
|
|__Movie: A movie that visualizes how our agent learns
```
# porscheai
# porscheai
