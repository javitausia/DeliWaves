# DeliWaves

### Calibration and validation of hindcast information

DeliWaves is an open-source software toolkit written in python that enables the users to perform different actions with wave data. A description of what has been done can be seen in this paper:

* Javier Tausia Hoyal, 2020.

## 1. Description

...

## 2. Data download

...

## 3. Main contents

...

## 4. Installation

### 4.1 Create an environment in conda

To run the toolbox you first need to install the required Python packages in an environment. To do this we will see **Anaconda**, which can be downloaded freely [here](https://www.anaconda.com/download/).

Once you have installed it on your PC, open the Anaconda prompt (in Mac and Linux, open a terminal window) and use the `cd` command to go to the folder where you have cloned this repository.

Create a new environment named `deli` with all the required packages:

```
conda env create -f environment.yml -n deli
```
### 4.2 Activate conda environment

All the required packages have been now installed in an environment called `deli`. Now, activate this new environment:

```
conda activate deli
```

## 5. Play

Now everything has been installed, you can now start to play with the python code and the jupyter notebooks. Be careful, as some important parameters can be adjusted during the different processes (construction of the object of the classes, first line code in the jupyter notebook). Nevertheless, parameters used are also shown in the example.

## Additional support:

Data used in the project and a detailed explanation of the acquisition can be requested from jtausiahoyal@gmail.com.

## Author:

* Javier Tausía Hoyal
