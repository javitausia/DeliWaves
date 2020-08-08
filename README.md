[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/javitausia/DeliWaves/master)

# DeliWaves

### Calibration and validation of hindcast information

DeliWaves is an open-source software toolkit written in python that enables the users to perform different actions with wave data. A description of what has been done can be seen in this paper:

* Javier Tausia Hoyal, 2020. (attached)

## 1. Description

In coastal engineering studies exist some different models and actions that are usually performed but differently by similar investigation groups. The aim of this repository is proportioning the user an easy way to perform these widely used actions worldwide. For the correct usage of the different notebooks and python scripts (it is also required in CalValWaves), the only thing needed is an ordered dataframe with the reanalysis information of the location of interest. For the rest, everything is proportiones, as once this initial dataframe exists, all the code can be run.

## 2. Data download

As it has been previously mentioned, the only thing needed is an initial dataframe with the wave historic reanalysis. More information about this downloading will be posted, but while this part is added, the shape of this dataframe must be as the one shown below:

![dataframe](/images/data/dataframe.png)

## 3. Main contents

This repository proportionates code to perform the following actions:

* MDA (Maximum Dissimilarity Algorithm)
* [SWAN](http://swanmodel.sourceforge.net/download/download.htm) (Simulating WAves Nearshore)
* RBF (Radial Basis Functions reconstruction)
* Foecast (Predictions of the 7-days wave climate worldwide)

### 3.1 MDA

The main goal of the workflow followed is the reconstruction of the wave climate in coastal areas and for this purpose, the first step is the selection of the maximum dissimilar cases in the initial dataset. This algotithm can be used for other objectives too. This is the first part that has to be performed and is briefly explained in:

- [MDA](./mda/mda_notebook.ipynb): MDA explanatory notebook

With this notebook, an image as the one shown below can be otained:

![mdass](/images/mda/mdass.png)

### 3.2 SWAN

Once the most dissimilar cases have been selected, they are propagated to coastal waters obtaining results as the one shown in a region in the cantabric sea, in the north of Spain:

![liencres](/images/swan/liencres.png)

These propagations are performed running a nummerical model and using the friendly software developed by the [TU Delft](https://www.tudelft.nl/) (Delft University of Technology) which is called [SWAN](http://swanmodel.sourceforge.net/download/download.htm) (Simulating WAves Nearshore). This software propagates windseas and swells to coast from offshore points as the one required for the development of this entire DeliWaves toolbox. The autocontent notebook can be found in:

- [SWAN](./swan/swan_notebook.ipynb): SWAN explanatory notebook

### 3.3 RBF

Radial basis function (RBF) interpolation is an advanced method in approximation theory for constructing high-order accurate interpolants of unstructured data, possibly in high-dimensional spaces. The interpolant takes the form of a weighted sum of radial basis functions. RBF interpolation is a mesh-free method, meaning the nodes (points in the domain) need not lie on a structured grid, and does not require the formation of a mesh. It is often spectrally accurate and stable for large numbers of nodes even in high dimensions. The figure below illustrates how various of these radial basis functions form jointly the required final function:

![RBF](/images/rbf/rbf.png)

This description matches perfectly the purpose of the study, as our goal is to obtain the historical time series but just using the cases propagated, which is an interpolation by definition. The autocontent notebook with all the necessary information can be found at:

- [RBF](./rbf/rbf_notebook.ipynb): RBF explanatory notebook

### 3.4 Spectra and free surface reconstructions

...

### 3.5 Forecast

Forecast predictions have been also included worldwide. Everyday, thousands of surfers request information from websites such as [surf forecast](https://es.surf-forecast.com/breaks/Liencres/forecasts/latest/six_day) or [windguru](https://www.windguru.cz/48699), but these predictions are proportioned in offshore locations, so the actual prediction in the coast is not always the one available offshore. With the scripts existent in the toolbox, coastal forecast with a good resolution can be obtained.

- [Forecast class](./forecast/forecast.py): Forecast main class
- [Forecast main notebook](./forecast/forecast_notebook.ipynb): Forecast explanatory notebook

Using this code, a GIF as the one shown below will be obtained:

![FoGIF](/images/forecast/forecastgit.gif)

And the predictions have the next aspect:

![FoPred](/images/forecast/forecast.png)

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
