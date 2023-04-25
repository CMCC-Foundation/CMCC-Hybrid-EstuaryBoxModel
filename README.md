# Hybird-EBM-with-Machine-Learning-techniques-Po_Goro_River-Test-Case
This project concerning the development of predictive regression models for the estimation of the salt wedge intrusion length (L<sub>x</sub>) and the salinity concentration in the Po River (Po-Goro-Branch). 
The aim is to compare the results provided by a ML-based models with respect a physics-based model. Also, an Hybrid-Model has been implemented by mixing the ML-based model with the Physics-based model. A significant improvement has been achived both with the ML and Hybrid-based models, with respect the physics-model.
<br>In particular, for the estimation of the salt wedge intrusion length (L<sub>x</sub>), Random Forest and Least-Square Boosting Algorithms have been trained and validated, while for the salinity estimation also an Artificial Neural Network has been built. 
Lastly, for the development of an Hybrid-model, Random Forest and LSBoost algorithms have been trained to deploy a model for the non-dimensional eddy diffusivity coefficient (C<sub>k</sub>) estimation. The predictions given by ML models for (L<sub>x</sub>) and (C<sub>k</sub>) have been used as input for the physic-model to predict the salinity concentration.

## Project structure
Project is organized as follow:
* ```/Machine-Learning-Tools/```
  * ```/ Machine-Learning-Tools / 1-Utility /```
  * ```/ Machine-Learning-Tools / 2-Machine-Learning-Function /```  
  * ```/ Machine-Learning-Tools / 3-Plot-Figure /```
* ```/Parameters-Estimation/```
  * ```/ Parameters-Estimation / Lx_Parameters_Estimation /```
  * ```/ Parameters-Estimation / Ck_Parameters_Estimation /```  
  * ```/ Parameters-Estimation / Salinity_Estimation /```
  * ```/ Parameters-Estimation / Hybrid_Model_Predictions /```

## Prerequisites
* MATLAB Version 9.14 (R2023a) (https://it.mathworks.com/products/matlab.html)
* Statistics and Machine Learning Toolbox Version 12.5 (R2023a) (https://it.mathworks.com/products/statistics.html)
* Parallel Computing Toolbox Version 7.8 (R2023a) (https://it.mathworks.com/products/parallel-computing.html)

## Running the experiments
To run the experiment to train the salt wedge intrusion length (L<sub>x</sub>) ML models:
````
/ Parameters-Estimation / Lx_Parameters_Estimation / run_experiment_train_2003_2012_test_2013_2017.m
````
To run the experiment to train the non-dimensional eddy diffusivity coefficient (C<sub>k</sub>) ML models:

````
/ Parameters-Estimation / Ck_Parameters_Estimation / run_experiment_training_test_2016_2019.m
````
To run the experiment to train the salinity ML models:

````
/ Parameters-Estimation / Salinity_Estimation / run_experiment_training_test_2016_2019.m
````


