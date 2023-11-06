# CMCC-Hybrid-EBM
 
## Table of Contents:
- [Description](#description)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Contact](#contact)
- [License](#license)

## Description [[to ToC]](#table-of-contents)

This project concerning the development of an hybrid model for the estimation of the salt-wedge intrusion length (L<sub>x</sub>) and the salinity concentration in the Po River (Po-Goro-Branch). 
The Hybrid-EBM has been implemented by combining the ML-based model with the fully-physics EBM model. In particular, the first and the second component of this new model has been obtained replacing the two equations of the fully-physics model by ML algorithms like Random Forest and LSBoost. 

## Project Structure [[to ToC]](#table-of-contents)

The project structure is organized as follows:

- `data` folder contains two subfolders:
  - `raw` folder contains three subfolders with the raw data:
	- `Component-1-Lx`: Contains Excel files with the raw dataset related to the Component-1 of Hybrid-EBM.
	- `Component-2-Ck`: Contains three subfolders:
		- `Ck-Obs-LSBoost`: Contains Excel files with the raw dataset related to the Component-2 of Hybrid-EBM, generated using the Component-1-LSBoost.
		- `Ck-Obs-RF`: Contains Excel files with the raw dataset related to the Component-2 of Hybrid-EBM, generated using the Component-1-RF.
		- `Input-Features-For-Synthetic-Ck-Obs-Generation`: Contains Excel files with the raw dataset required to generate the synthetic Ck observations.
	- `Component-4-Sul`: Contains Excel files with the raw dataset related to the component-4 of hybrid-model.
  - `processed` folder contains two subfolders with the precessed training and test dataset:
	- `Component-1-Lx`: Contains Excel files with the processed dataset related to the Component-1 of Hybrid-EBM.
	- `Component-2-Ck`: Contains two subfolders:
		- `Ck-Obs-LSBoost`: Contains Excel files with the processed dataset related to the Component-2 of Hybrid-EBM, generated using the Component-1-LSBoost.
		- `Ck-Obs-RF`: Contains Excel files with the processed dataset related to the Component-2 of Hybrid-EBM, generated using the Component-1-RF.
- `src` folder contains the source code files and subfolders:
  - `models` folder contains four subfolders, each of one contains the main scripts for running the modeling and analysis :
	 - `Component-1-Lx` folder includes the following files:
		- `train_model_component_1_lx.m`: The script to trains ML models for Component-1 of Hybrid-EBM.
	 - `Component-2-Ck` folder includes the following folder and file:
		- `train_model_component_2_ck.m`: The script to trains ML models for Component-2 of Hybrid-EBM.
		- `Component-2-1-Generate-Syntethic-Ck-Observations` folder with the files:
			- `run_equation_synthetic_ck_observations.m`: The script to run creation of new synthetic observations for ck values.
			- `generate_synthetic_ck.m`: The function (equation) to generate the ck observations.
	 - `Component-3-Qul` folder includes the following files:
		- `compute_qul.m`: The function (equation) to compute the Component-3 of Hybrid-EBM.
	 - `Component-4-Sul` folder includes the following files:
		- `run_experiment_component_4_sul.m`: The script to run the Component-4 of Hybrid-EBM.
		- `compute_sul.m`: The function (equation) to compute the Component-4 of Hybrid-EBM.
  - `lib` folder contains libraries for analysis, machine learning, and utility functions.
- `models` folder contains the trained models, models predictions and model summaries for each component of hybrid-model.
- `reports` folder contains a brief reports of the analysis with graphics and figures.

## Requirements [[to ToC]](#table-of-contents)
- MATLAB Version 9.14 (R2023a) (https://it.mathworks.com/products/matlab.html)
- Statistics and Machine Learning Toolbox Version 12.5 (R2023a) (https://it.mathworks.com/products/statistics.html)
- Parallel Computing Toolbox Version 7.8 (R2023a) (https://it.mathworks.com/products/parallel-computing.html)

## Setup [[to ToC]](#table-of-contents)
To set up the project, follow these steps:

1. Clone the repository: 
    ```
	https://github.com/CMCC-Foundation/CMCC-Hybrid-EBM.git
    ```
2. Navigate to the project directory:
    ```
    cd CMCC-Hybrid-EBM
    ```

## Usage [[to ToC]](#table-of-contents)
To run the experiment follow these steps:

1. Run the script to train machine learning models for the Component-1 of Hybrid-EBM (L<sub>x</sub>):
````
 \src\models\Component-1-Lx\train_model_component_1_lx.m
````
2. Run the script to generate synthetic observations for the C<sub>k</sub> coefficient:
````
\src\models\Component-2-Ck\Component-2-1-Generate-Syntethic-Ck-Observations\run_equation_synthetic_ck_observations.m
````

3. Run the script to train machine learning models for the Component-2 of Hybrid-EBM (C<sub>k</sub>):
````
\src\models\Component-2-Ck\train_model_component_2_ck.m
````

4. Run the script to compute Component-4 of Hybrid-EBM (S<sub>ul</sub>):
````
\src\models\Component-4-Sul\run_experiment_component_4_sul.m
````


## Contact [[to ToC]](#table-of-contents)

For any questions or inquiries, please contact [Leonardo Saccotelli](mailto:leonardo.saccotelli@cmcc.it), [Rosalia Maglietta](mailto:rosalia.maglietta@cnr.it) or [Giorgia Verri](mailto:giorgia.verri@cmcc.it).

## License [[to ToC]](#table-of-contents)

This project is licensed under the [GPL 3.0 License](LICENSE).
