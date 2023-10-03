# CMCC-Hybird-EBM-Po_Goro_River-Test-Case
 
## Table of Contents:
- [Description](#description)
- [Data](#data)
- [Source Code](#source-code)
- [Contact](#contact)
- [License](#license)

## Description [[to ToC]](#table-of-contents)

This project concerning the development of an hybrid model for the estimation of the salt-wedge intrusion length (L<sub>x</sub>) and the salinity concentration in the Po River (Po-Goro-Branch). 
The Hybrid-EBM has been implemented by combining the ML-based model with the fully-physics EBM model. In particular, the first and the second component of this new model has been obtained replacing the two equations of the fully-physics model by ML algorithms like Random Forest and LSBoost.

## Data [[to ToC]](#table-of-contents)
The input data for this project is organized as follows:

- `data` folder: Contains two subfolders:
  - `raw` folder: Contains three subfolders:
	- `Component-1-Lx`: Contains Excel files with the raw data related to the component-1 of hybrid-model.
	- `Component-2-Ck`: Contains three subfolders:
		- `Ck-Obs-LSBoost`: Contains Excel files with the raw data related to the component-2 of hybrid-model generated using the Component-1-LSBoost.
		- `Ck-Obs-RF`: Contains Excel files with the raw data related to the component-2 of hybrid-model generated using the Component-1-RF.
		- `Input-Features-For-Synthetic-Ck-Obs-Generation`:Contains Excel files with the raw data required to generate the Ck observations.
	- `Component-4-Sul`: Contains Excel files with the raw data related to the component-4 of hybrid-model.
  - `processed` folder: Contains two subfolders:
	- `Component-1-Lx`: Contains Excel files with the processed training and test dataset.
	- `Component-2-Ck`: Contains two subfolders:
		- `Ck-Obs-LSBoost`: Contains Excel files with the processed training and test dataset generated using the Component-1-LSBoost.
		- `Ck-Obs-RF`: Contains Excel files with the processed training and test dataset generated using the Component-1-RF.
  

## Source Code [[to ToC]](#table-of-contents)

The source code for this project is organized as follows:

- `src` folder: Contains the source code files and subfolders:
  - `models`: Contains four subfolders, each of one contains the main scripts for running the modeling and analysis :
	 - `Component-1-Lx` folder includes the following files:
		- `train_model_component_1_lx.m`: The script to trains ML models for the component-1.
	 - `Component-2-Ck` folder includes the following folder and file:
		- `train_model_component_2_ck.m`: The script to trains ML models for the component-2.
		- `Component-2-1-Generate-Syntethic-Ck-Observations` folder with the files:
			- `run_equation_synthetic_ck_observations.m`: The script to run creation of new synthetic observations for ck values.
			- `generate_synthetic_ck.m`: The function (equation) to generate the ck observations.
	 - `Component-3-Qul` folder includes the following files:
		- `compute_qul.m`: The function (equation) to compute the component-3 of hybrid-ebm.
	 - `Component-4-Sul` folder includes the following files:
		- `run_experiment_component_4_sul.m`: The script to run the component-4 of hybrid-ebm.
		- `compute_sul.m`: The function (equation) to compute the component-4 of hybrid-ebm.
  - `lib`: Contains libraries for analysis, machine learning, and utility functions.
- `models` folder: Contains the trained models, models predictions and model summaries for each component of hybrid-model.
- `reports` folder: Contains a brief reports of the analysis with graphics and figures.



## Contact [[to ToC]](#table-of-contents)

For any questions or inquiries, please contact [Leonardo Saccotelli](mailto:leonardo.saccotelli@cmcc.it) or [Rosalia Maglietta](mailto:rosalia.maglietta@cnr.it).

## License [[to ToC]](#table-of-contents)

This project is licensed under the [GPL 3.0 License](LICENSE).
