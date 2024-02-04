# CMCC-ML-Estuary-Salinity-Estimation
 
## Table of Contents:
- [Description](#description)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Setup](#setup)
- [Usage](#usage)
- [Contact](#contact)
- [License](#license)

## Description [[to ToC]](#table-of-contents)

The repository collects all the machine learning and deep learning experiments runned in the context of CMCC-EBM-Model improvements. The aim is to develop a model to predict salinity concentration at estuary mouth. The estuary test case was Po-di-Goro.
Two types of experiments have been runned. The first set of experiments employed different ML models and performance are compared with CMCC-EBM and CMCC-Hybrid-EBM.
The second set of experiments test different LSTM input configuration to perform a 7-steps ahead forecasting. 

## Project Structure [[to ToC]](#table-of-contents)

The project structure is organized as follows:

- `data` folder contains two subfolders:
  - `raw` folder contains the raw data:
  - `processed` folder contains the processed data used for training models.
- `src` folder contains the source code files and subfolders:
  - `models` folder contains two scipts, one for the statistical analysis and one to run the ML experiments.
  - `lib` folder contains libraries for analysis, machine learning, and utility functions.
- `notebook` folder contains the jupyter notebook to run the LSTM models.
- `models` folder contains the trained models, models predictions and model summaries.
- `reports` folder contains a brief reports of the analysis with graphics and figures.

## Requirements [[to ToC]](#table-of-contents)
- Matlab
  - MATLAB Version 9.14 (R2023a) (https://it.mathworks.com/products/matlab.html)
  - Statistics and Machine Learning Toolbox Version 12.5 (R2023a) (https://it.mathworks.com/products/statistics.html)
  - Parallel Computing Toolbox Version 7.8 (R2023a) (https://it.mathworks.com/products/parallel-computing.html)
- Jupyter Notebook
  - Pandas
  - Tensorflow
  - Keras
  - Kerastuner

## Setup [[to ToC]](#table-of-contents)
To set up the project, follow these steps:

1. Clone the repository: 
    ```
	https://github.com/CMCC-Foundation/CMCC-ML-Estuary-Salinity-Estimation.git
    ```
2. Navigate to the project directory:
    ```
    cd CMCC-ML-Estuary-Salinity-Estimation
    ```

## Usage [[to ToC]](#table-of-contents)
To run the experiment follow these steps:

1. Run the script to convert the raw data in processed data:
````
 \src\features\build_features.m
````
2. Run the script to run the statistical analysis:
````
\src\models\preliminars_analysis.m
````

3. Run the script to train machine learning models:
````
\src\models\train_model.m
````

4. To run the experiments releated to LSTM models run the notebook:
````
\notebook\
````

## Contact [[to ToC]](#table-of-contents)

For any questions or inquiries, please contact [Leonardo Saccotelli](mailto:leonardo.saccotelli@cmcc.it), [Rosalia Maglietta](mailto:rosalia.maglietta@cnr.it) or [Giorgia Verri](mailto:giorgia.verri@cmcc.it).

## License [[to ToC]](#table-of-contents)

This project is licensed under the [Apache License 2.0](LICENSE).
