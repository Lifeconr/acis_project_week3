# ACIS Insurance Analytics Project

## Overview
This repository contains the code and data for the AlphaCare Insurance Solutions (ACIS) project, aimed at analyzing insurance data to optimize marketing strategies and identify risk/profitability patterns. The project spans multiple tasks, leveraging Python, Git, and Data Version Control (DVC) for reproducibility and auditing.

## Project Structure
- `src/`: Contains Python scripts (`data_cleaning.py`, `eda_analysis.py`) for data processing and analysis.
- `notebooks/`: Includes Jupyter notebooks (`eda.ipynb`) for exploratory data analysis (EDA).
- `data/`: Stores raw (`MachineLearningRating_v3.txt`) and cleaned (`cleaned_data.csv`) datasets.
- `.dvc/`: DVC configuration and cache files for data versioning.

## Task 1: Exploratory Data Analysis (EDA)
### Objective
Develop a foundational understanding of the insurance dataset, assess its quality, and uncover initial risk and profitability patterns.

### Steps
1. **Data Cleaning**: Loaded and cleaned `MachineLearningRating_v3.txt` using `data_cleaning.py`, handling pipe-delimited data and imputing missing values.
2. **EDA Implementation**: Used `eda_analysis.py` and `eda.ipynb` to perform profiling, statistical tests (e.g., ANOVA), and visualizations (e.g., histograms, bar plots, heatmaps).
3. **Results**: Generated insights on loss ratios, vehicle claims (e.g., Toyota at ~52.3M, Ford at 0.0), and temporal trends.

### Files
- `src/data_cleaning.py`: Data loading and cleaning logic.
- `src/eda_analysis.py`: EDA functions (e.g., `profile_data`, `plot_eda_visualizations`).
- `notebooks/eda.ipynb`: Interactive EDA notebook.
- `data/cleaned_data.csv`: Cleaned dataset.

## Task 2: Data Version Control (DVC)
### Objective
Establish a reproducible and auditable data pipeline using DVC for regulatory compliance in finance/insurance.

### Steps
1. **Install DVC**: Installed via `pip install dvc`.
2. **Initialize DVC**: Ran `dvc init` in the project directory.
3. **Set Up Local Remote**: Created `C:\Users\Senayit\Documents\1\week3\dvc_storage` and configured it with `dvc remote add -d localstorage`.
4. **Track Data**: Added `data/cleaned_data.csv` with `dvc add`.
5. **Commit and Push**: Committed `.dvc` files to Git and pushed data to the local remote with `dvc push`.

### Files
- `.dvc/`: DVC configuration.
- `data/cleaned_data.csv.dvc`: Metadata for tracked data.
- `.dvc/config`: Remote storage configuration.

## How to Run
1. **Setup**:
   - Install dependencies: `pip install notebook pandas numpy matplotlib seaborn scipy statsmodels dvc`.
   - Clone the repository: `git clone <repository-url>`.
2. **Task 1**:
   - Navigate to `notebooks/` and run `eda.ipynb` in Jupyter Notebook to reproduce EDA.
   - Check `plots/` for visualizations.
3. **Task 2**:
   - Ensure DVC is installed and initialized (`dvc init`).
   - Pull data: `dvc pull` to restore `data/cleaned_data.csv` from the local remote.

