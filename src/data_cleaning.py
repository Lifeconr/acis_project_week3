import pandas as pd
from pathlib import Path
import os

# Ensure plots directory exists
os.makedirs('plots', exist_ok=True)

def load_and_clean_data(file_path: str, delimiter: str = '|') -> pd.DataFrame:
    """
    Load insurance dataset from a text file and perform initial cleaning.

    Args:
        file_path (str): Path to the text file.
        delimiter (str): Delimiter for the text file (default: pipe '|').

    Returns:
        pd.DataFrame: Cleaned dataset.

    Raises:
        FileNotFoundError: If the file_path does not exist.
        pd.errors.EmptyDataError: If the text file is empty.
        pd.errors.ParserError: If the file cannot be parsed with the given delimiter.
        ValueError: If the loaded DataFrame is empty or has no columns.
    """
    try:
        if not Path(file_path).is_file():
            raise FileNotFoundError(f"Data file not found at: {file_path}")
        
        # Read the file
        df = pd.read_csv(file_path, sep=delimiter, low_memory=False)
        
        # Validate DataFrame
        if df.empty or df.columns.size == 0:
            with open(file_path, 'r', encoding='utf-8') as f:
                print(f"Debug: First 5 lines of {file_path}:")
                for i, line in enumerate(f.readlines()[:5], 1):
                    print(f"Line {i}: {line.strip()}")
            raise ValueError(f"Loaded DataFrame is empty or has no columns. Check delimiter or file format.")
        
        # Convert TransactionMonth to datetime
        if 'TransactionMonth' in df.columns:
            df['TransactionMonth'] = pd.to_datetime(
                df['TransactionMonth'], errors='coerce'
            )
        
        # Convert categorical columns to category type
        categorical_cols = ['Province', 'Gender', 'VehicleType', 'CoverType', 'make', 'Model']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        # Handle missing values
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_cols:
            df[col] = df[col].fillna(df[col].median())
        
        categorical_cols = df.select_dtypes(include=['category']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        # Handle other columns (e.g., object types like Citizenship)
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            df[col] = df[col].fillna('Unknown')
        
        return df
    
    except pd.errors.EmptyDataError:
        raise pd.errors.EmptyDataError(f"Text file at {file_path} is empty")
    except pd.errors.ParserError:
        with open(file_path, 'r', encoding='utf-8') as f:
            print(f"Debug: First 5 lines of {file_path}:")
            for i, line in enumerate(f.readlines()[:5], 1):
                print(f"Line {i}: {line.strip()}")
        raise pd.errors.ParserError(f"Failed to parse {file_path} with delimiter '{delimiter}'.")
    except Exception as e:
        raise Exception(f"Error loading data: {str(e)}")