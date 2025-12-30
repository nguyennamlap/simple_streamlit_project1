import pandas as pd
import numpy as np
from src.handle.fill_null import fill_median, fill_mode, check_name_type_suite
def main():

    df = pd.read_csv('data/application_train.csv')
    
    # Fill null values in 'AMT_INCOME_TOTAL' with median
    df = fill_median(df, 'AMT_INCOME_TOTAL')
    
    # Create new column 'Tỉ lệ vay so với nhu cầu'
    df['Tỉ lệ vay so với nhu cầu'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    
    # Handle null values in 'OWN_CAR_AGE' and create 'Sở hữu xe' column
    df['Sở hữu xe'] = df['OWN_CAR_AGE'].apply(
        lambda x: 'Không có xe' if pd.isna(x) else 'Có xe'
    )
    
    # Fill null values in 'NAME_TYPE_SUITE' with 'unknown'
    df['NAME_TYPE_SUITE'] = df['NAME_TYPE_SUITE'].fillna('unknown')
    
    # Create OCCUPATION_TYPE_ENHANCED column
    df['OCCUPATION_TYPE_ENHANCED'] = df.apply(
        lambda row: f"NO_OCC_{row['NAME_INCOME_TYPE'].upper()}" 
        if pd.isna(row['OCCUPATION_TYPE']) 
        else row['OCCUPATION_TYPE'],
        axis=1
    )
    
    # Create OCCUPATION_MISSING_TYPE column
    df['OCCUPATION_MISSING_TYPE'] = df.apply(
        lambda row: row['NAME_INCOME_TYPE'] 
        if pd.isna(row['OCCUPATION_TYPE']) 
        else 'HAS_OCCUPATION',
        axis=1
    )
    
    # Create IS_RETIRED_NO_OCCUPATION column
    df['IS_RETIRED_NO_OCCUPATION'] = (
        (df['OCCUPATION_TYPE'].isna()) & 
        (df['NAME_INCOME_TYPE'] == 'Pensioner')
    ).astype(int)
    
    # Create IS_WORKING_NO_OCCUPATION column
    df['IS_WORKING_NO_OCCUPATION'] = (
        (df['OCCUPATION_TYPE'].isna()) & 
        (df['NAME_INCOME_TYPE'] == 'Working')
    ).astype(int)
    
    # Create missing indicators for EXT_SOURCE columns
    for col_name in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
        df[f'{col_name}_is_missing'] = df[col_name].isna().astype(int)
    
    # Impute EXT_SOURCE columns with median
    for col_name in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
        df = fill_median(df, col_name)
    
    # Handle MODE columns
    num_mode, str_mode = check_name_type_suite(df, "MODE")
    
    for col in num_mode:
        df = fill_mode(df, col)
    
    for col in str_mode:
        df[col] = df[col].fillna('UNKNOWN')
    
    # Handle AVG and MEDI columns
    for col in df.columns:
        if "AVG" in col.upper() or "MEDI" in col.upper():
            df = fill_median(df, col)
        
        # Handle AMT_REQ_CREDIT_BUREAU columns
        if "AMT_REQ_CREDIT_BUREAU" in col:
            df[col] = df[col].fillna(0)
    
    print(f"Processing complete. Shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    df.to_csv('data/df_processed.csv', index=False)
if __name__ == "__main__":
    main()