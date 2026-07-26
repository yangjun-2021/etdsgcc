import pandas as pd, numpy as np

# Project 1: etdsgcc
print('=== etdsgcc ===')
raw1 = pd.read_csv(r'D:\Project\etdsgcc\data\raw_data.csv')
print(f'Shape: {raw1.shape}')
print(f'Columns: {raw1.columns[:5].tolist()} ... {raw1.columns[-3:].tolist()}')
if 'FLAG' in raw1.columns:
    print(f'FLAG: {raw1["FLAG"].value_counts().to_dict()}')
else:
    print(f'Last col: {raw1.iloc[:,-1].value_counts().to_dict()}')
print(f'Missing ratio: {raw1.isnull().sum().sum() / raw1.size:.4f}')

# Project 2: ThiefElectricity
print('\n=== ThiefElectricity ===')
raw2 = pd.read_csv(r'D:\Project\ThiefElectricity\data\raw_data.csv')
print(f'Shape: {raw2.shape}')
print(f'Columns: {raw2.columns[:5].tolist()} ... {raw2.columns[-3:].tolist()}')
if 'FLAG' in raw2.columns:
    print(f'FLAG: {raw2["FLAG"].value_counts().to_dict()}')
else:
    print(f'Last col: {raw2.iloc[:,-1].value_counts().to_dict()}')
print(f'Missing ratio: {raw2.isnull().sum().sum() / raw2.size:.4f}')

# Check if same data
print(f'\nSame shape: {raw1.shape == raw2.shape}')
if raw1.shape == raw2.shape and 'FLAG' in raw1.columns and 'FLAG' in raw2.columns:
    print(f'Same FLAG: {(raw1["FLAG"].values == raw2["FLAG"].values).all()}')

# Check df.csv in etdsgcc
print('\n=== etdsgcc/df.csv ===')
df1 = pd.read_csv(r'D:\Project\etdsgcc\data\df.csv')
print(f'Shape: {df1.shape}')
print(f'Columns: {df1.columns[:10].tolist()}')
