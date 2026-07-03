

import numpy as np 
import pandas as pd  
import matplotlib.pyplot as plot  
import seaborn as sus  
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split  



raw_data = pd.read_csv(r"C:\Users\Diksha\OneDrive\Documents\train.csv")

data = pd.DataFrame(raw_data)

print("\n" + "="*60)
print("STEP 1: CHECKING FOR MISSING VALUES")
print("="*60)

data = data[data['GrLivArea'] < 4000]
data = data[data['SalePrice'] < 700000]

# Log transform skewed features
data['SalePrice'] = np.log1p(data['SalePrice'])
data['LotArea'] = np.log1p(data['LotArea'])
data['GrLivArea'] = np.log1p(data['GrLivArea'])

print("\nMissing value count per column:")
print(data.isnull().sum())


data_null_clm = data.isnull().sum() / data.shape[0] * 100  

print("\nMissing value percentage per column:")
print(data_null_clm)

# Find columns where more than 20% of data is missing
null20_clm_list = data_null_clm[data_null_clm > 20].index

# Remove columns with more than 20% missing data (too much data lost)
data_drop_clm = data.drop(columns=null20_clm_list)

print(f"\nRemoved {len(null20_clm_list)} columns with >20% missing values")
print("Remaining missing values:")
print(data_drop_clm.isnull().sum())



print("\nData shapes:")
print(f"Original data shape: {data.shape}") 
print(f"After dropping columns with >20% missing: {data_drop_clm.shape}")

data_object = data_drop_clm.select_dtypes(include=['object'])
print(f"Categorical (text) columns shape: {data_object.shape}")

data_numaric = data_drop_clm.select_dtypes(include=['int', 'float'])
print(f"Numerical columns shape: {data_numaric.shape}")



print("\n" + "="*60)
print("STEP 2: FILLING MISSING VALUES")
print("="*60)

print("\nFilling numerical columns with MEDIAN:")
numeic_clo = data_numaric.isnull().sum()
print(data_numaric.isnull().sum())

for i in numeic_clo.index:
    data_numaric[i].fillna(data_numaric[i].median(), inplace=True)

print("All missing numerical values filled with median ")
print(data_numaric.isnull().sum())



print("\nFilling categorical columns with MODE (most common value):")
string_clo = data_object.isnull().sum()

for i in string_clo.index:
    data_object[i].fillna(data_object[i].mode()[0], inplace=True)

print(f"All missing categorical values filled")
print(f"Total remaining missing values: {data_object.isnull().sum().sum()}")

print(data_object.head())


print("\n" + "="*60)
print("STEP 5: ENCODING CATEGORICAL COLUMNS")
print("="*60)
data_encoded = data_object.copy()



print("\n" + "="*60)
print("ENCODING TYPE 1: ORDINAL ENCODING (Quality/Condition Levels)")
print("="*60)

ordinal_mappings = {
    'ExterQual': {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},  
    'ExterCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},  
    'BsmtQual': {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},  
    'BsmtCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4},  
    'BsmtExposure': {'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},  
    'BsmtFinType1': {'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},  
    'BsmtFinType2': {'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},  
    'HeatingQC': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}, 
    'KitchenQual': {'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4},  
    'Functional': {'Sev': 1, 'Maj2': 2, 'Maj1': 3, 'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7},  
    'GarageFinish': {'Unf': 1, 'RFn': 2, 'Fin': 3}, 
    'GarageQual': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},  
    'GarageCond': {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},  
    'LandSlope': {'Sev': 1, 'Mod': 2, 'Gtl': 3},  
    'PavedDrive': {'N': 1, 'P': 2, 'Y': 3}  
}

print("\n Applying Ordinal Encoding:")
for col, mapping in ordinal_mappings.items():
    # Check if column exists in our data
    if col in data_encoded.columns:
        # Use .map() to replace values: 'Fa'→1, 'TA'→2, etc.
        data_encoded[col] = data_encoded[col].map(mapping)
        print(f"   {col} → {mapping}")


print("\n" + "="*60)
print("ENCODING TYPE 2: ONE-HOT ENCODING (Nominal Categories)")
print("="*60)

onehot_columns = [
    'MSZoning','Street','LotShape','LotConfig','LandContour','Utilities','BldgType',      
    'HouseStyle','RoofStyle','RoofMatl','Foundation','Heating','CentralAir','Electrical',   
    'GarageType','Condition1','Condition2','SaleType','SaleCondition'  
]

print(f"\n Applying One-Hot Encoding to {len(onehot_columns)} columns:")
print(f"  Columns: {onehot_columns}")

# pd.get_dummies() creates binary columns for each unique value
# drop_first=True removes the first category to avoid multicollinearity
data_encoded = pd.get_dummies(data_encoded, columns=onehot_columns, drop_first=True, dtype='int')

print(f"  New shape after one-hot encoding: {data_encoded.shape}")
print(f"  (Original shape: {data_drop_clm.shape})")



print("\n" + "="*60)
print("ENCODING TYPE 3: TARGET ENCODING (High Cardinality)")
print("="*60)

if 'Neighborhood' in data_encoded.columns and 'SalePrice' in data_encoded.columns:
    neighborhood_encoding = data_encoded.groupby('Neighborhood')['SalePrice'].mean()
    data_encoded['Neighborhood'] = data_encoded['Neighborhood'].map(neighborhood_encoding)
    print(f"   Neighborhood → Target Encoding (Average Price per neighborhood)")
    print(f"    Example mapping: {dict(list(neighborhood_encoding.to_dict().items())[:3])}")  # Show first 3 examples
else:
    
    neighborhood_counts = data_encoded['Neighborhood'].value_counts()
    data_encoded['Neighborhood'] = data_encoded['Neighborhood'].map(neighborhood_counts)
    print(f"   Neighborhood → Frequency Encoding")
    print(f"    (SalePrice not found, using neighborhood frequency)")


print("\n" + "="*60)
print("ENCODING TYPE 4: ONE-HOT WITH RARE GROUPING")
print("="*60)

for col in ['Exterior1st', 'Exterior2nd']:
    if col in data_encoded.columns:
        value_counts = data_encoded[col].value_counts()
        rare_threshold = len(data_encoded) * 0.02  # 2% threshold
        rare_values = value_counts[value_counts < rare_threshold].index  # Categories below threshold
        
        data_encoded[col] = data_encoded[col].replace(rare_values, 'Other')
        print(f"   {col} → Grouped {len(rare_values)} rare categories as 'Other'")
        
        data_encoded = pd.get_dummies(data_encoded, columns=[col], prefix=col, drop_first=True, dtype='int')
        print(f"    Created binary columns for each exterior type")


final_data = pd.concat([data_numaric, data_encoded], axis=1)
final_data = final_data.loc[:, ~final_data.columns.duplicated()]

print(final_data.shape)

X = final_data.drop(columns=['SalePrice'])
y = final_data['SalePrice']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train_target, y_test_target = train_test_split(
    X, y, test_size=0.2, random_state=42)

arr1 = x_train
arr2 = y_test_target

arr3 = x_test
arr4 = y_train_target

print(x_train.columns)
# stad_scaler.fit(data[['total_bill']])
# print(stad_scaler.mean_)
# print(stad_scaler.scale_)
# print(data['total_bill'].describe())
# arr=stad_scaler.transform(data[['total_bill']]).round(1)
# arr2=pd.DataFrame(arr,columns=['total_bill'])
# print(arr2['total_bill'].describe)


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Fit only on training data
scaled_data_final_X_tarin_arr = scaler.fit_transform(x_train)
scaled_data_final_X_tarin = pd.DataFrame(scaled_data_final_X_tarin_arr, columns=x_train.columns)

# Transform test data using the same scaler
scaled_data_final_X_test_arr = scaler.transform(x_test)
scaled_data_final_X_test = pd.DataFrame(scaled_data_final_X_test_arr, columns=x_test.columns)

print(scaled_data_final_X_tarin.shape)
print(scaled_data_final_X_test.shape)
print(scaled_data_final_X_tarin.describe())
print(scaled_data_final_X_test.describe())




from sklearn.linear_model import LinearRegression 
lr=LinearRegression()
lr.fit(scaled_data_final_X_tarin,y_train_target)

Y=lr.predict(scaled_data_final_X_test)
print(Y)

# Example: plot GrLivArea vs SalePrice
fig, axs = plot.subplots(3, 1, figsize=(10, 15))

# Scatter plot
axs[0].scatter(scaled_data_final_X_tarin['GrLivArea'], y_train_target,
               marker='.', color='blue')
axs[0].set_title("Scatter plot: GrLivArea vs SalePrice")
axs[0].set_xlabel("GrLivArea (scaled)")
axs[0].set_ylabel("SalePrice (log)")

# Actual vs Predicted (unsorted)
axs[1].plot(y_test_target.values, label="Actual", color="blue")
axs[1].plot(Y, label="Predicted", color="red", linestyle="--")
axs[1].legend()
axs[1].set_title("Actual vs Predicted SalePrice")
axs[1].set_xlabel("Samples")
axs[1].set_ylabel("SalePrice (log)")

# Sorted Actual vs Predicted
sorted_index = np.argsort(y_test_target.values)
y_test_sorted = y_test_target.values[sorted_index]
Y_sorted = Y[sorted_index]

axs[2].plot(y_test_sorted, label="Actual (sorted)", color="blue")
axs[2].plot(Y_sorted, label="Predicted (sorted)", color="red", linestyle="--")
axs[2].legend()
axs[2].set_title("Sorted Actual vs Predicted SalePrice")
axs[2].set_xlabel("Samples ")
axs[2].set_ylabel("SalePrice ")

plot.tight_layout()
plot.show()


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_test_target, Y)
mae = mean_absolute_error(y_test_target, Y)
r2 = r2_score(y_test_target, Y)

print("MSE:", mse)
print("MAE:", mae)
print("R²:", r2)
print("rmse:",np.sqrt(mse))
print("Model Score (R²):", lr.score(scaled_data_final_X_test, y_test_target))
