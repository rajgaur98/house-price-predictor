import numpy as np
import scipy.optimize as op
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import skew

train = pd.read_csv('C:\\Users\\Rajkumar\\Desktop\\studies\\Kaggle\\housing_prices\\train.csv')
test = pd.read_csv('C:\\Users\\Rajkumar\\Desktop\\studies\\Kaggle\\housing_prices\\test.csv')
train.drop(['Id'], axis =1 ,inplace = True)
test.drop(['Id'], axis =1 ,inplace = True)
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<300000)].index)
fig, ax = plt.subplots()
ax.scatter(train['GrLivArea'], train['SalePrice'])
#plt.show()

train['SalePrice'] = np.log1p(train['SalePrice'])
sns.distplot(train['SalePrice'], fit = norm)
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot = plt)
#plt.show()

ntrain = train.shape[0]
ntest = test.shape[0]
y_train = np.array(train['SalePrice'])
all_data = pd.concat((train, test)).reset_index(drop = True)
all_data.drop(['SalePrice'], axis = 1, inplace = True)
print(all_data.shape)

all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na==0].index).sort_values(ascending = False)[:30]
#print(all_data_na)
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)

corrmat = train.corr()
#plt.subplots(figsize=(12,9))
#sns.heatmap(corrmat, vmax=0.9, square=True)

all_data["PoolQC"] = all_data["PoolQC"].fillna("None")
all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")
all_data["Alley"] = all_data["Alley"].fillna("None")
all_data["Fence"] = all_data["Fence"].fillna("None")
all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):
    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    all_data[col] = all_data[col].fillna('None')
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")
all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data = all_data.drop(['Utilities'], axis=1)
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")


numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending = False)
skewness = pd.DataFrame({'skew': skewed_feats})

skewness = skewness[abs(skewness) > 0.75]
from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    #all_data[feat] += 1
    all_data[feat] = boxcox1p(all_data[feat], lam)
    
all_data[all_data.select_dtypes(include='object').columns.values] = all_data[all_data.select_dtypes(include='object').columns.values].astype('category')
all_data = pd.get_dummies(all_data)

train = np.array(all_data[:ntrain])
test = np.array(all_data[ntest:])
m, n = np.shape(test)

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(train, y_train)
y_test = rf.predict(test)
y_test = y_test.reshape(m, )
print(rf.score(train, y_train))
'''
Id = [i for i in range(1461,2920)]
pred = {'Id': Id,
		'SalePrice': y_test}
df2 = pd.DataFrame(pred, columns = ['Id', 'SalePrice'])
export_csv = df2.to_csv(r'output.csv', index = None, header = True)
'''
