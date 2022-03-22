# Import packages
# Data import and manipulation
import pandas as pd
import numpy as np
import cpi
# Data visualization
import seaborn as sbn; sbn.set()
import matplotlib.pyplot as plt
# Model packages
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold

# Assessing the question
"""
Determining the current market value of the King's Estate is a particularly difficult question to answer because there
aren't many comparable properties in Pittsburgh. It is particularly unique because of its historical designation, along with the fact that
the county value and fair market price are drastically greater than the sale price in 1994.
Some potential approaches:
- Look at comparable homes in other cities (Cleveland and Pittsburgh example from Randy)
- Clustering 
- Regression: Supervised ML is likely not the best way to go here since we don't have many labels of similar characteristics to train
the model on given Pittsburgh only has a few mansions of high quality like the King Estate

Thoughts on value:
- True value may be higher than market value (which takes into account demand)
- Use assessment data to predict true value 
- Assessed value (true value) base year is 2012
- Cheaper homes tend to be more 

Notes from data file:
- Deeds recorded since 1788 but some properties may default to 1950 as earliest date
- Number of  bedrooms has little impact on value
- CDU (condition): measure of condition of property and desirability/utility is accounted for a bit as well (less weight)

Likely predictors:
- PREVIOUSSALEPRICE, PREVIOUSSALEPRICE2, Square footage
"""

# # Read in data
data_original = pd.read_csv('assessments.csv')
data_original.describe()

# DATA CLEANING
# Limit data to privately owned property (not commercial) with CLASSDESC field
data = data_original[data_original['CLASSDESC'] != 'AGRICULTURAL']
data = data[data['CLASSDESC'] != 'INDUSTRIAL']
data = data[data['CLASSDESC'] != 'UTILITIES']
# Change data type of date variables
# Latest
data['Sale_Date'] = pd.to_datetime(data['SALEDATE'])
data['Sale_Date'] = pd.to_datetime(data['Sale_Date'], format='%Y%m%d')
data['Sale_Year'] = pd.DatetimeIndex(data['Sale_Date']).year
data['Sale_Month'] = pd.DatetimeIndex(data['Sale_Date']).month
data = pd.get_dummies(data, columns=['Sale_Month'])

# Previous
data['Prev_Sale_Date'] = pd.to_datetime(data['PREVSALEDATE'])
data['Prev_Sale_Date'] = pd.to_datetime(data['Prev_Sale_Date'], format='%Y%m%d')
data['Prev_Sale_Year'] = pd.DatetimeIndex(data['Prev_Sale_Date']).year
data['Prev_Sale_Month'] = pd.DatetimeIndex(data['Prev_Sale_Date']).month
# Previous 2
data['Prev_Sale_Date2'] = pd.to_datetime(data['PREVSALEDATE2'])
data['Prev_Sale_Date2'] = pd.to_datetime(data['Prev_Sale_Date2'], format='%Y%m%d')
data['Prev_Sale_Year2'] = pd.DatetimeIndex(data['Prev_Sale_Date2']).year
data['Prev_Sale_Month2'] = pd.DatetimeIndex(data['Prev_Sale_Date2']).month
# Drop observations before 1950 (irrelevant)
data = data[(data['Sale_Year'] > 1949)]
data = data[(data['FAIRMARKETBUILDING'] > 0)]
# Create years since previous sale
# data['Yrs_Since_PrevSale'] = (data['Sale_Year'] - data['Prev_Sale_Year']).astype('int32')
# Create baths variable
data['BATHS'] = data['FULLBATHS'] + data['HALFBATHS']
# Create Street variable
data['Street_Suffix'] = data['PROPERTYADDRESS'].str.split().str[-1]
# Dropping insignificant variables (high cardinality, duplicates)
data_v2 = data.drop(["CARDNUMBER", "PROPERTYADDRESS", "ASOFDATE", "MUNICODE", "PROPERTYUNIT", "PROPERTYHOUSENUM", "PROPERTYFRACTION",
                     "LEGAL1", "LEGAL2", "LEGAL3", "NEIGHDESC", "MUNIDESC", "TAXYEAR", "HEATINGCOOLING",
                     "CONDITIONDESC", "ROOFDESC", "ROOF", "EXTFINISH_DESC", "LOCALTOTAL", "LOCALLAND",
                     "LOCALBUILDING", "COUNTYEXEMPTBLDG", "TAXDESC", "TAXSUBCODE_DESC", "OWNERDESC", "USECODE",
                     "USEDESC", "FARMSTEADFLAG", "CLEANGREEN", "PROPERTYSTATE", "PARID",
                     "PROPERTYZIP", "SCHOOLDESC", "NEIGHCODE", "TAXSUBCODE", "RECORDDATE", "SALECODE",
                     "PREVSALEDATE","PREVSALEDATE2", "SALEDATE", "SALEDESC", "DEEDBOOK", "DEEDPAGE", "COUNTYBUILDING", "COUNTYLAND",
                     "Prev_Sale_Date", "Prev_Sale_Date2", "Prev_Sale_Month", "Prev_Sale_Month2", "ALT_ID",
                     "ABATEMENTFLAG", "CDU", "CDUDESC", "CLASS",
                     "CHANGENOTICEADDRESS1", "CHANGENOTICEADDRESS2", "CHANGENOTICEADDRESS3", "CHANGENOTICEADDRESS4"],
                     axis=1) #most of this will get filtered out by sq footage,
                    # lot area for commercial
data_v2 = data_v2[(data_v2['Street_Suffix'] != '885')]
data_v2 = data_v2[(data_v2['Street_Suffix'] != '8')]
data_v2 = data_v2[(data_v2['Street_Suffix'] != '30')]
data_v2 = data_v2[(data_v2['Street_Suffix'] != '908')]
data_v2 = data_v2[(data_v2['Street_Suffix'] != '910')]

# Difference between fairmarket (value) and sale price
data_v2['MARKUP_Sale-Fair'] = data_v2['SALEPRICE'] - data_v2['FAIRMARKETTOTAL'] # need to remove before running models
# data_v2['TREND_Fair-Prev'] = data_v2['FAIRMARKETTOTAL'] - data_v2['PREVSALEPRICE']/data_v2['Yrs_Since_PrevSale']

# Drop observations without STYLE
data_v3 = data_v2[data_v2['STYLE'].notnull()]

# Dummy indicating property in big city (Pittsburgh)
data_v3['In_City'] = 0
data_v3.loc[data_v3['PROPERTYCITY'] == 'PITTSBURGH', 'In_City'] = 1
data_v3 = data_v3.drop('PROPERTYCITY', axis=1)

# Dummy for corporation owner
data_v3['Owner_Corp'] = 0
data_v3.loc[data_v3['OWNERCODE'] > 19, 'Owner_Corp'] = 1

# Dummy for taxcode
data_v3['TAX'] = 0
data_v3.loc[data_v3['TAXCODE'] == 'T', 'TAX'] = 1
data_v3 = data_v3.drop('TAXCODE', axis=1)

# Make homestead flag numeric
data_v3['HOMESTEAD'] = [1 if x == 'HOM' else 0 for x in data_v3['HOMESTEADFLAG']]
data_v3 = data_v3.drop('HOMESTEADFLAG', axis=1)

# Grade scale
data_v3['GRADE_Score'] = 0
data_v3.loc[data_v3['GRADE'] == 'X+', 'GRADE_Score'] = 16
data_v3.loc[data_v3['GRADE'] == 'X-', 'GRADE_Score'] = 16
data_v3.loc[data_v3['GRADE'] == 'X', 'GRADE_Score'] = 16
data_v3.loc[data_v3['GRADE'] == 'XX', 'GRADE_Score'] = 16
data_v3.loc[data_v3['GRADE'] == 'XX-', 'GRADE_Score'] = 16
data_v3.loc[data_v3['GRADE'] == 'XX+', 'GRADE_Score'] = 16
data_v3.loc[data_v3['GRADE'] == 'A+', 'GRADE_Score'] = 15
data_v3.loc[data_v3['GRADE'] == 'A', 'GRADE_Score'] = 14
data_v3.loc[data_v3['GRADE'] == 'A-', 'GRADE_Score'] = 13
data_v3.loc[data_v3['GRADE'] == 'B+', 'GRADE_Score'] = 12
data_v3.loc[data_v3['GRADE'] == 'B', 'GRADE_Score'] = 11
data_v3.loc[data_v3['GRADE'] == 'B-', 'GRADE_Score'] = 10
data_v3.loc[data_v3['GRADE'] == 'C+', 'GRADE_Score'] = 9
data_v3.loc[data_v3['GRADE'] == 'C', 'GRADE_Score'] = 8
data_v3.loc[data_v3['GRADE'] == 'C-', 'GRADE_Score'] = 7
data_v3.loc[data_v3['GRADE'] == 'D+', 'GRADE_Score'] = 6
data_v3.loc[data_v3['GRADE'] == 'D', 'GRADE_Score'] = 5
data_v3.loc[data_v3['GRADE'] == 'D-', 'GRADE_Score'] = 4
data_v3.loc[data_v3['GRADE'] == 'E+', 'GRADE_Score'] = 3
data_v3.loc[data_v3['GRADE'] == 'E', 'GRADE_Score'] = 2
data_v3.loc[data_v3['GRADE'] == 'E-', 'GRADE_Score'] = 1
data_v3 = data_v3[(data_v3['GRADE_Score'] > 0)]
data_v3 = data_v3.drop('GRADE', axis=1)

# Condition numeric (score)
data_v3['Condition_Score'] = 0
data_v3.loc[data_v3['CONDITION'] == 8, 'Condition_Score'] = 1
data_v3.loc[data_v3['CONDITION'] == 7, 'Condition_Score'] = 7
data_v3.loc[data_v3['CONDITION'] == 1, 'Condition_Score'] = 8
data_v3.loc[data_v3['CONDITION'] == 2, 'Condition_Score'] = 6
data_v3.loc[data_v3['CONDITION'] == 3, 'Condition_Score'] = 5
data_v3.loc[data_v3['CONDITION'] == 4, 'Condition_Score'] = 4
data_v3.loc[data_v3['CONDITION'] == 5, 'Condition_Score'] = 3
data_v3.loc[data_v3['CONDITION'] == 6, 'Condition_Score'] = 2
data_v3 = data_v3[(data_v3['Condition_Score'] > 0)]
data_v3 = data_v3.drop('CONDITION', axis=1)

# Heat dummy
data_v3['Heat'] = 0
data_v3.loc[data_v3['HEATINGCOOLINGDESC'] == 'Central Heat', 'Heat'] = 1
data_v3.loc[data_v3['HEATINGCOOLINGDESC'] == 'Central Heat with AC', 'Heat'] = 1
data_v3.loc[data_v3['HEATINGCOOLINGDESC'] == 'Electric', 'Heat'] = 1
data_v3.loc[data_v3['HEATINGCOOLINGDESC'] == 'Electric Heat with AC', 'Heat'] = 1
data_v3.loc[data_v3['HEATINGCOOLINGDESC'] == 'Floor Furnace', 'Heat'] = 1
data_v3.loc[data_v3['HEATINGCOOLINGDESC'] == 'Floor Furnace with AC', 'Heat'] = 1
data_v3.loc[data_v3['HEATINGCOOLINGDESC'] == 'Heat Pump', 'Heat'] = 1
data_v3.loc[data_v3['HEATINGCOOLINGDESC'] == 'Heat Pump with AC', 'Heat'] = 1
data_v3.loc[data_v3['HEATINGCOOLINGDESC'] == 'Unit Heat', 'Heat'] = 1
data_v3.loc[data_v3['HEATINGCOOLINGDESC'] == 'Unit Heat with AC', 'Heat'] = 1
data_v3.loc[data_v3['HEATINGCOOLINGDESC'] == 'Wall Furnace', 'Heat'] = 1
data_v3.loc[data_v3['HEATINGCOOLINGDESC'] == 'Wall Furnace with AC', 'Heat'] = 1
data_v3.loc[data_v3['HEATINGCOOLINGDESC'] == 'Other', 'Heat'] = 1
data_v3.loc[data_v3['HEATINGCOOLINGDESC'] == 'No Heat but with AC', 'Heat'] = 0
data_v3.loc[data_v3['HEATINGCOOLINGDESC'] == 'None', 'Heat'] = 0

# AC dummy
data_v3['AC'] = 0
data_v3.loc[data_v3['HEATINGCOOLINGDESC'] == 'Central Heat', 'AC'] = 0
data_v3.loc[data_v3['HEATINGCOOLINGDESC'] == 'Central Heat with AC', 'AC'] = 1
data_v3.loc[data_v3['HEATINGCOOLINGDESC'] == 'Electric', 'AC'] = 0
data_v3.loc[data_v3['HEATINGCOOLINGDESC'] == 'Electric Heat with AC', 'AC'] = 1
data_v3.loc[data_v3['HEATINGCOOLINGDESC'] == 'Floor Furnace', 'AC'] = 0
data_v3.loc[data_v3['HEATINGCOOLINGDESC'] == 'Floor Furnace with AC', 'AC'] = 1
data_v3.loc[data_v3['HEATINGCOOLINGDESC'] == 'Heat Pump', 'AC'] = 0
data_v3.loc[data_v3['HEATINGCOOLINGDESC'] == 'Heat Pump with AC', 'AC'] = 1
data_v3.loc[data_v3['HEATINGCOOLINGDESC'] == 'Unit Heat', 'AC'] = 0
data_v3.loc[data_v3['HEATINGCOOLINGDESC'] == 'Unit Heat with AC', 'AC'] = 1
data_v3.loc[data_v3['HEATINGCOOLINGDESC'] == 'Wall Furnace', 'AC'] = 0
data_v3.loc[data_v3['HEATINGCOOLINGDESC'] == 'Wall Furnace with AC', 'AC'] = 1
data_v3.loc[data_v3['HEATINGCOOLINGDESC'] == 'Other', 'AC'] = 1
data_v3.loc[data_v3['HEATINGCOOLINGDESC'] == 'No Heat but with AC', 'AC'] = 1
data_v3.loc[data_v3['HEATINGCOOLINGDESC'] == 'None', 'AC'] = 0
data_v3 = data_v3.drop('HEATINGCOOLINGDESC', axis=1)

# CONDO DUMMY
data_v3['Condo'] = 0
data_v3.loc[data_v3['STYLE'] == 21, 'Condo'] = 1
data_v3.loc[data_v3['STYLE'] == 27, 'Condo'] = 1
data_v3.loc[data_v3['STYLE'] == 25, 'Condo'] = 1
data_v3.loc[data_v3['STYLE'] == 8, 'Condo'] = 1
data_v3.loc[data_v3['STYLE'] == 22, 'Condo'] = 1
data_v3.loc[data_v3['STYLE'] == 24, 'Condo'] = 1

# MULTI FAMILY
data_v3['Multi-Family'] = 0
data_v3.loc[data_v3['STYLE'] == 13, 'Multi-Family'] = 1

# TOWNHOUSE DUMMY
data_v3['TWNHouse'] = 0
data_v3.loc[data_v3['STYLE'] == 11, 'TWNHouse'] = 1
data_v3.loc[data_v3['STYLE'] == 9, 'TWNHouse'] = 1
data_v3.loc[data_v3['STYLE'] == 12, 'TWNHouse'] = 1

# BASEMENT DUMMY
data_v3 = data_v3[data_v3['BASEMENT'].notnull()]
data_v3['Basement'] = 1
data_v3.loc[data_v3['BASEMENT'] == 1, 'Basement'] = 0

# Additional filters
data_v4 = data_v3.drop(["STYLE", "STYLEDESC", "FAIRMARKETLAND", "FAIRMARKETBUILDING", "BASEMENT", "BASEMENTDESC",
                     "FULLBATHS", "HALFBATHS", "BSMTGARAGE", "GRADEDESC", "EXTERIORFINISH", "Prev_Sale_Year2"], axis=1)

# SCHOOLCODE
data_v5 = pd.get_dummies(data_v4, columns=['SCHOOLCODE'])
data_v5 = pd.get_dummies(data_v5, columns=['OWNERCODE'])
data_v5 = pd.get_dummies(data_v5, columns=['CLASSDESC'])

"""
CLEAN:
- Indicator var for investment property: Combine change notice address 1-3 into one, see if it matches the address of the purchased property
- Add investment property indicator variable (not helpful since we don't have the outcome of investors work?)
"""
# Identify private investment properties by looking at lack of Homestead flag, and difference (if any) between homeowner address (CHANGEOWNER var as proxy) and property address

# Save as csv
data_v5.to_csv('assessment_clean.csv', index=False)
data_v5 = pd.read_csv('assessment_clean.csv')
# # ANALYSIS
data_final = data_v5.drop(['MARKUP_Sale-Fair','Street_Suffix', 'COUNTYTOTAL'], axis=1)
data_final = pd.DataFrame(data_final)
# Filter to home prices > $50,000
data_final = data_final[(data_final['SALEPRICE'] > 15000)]
# Drop Kings Estate from df
Kings_Estate_Prediction = pd.DataFrame(data_final[(data_final['YEARBLT'] == 1880) & (data_final['LOTAREA'] == 80368)])
Kings_Estate_Prediction.to_csv('Kings_Estate.csv', index=False)
data_final.drop(data_final[(data_final['YEARBLT'] == 1880) & (data_final['LOTAREA'] == 80368)].index)
data_final.to_csv('assessment_final_p2.csv', index=False)
data_final = data_final.drop('Sale_Date', axis=1)
# Replacing NAs with median/feature given skewness
column_avgs = (data_final.median())
column_avgs = column_avgs.astype(float).round()
data_final = data_final.fillna(column_avgs)
# Save df
data_final.to_csv('assessment_final.csv', index=False)


# # # VISUALIZATIONS
# plt.scatter(data_final['LOTAREA'], data_final['SALEPRICE'], alpha=0.5)
# plt.show()

# data_features = data_final.columns.astype(str)
# for feature in data_features:
#     sbn.countplot(x=(feature), data=data_final)
#     plt.savefig("hist" + "_" + str(feature) + ".png")
#     plt.show()

"""METHOD ONE"""
# # APPROACH ONE (Decision Tree)
# Read in data
data_final = pd.read_csv('assessment_final.csv')
# divide into training/testing data
kf = KFold(n_splits=10, shuffle=False)
# Cross validation
i = 1
for train_index, test_index in kf.split(data_final):
    x_train = data_final.iloc[train_index].drop('SALEPRICE', axis=1)
    y_train = data_final.iloc[train_index].loc[:,'SALEPRICE']
    x_test = data_final.iloc[test_index].drop('SALEPRICE', axis=1)
    y_test = data_final.iloc[test_index].loc[:, 'SALEPRICE']

    # Train model
    tree = DecisionTreeRegressor(criterion = 'squared_error', random_state = 42, min_samples_leaf = 10, ccp_alpha=0.1)
    tree.fit(x_train, y_train)
    pred = tree.predict(x_test)

# Calculate accuracy
    print(f"SqRt MSE for fold no.{i} on the test set: {np.sqrt(metrics.mean_squared_error(y_test, pred))}")
    i += 1


# # LINEAR MODEL
# Baseline OLS model
linreg = LinearRegression()
model_OLS = linreg.fit(x_train, y_train)
y_pred_reg = model_OLS.predict(x_test)
# Performance metric (mean squared error)
print("OLS Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred_reg)))

# Cost complexity pruning
path = tree.cost_complexity_pruning_path(x_train, y_train)
ccp_alphas = np.linspace(0, 3, 25)
clfs = []
for alpha in ccp_alphas:
    clf = DecisionTreeRegressor(random_state = 42, min_samples_leaf = 10, ccp_alpha=alpha)
    clf.fit(x_train, y_train)
    clfs.append(clf)

train_acc = []
test_acc = []
for c in clfs:
    y_train_pred = c.predict(x_train)
    y_test_pred = c.predict(x_test)
    train_acc.append(mean_squared_error(y_train, y_train_pred))
    test_acc.append(mean_squared_error(y_test, y_test_pred))

# plot the changes in accuracy for each alpha
plt.scatter(ccp_alphas, train_acc)
plt.scatter(ccp_alphas, test_acc)
plt.plot(ccp_alphas, train_acc, label = 'train_mse', drawstyle = "steps-post")
plt.plot(ccp_alphas, test_acc, label = 'test_mse', drawstyle = "steps-post")
plt.legend()
plt.title('MSE vs alpha')
plt.show()

# get and summarize feature importance
importance = tree.feature_importances_
for i, v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.title("Feature importance using Decision Tree")
plt.savefig("Feature_Imp.png")
plt.show()

# Alpha <0.10 works best
# Tree Model
tree_2 = DecisionTreeRegressor(criterion = 'squared_error', random_state = 42, ccp_alpha=.01)
model2 = tree_2.fit(x_train, y_train)
y_pred_t2 = model2.predict(x_test)

# compute accuracy in the test data
print("Pruned Tree Mean Squared Error:", metrics.mean_squared_error(y_test, y_pred_t2))
print("Pruned Tree Sq Rt MSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred_t2)))

# Test on Kings Estate
test = pd.read_csv('test.csv')
Estate_Prediction_Regression = model_OLS.predict(test)
Estate_Prediction_Tree = model2.predict(test)
print(Estate_Prediction_Regression) # $984,345
print(Estate_Prediction_Tree) # $2,221,951


"""METHOD TWO"""
# # Import packages
# Data import and manipulation
import pandas as pd
import numpy as np
import cpi
# Data visualization
import seaborn as sbn; sbn.set()
import matplotlib.pyplot as plt
# Model packages
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, RidgeCV
# Time Series Regression
import warnings
import itertools
import numpy as np
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib
from pylab import rcParams

# # # Read in data
# data_original = pd.read_csv('assessment_final_p2.csv')
inflation = pd.read_csv('CSUSHPINSA.csv')
#
# # # Filter
# data = data_original[data_original['Condo'] == 0]
# data = data[data['Multi-Family'] == 0]
# data = data[data['TWNHouse'] == 0]
# data = data[data['In_City'] == 1]
# data = data[data['LOTAREA'] >= 50000]
# data = data[data['YEARBLT'] <= 1940]
# data = data[data['Sale_Year'] < 2022]
# # drop vars
# data = data.drop(['PREVSALEPRICE', 'PREVSALEPRICE2', 'Prev_Sale_Year', 'Multi-Family', 'TWNHouse', 'In_City',
#                   'Condo'], axis=1)
#
# data.to_csv('assessment_cleaned_p2.csv', index=False)

# # Clean inflation df
# create index multiplier (indexed t0 12/1/2021)
inflation['HPI_multiplier'] = inflation['AVG'].iloc[-1]/inflation['AVG']
# merge dataframe
data = pd.read_csv('assessment_cleaned_p2.csv')
data_v2 = pd.merge(data, inflation, how='left', on='Sale_Year')
data_v2['SALEPRICE_adj'] = data_v2['SALEPRICE']*data_v2['HPI_multiplier']
# clean up df
data_v3 = data_v2.drop(['SUM', 'AVG'], axis=1)

# # Analyze
data_v3['ROI'] = data_v3['FAIRMARKETTOTAL'] - data_v3['SALEPRICE_adj']
data_v3['ROI_Pct'] = data_v3['ROI']/data_v3['SALEPRICE_adj']
# save df
data_v3.to_csv('assessment_p2.csv', index=False)
# visualize
sbn.regplot(data_v3['Sale_Year'], data_v3['ROI'])
plt.savefig("Sales_Investment.png")
#plt.show()
sbn.regplot(data_v3['Sale_Year'], data_v3['ROI_Pct'])
plt.savefig("Sales_InvestmentPct.png")
# plt.show()

# Save final df
df = pd.DataFrame(data_v3.drop(['ROI', 'Sale_Year', 'Sale_Month_1.0', 'Sale_Month_2.0', 'Sale_Month_3.0',
                                'Sale_Month_4.0', 'Sale_Month_5.0', 'Sale_Month_6.0', 'Sale_Month_7.0',
                                'Sale_Month_8.0', 'Sale_Month_9.0', 'Sale_Month_10.0', 'Sale_Month_11.0',
                                'Sale_Month_12.0'], axis=1))
df.dropna()

# # TIME SERIES ANALYSIS
df['Sale_Date'].min(), df['Sale_Date'].max()
# label the data frame columns
df_features_1 = df.drop(['ROI_Pct'], axis=1)
data_features1 = df_features_1.astype(str)
df_features_2 = df.drop(['ROI_Pct', 'Sale_Date'], axis=1)
data_features2 = df_features_2.astype(str)
# drop the outcome from the frame that includes the other predictors
df_outcomes = df.drop(data_features2, axis=1)
# adding column name to the respective columns
df_outcomes.columns =['Sale_Date', 'ROI_Pct']
# sort everything by time
df_outcomes = df_outcomes.sort_values('Sale_Date')
df_outcomes = df_outcomes.set_index('Sale_Date')
df_outcomes.index
y = df_outcomes['Sale_Date'].resample('MS').mean()
# perform the augmented Dickey Fuller test
result = adfuller(df_outcomes)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
# generate a time series plot until the end of 2017
y['2021':]
y.plot(figsize=(15, 6))
plt.show()
# # decompose this time series into its trend, seasonal, and residual components
# rcParams['figure.figsize'] = 18, 8
# decomposition = sm.tsa.seasonal_decompose(y, model='additive')
# fig = decomposition.plot()
# plt.show()
# # estimate the p, d, and q terms for our ARIMA via grid search
# p = d = q = range(0, 2)
# pdq = list(itertools.product(p, d, q))
# seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
# print('Examples of parameter combinations for Seasonal ARIMA...')
# print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
# print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
# print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
# print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
# # find the best seasonal ARIMA model given the parameters from the prev step
# for param in pdq:
#     for param_seasonal in seasonal_pdq:
#         try:
#             mod = sm.tsa.statespace.SARIMAX(y,
#                                             order=param,
#                                             seasonal_order=param_seasonal,
#                                             enforce_stationarity=False,
#                                             enforce_invertibility=False)
#             results = mod.fit()
#             print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal,
# results.aic))
#         except:
#             continue
# # fit the selected model and print the results table
# mod = sm.tsa.statespace.SARIMAX(y,
#                                 order=(1, 1, 0),
#                                 seasonal_order=(1, 1, 0, 12))
# results = mod.fit()
# print(results.summary().tables[1])
# # plot the forecast with its confidence interval and the actual value (must run
# lines 99 to 110 together)
# pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)
# pred_ci = pred.conf_int()
# ax = y['2014':].plot(label='observed')
# pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7,
# figsize=(14, 7))
# ax.fill_between(pred_ci.index,
#                 pred_ci.iloc[:, 0],
#                 pred_ci.iloc[:, 1], color='k', alpha=.2)
# ax.set_xlabel('Date')
# ax.set_ylabel('Office Supply Sales')
# plt.legend()
# plt.show()
# # compute the forecast error
# y_forecasted = pred.predicted_mean
# y_truth = y['2017-01-01':]
#
