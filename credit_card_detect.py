import pandas as pd
import warnings
import joblib
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, roc_curve, classification_report, precision_score, recall_score, accuracy_score, f1_score, roc_auc_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from azureml.core import Run
run = Run.get_context()
warnings.filterwarnings("ignore")

#load the csv dataset into the pandas dataframe(data)
data = pd.read_csv('./creditcard.csv')



# Separate majority and minority class instances
majority_class = data[data['Class'] == 0]  # Valid transactions
minority_class = data[data['Class'] == 1]  # Fraud transactions

# Undersample the majority class
undersampled_majority_class = resample(majority_class,
                                       replace=False,
                                       n_samples=len(minority_class),
                                       random_state=42)

# Combine the minority class DataFrame with the undersampled majority class DataFrame
undersampled_data = pd.concat([undersampled_majority_class, minority_class])

# Shuffle the undersampled DataFrame
data = undersampled_data.sample(frac=1, random_state=42)

# standardization
sc = StandardScaler()
data = data.drop(['Amount', 'Time'], axis=1).assign(scaled_amount=sc.fit_transform(data['Amount'].values.reshape(-1, 1)),scaled_time=sc.fit_transform(data['Time'].values.reshape(-1, 1)))

features = list(data.columns[:28])
def get_thr_min_max(df, var):
    
    p25 = df[var].quantile(0.25)
    p75 = df[var].quantile(0.75)
    iqr = p75-p25

    thr_min = p25-1.5*iqr
    thr_max = p75+1.5*iqr
    
    return thr_min, thr_max

def outlier_treatment_iqr(val,thr_min,thr_max):
    
    if val>thr_max:
        return thr_max

    elif val<thr_min:
        return thr_min
    
    else:
        return val
for i in features:
    thr_min, thr_max = get_thr_min_max(df=data, var= i)
    data[i] = data[i].apply(lambda x: outlier_treatment_iqr(x, thr_min, thr_max))
    

X= data.drop(['Class'], axis=1)
y= data['Class']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=100)

param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

#model train

rfc=RandomForestClassifier( random_state = 100 ) 

grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, scoring='f1', cv=5)
grid_search.fit(X_train, y_train)

# Get the best model with the optimal hyperparameters
best_model = grid_search.best_estimator_

Y_pred = best_model.predict(X_test)

run.log("Parameter",grid_search.get_params())

run.log("Accuracy",accuracy_score(y_test, Y_pred))
run.log("Precison",precision_score(y_test, Y_pred))
run.log("Recall",recall_score(y_test, Y_pred))
run.log("F1 Score",f1_score(y_test, Y_pred))

joblib.dump(grid_search, "credit_card.pkl")
