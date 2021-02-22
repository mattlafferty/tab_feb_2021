import kaggle
import pandas as pd
import os
import lightgbm as lgbm
import optuna as opt
import seaborn as sns
import numpy as np
import gc
import shap
from scipy.stats import binom_test
from sklearn.model_selection import train_test_split


os.chdir("/Users/matthewlafferty/Dropbox/Kaggle/Tabular_Feb_2021/tabular-playground-series-feb-2021")
data = pd.read_csv('train.csv')


#!##########################################################################################################
#!##########################################################################################################
#! Functions
#!##########################################################################################################

def categorical_convert(X:pd.Series) -> pd.Series:
    temp = X.astype('category').cat.codes
    temp += 1
    return temp

def create_shadow_feature(X:pd.Series) -> np.array:
    temp = X.to_numpy(copy=True)
    np.random.shuffle(temp)
    return temp




#!##########################################################################################################
#!##########################################################################################################
#! Fixed parameters
#!##########################################################################################################
NUM_ITER = 3
TARGET = 'target'
NUM_BOOST_ROUNDS = 10000
NUM_FOLDS = 5

features = [x for x in list(data) if ('cat' in x) | ('cont' in x)]
cat_features = [x for x in list(data) if ('cat' in x)]
starting_params = {'boosting_type': 'gbdt',
                    'learning_rate': 0.25,
                    'metric':'rmse',
                    'objective':'regression'}




#!##########################################################################################################
#!##########################################################################################################
#! Converting categorical features
#!##########################################################################################################
for f in cat_features:
    data[f] = categorical_convert(data[f])




#!##########################################################################################################
#!##########################################################################################################
#! Feature Selection
#!##########################################################################################################

shadow_features = features + [f + '_shadow' for f in features]
shadow_cat_features = cat_features + [x + '_shadow' for x in cat_features]
shap_values_results_df = pd.DataFrame(columns = shadow_features + [TARGET])

data_shadow = data.copy()
    
for i in range(NUM_ITER):
    # We will determine optimal parameters on a random sample. We'll use these parameters to construct a new model for each iteration.
    # Determining the hyperparameters takes time, so we'll do it once, and use these parameters to construct our models.
    for f in features:
        data_shadow[f + '_shadow'] = create_shadow_feature(data_shadow[f])
        
    data_train, data_temp = train_test_split(data_shadow, train_size=0.6, shuffle=True)
    
    dtrain = lgbm.Dataset(data=data_train[shadow_features],
                            label=data_train[TARGET],
                            feature_name=shadow_features,
                            categorical_feature=shadow_cat_features)
    
    model = opt.integration.lightgbm.LightGBMTunerCV(params = starting_params,
                                                        train_set = dtrain,
                                                        num_boost_round = NUM_BOOST_ROUNDS,
                                                        nfold = NUM_FOLDS,
                                                        stratified=False,
                                                        shuffle = True,
                                                        feature_name=shadow_features,
                                                        categorical_feature=shadow_cat_features,
                                                        early_stopping_rounds=0.05*NUM_BOOST_ROUNDS,
                                                        verbose_eval = 100,
                                                        seed = 51)
    model.run()
    print(model.best_score)
    shadow_best_params =  model.best_params
    
    data_val, data_test = train_test_split(data_temp, train_size=0.6, shuffle=True)
    del(data_temp)
    
    dval= lgbm.Dataset(data=data_val[shadow_features],
                        label=data_val[TARGET],
                        feature_name=shadow_features,
                        categorical_feature=shadow_cat_features)
    
    shadow_model = lgbm.train(params=shadow_best_params,
                                train_set=dtrain,
                                num_boost_round=NUM_BOOST_ROUNDS,
                                valid_sets=[dval],
                                feature_name=shadow_features,
                                categorical_feature=shadow_cat_features,
                                early_stopping_rounds=0.05*NUM_BOOST_ROUNDS,
                                verbose_eval=100)
    shadow_model.params["objective"] = "regression"
    shap_values = shap.TreeExplainer(shadow_model).shap_values(data_test[shadow_features])
    shap_values_df = pd.DataFrame(data = shap_values, columns = shadow_features)
    shap_values_results_df = pd.concat([shap_values_results_df, shap_values_df], axis = 0)
    
cols_to_keep = []
n = shap_values_results_df.shape[0]
for col in features:
    x = np.sum(np.where(shap_values_results_df[col] < shap_values_results_df[col + '_shadow'], 1, 0))
    p_val = binom_test(x=x, n=n, p=0.5, alternative='greater')
    if p_val <= 0.05:
        cols_to_keep += [col]
        
# print(model.best_score)        0.8455077135158744   
# cols_to_keep = ['cat1', 'cat2', 'cat4', 'cat6', 'cat7', 'cont0', 'cont1', 'cont3', 'cont4', 'cont5', 'cont7', 'cont8', 'cont9', 'cont10', 'cont11', 'cont12']




#!################################################################################################################################
#!################################################################################################################################
#! Determing optimal parameters for model with selected features.

dtrain = lgbm.Dataset(data=data[cols_to_keep],
                        label=data[TARGET],
                        feature_name=cols_to_keep,
                        categorical_feature=[x for x in cols_to_keep if 'cat' in x])

model = opt.integration.lightgbm.LightGBMTunerCV(params = starting_params,
                                                    train_set = dtrain,
                                                    num_boost_round = NUM_BOOST_ROUNDS,
                                                    nfold = NUM_FOLDS,
                                                    stratified=False,
                                                    shuffle = True,
                                                    feature_name=cols_to_keep,
                                                    categorical_feature=[x for x in cols_to_keep if 'cat' in x],
                                                    early_stopping_rounds=0.05*NUM_BOOST_ROUNDS,
                                                    verbose_eval = 100,
                                                    seed = 51)
model.run()
print(model.best_score)
selected_features_params =  model.best_params



dtrain = lgbm.Dataset(data=data[features],
                        label=data[TARGET],
                        feature_name=features,
                        categorical_feature=cat_features)

model2 = opt.integration.lightgbm.LightGBMTunerCV(params = starting_params,
                                                    train_set = dtrain,
                                                    num_boost_round = NUM_BOOST_ROUNDS,
                                                    nfold = NUM_FOLDS,
                                                    stratified=False,
                                                    shuffle = True,
                                                    feature_name=features,
                                                    categorical_feature=cat_features,
                                                    early_stopping_rounds=0.05*NUM_BOOST_ROUNDS,
                                                    verbose_eval = 100,
                                                    seed = 51)
model2.run()
print(model2.best_score)
all_features_params =  model.best_params





dtrain = lgbm.Dataset(data=data[cols_to_keep],
                        label=data[TARGET],
                        feature_name=cols_to_keep,
                        categorical_feature=[x for x in cols_to_keep if 'cat' in x])

model = opt.integration.lightgbm.LightGBMTunerCV(params = starting_params,
                                                    train_set = dtrain,
                                                    num_boost_round = NUM_BOOST_ROUNDS,
                                                    nfold = NUM_FOLDS,
                                                    stratified=False,
                                                    shuffle = True,
                                                    feature_name=cols_to_keep,
                                                    categorical_feature=[x for x in cols_to_keep if 'cat' in x],
                                                    early_stopping_rounds=0.05*NUM_BOOST_ROUNDS,
                                                    verbose_eval = 100,
                                                    seed = 51)
model.run()
print(model.best_score)
selected_features_params =  model.best_params





    params['learning_rate'] = learning_rate
    best_iter_list = []
        
    for i in range(num_iter):
        X_train, X_val, y_train, y_val = train_test_split(X[features], X[target], train_size = train_sample, shuffle = True, stratify = X[target])
        if len(cat_features) == 0:
            X_train_smote, y_train_smote = SMOTE(random_state=51).fit_resample(X_train, y_train)
        else:
            X_train_smote, y_train_smote = SMOTENC(categorical_features=cat_features_indices, random_state=51).fit_resample(X_train, y_train)
        
        del(X_train, y_train)
        gc.collect()
        
        dtrain = lgbm.Dataset(data = X_train_smote,
                                label = y_train_smote,
                                feature_name = features,
                                categorical_feature = cat_features)
        
        dval = lgbm.Dataset(data = X_val,
                            label = y_val,
                            feature_name = features,
                            categorical_feature = cat_features)
        
        del(X_train_smote, y_train_smote, X_val, y_val)
        gc.collect()
        
        current_model = lgbm.train(params=params,
                                    train_set=dtrain,
                                    num_boost_round=num_boost_rounds,
                                    valid_sets=[dval],
                                    feature_name=features,
                                    categorical_feature=cat_features,
                                    early_stopping_rounds=1000,
                                    verbose_eval=0)

        best_iter_current = current_model.best_iteration
        best_iter_list += [best_iter_current]
        
        print(current_model.best_score)

    return int(np.nanmean(best_iter_list))