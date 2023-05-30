import pandas as pd
import numpy as np
import json
import os
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import FeatureHasher
from sklearn import linear_model
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn import metrics
from matplotlib import style
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error , r2_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from scipy import stats
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline

from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from skforecast.utils import save_forecaster
from skforecast.utils import load_forecaster
from sklearn.ensemble import RandomForestRegressor
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

def hash(n_feat, df, column):
    fh = FeatureHasher(n_features=n_feat, input_type='string')
    hashed = fh.transform(df[column])
    hashed_df = pd.DataFrame(hashed.toarray())
    df = pd.concat([df, hashed_df], axis=1)
    df = df.drop(columns=['PROVINCIA'])
    df.rename(columns = {0:'HASH_0'}, inplace = True)
    df.rename(columns = {1:'HASH_1'}, inplace = True)
    df.rename(columns = {2:'HASH_2'}, inplace = True)
    df.rename(columns = {3:'HASH_3'}, inplace = True)
    df.rename(columns = {4:'HASH_4'}, inplace = True)
    df.rename(columns = {5:'HASH_5'}, inplace = True)
    return df

def datamining():
    print("Iniciando proceso de DataMining...")
    path = os.getcwd()
    path = path.replace("\\", "\\\\")
    path = path.replace("\\\\", "/") + "/"
    path_data = path + "Data/"
    path_est = path_data + "Estandarizada/"
    path_models = path_data + "Models/"
    path_metrics = path_data + "Metrics/"

    # ##################################################################################
    # ######################## Regresión Lineal Múltiple ###############################
    # ##################################################################################

    print("Iniciando Regresión Lineal Múltiple")
    df = pd.read_csv(path_est + "data_refined.csv", keep_default_na=False, na_values="", sep=',')
    df['FECHA'] = pd.to_datetime(df.FECHA).dt.to_period('m')
    df = hash(6, df, 'PROVINCIA')
    df = df.assign(MONTH = lambda x: (x['FECHA'].astype(str).str.slice(start=5).astype(int)))

    selected_features = ['MONTH', 'ALTITUD', 'TEMP_MED', 'PREC', 'DIR', 'VEL_MEDIA', 'RACHA', 'PRES_MIN', 'SOL'
                         ,'HASH_0', 'HASH_1', 'HASH_2', 'HASH_3', 'HASH_4', 'HASH_5']
    train = df[(df['FECHA'] >= '2020-01-01 00:00:00') & (df['FECHA'] <= '2022-01-31 23:59:59')]
    test  = df[(df['FECHA'] >= '2022-02-01 00:00:00')]
    # print(round(test.shape[0]/df.shape[0], 2))

    X_train = train[selected_features]
    X_test  = test[selected_features]
    y_train = train['TASA_INCIDENCIA'].values.reshape(-1,1)
    y_test  = test['TASA_INCIDENCIA'].values.reshape(-1,1)

    X_train = sm.add_constant(X_train, prepend=True)
    modelo = sm.OLS(endog=y_train, exog=X_train,)
    modelo = modelo.fit()

    # save the model to disk
    filename = 'model_regresion_lineal.pkl'
    pickle.dump(modelo, open(path_models + filename, 'wb'))

    # load the model from disk
    # loaded_model = pickle.load(open(filename, 'rb'))
    # result = loaded_model.score(X_test, Y_test)
    # print(result)

    y_train = y_train.flatten()
    prediccion_train = modelo.predict(exog = X_train)
    residuos_train   = prediccion_train - y_train

    # Error de test del modelo 
    X_test = sm.add_constant(X_test, prepend=True)
    predicciones = modelo.predict(exog = X_test)
    mse_rl = mean_squared_error(
            y_true  = y_test,
            y_pred  = predicciones,
            squared = True
           )
    rmse_rl = mean_squared_error(
            y_true  = y_test,
            y_pred  = predicciones,
            squared = False
           )
    mae_rl =  mean_absolute_error(y_true  = y_test, y_pred  = predicciones)
    r2_rl = modelo.rsquared
    # print("")
    # print(f"El error (R2) de test es: {r2_rl}")
    # print(f"El error (mse) de test es: {mse_rl}")
    # print(f"El error (rmse) de test es: {rmse_rl}")
    # print(f"El error (mae) de test es: {mae_rl}")
    # print("")

    # ##################################################################################
    # ######################## Random Forest Regresión #################################
    # ##################################################################################
    print("Iniciando Random Forest")
    df = pd.read_csv(path_est + "data_refined.csv", keep_default_na=False, na_values="", sep=',')
    df['FECHA'] = pd.to_datetime(df.FECHA).dt.to_period('m')
    df = hash(6, df, 'PROVINCIA')
    df = df.assign(MONTH = lambda x: (x['FECHA'].astype(str).str.slice(start=5).astype(int)))

    df_train, df_test = train_test_split(df, test_size=0.3)
    df_train = df_train.reset_index()
    df_test = df_test.reset_index()
    df_train = df_train.drop(columns=["index"])
    df_test = df_test.drop(columns=["index"])

    df_train = df_train.set_index('FECHA')
    df_test = df_test.set_index('FECHA')
    df_train.reset_index(drop = True, inplace = True)

    cv = TimeSeriesSplit()
    l_estimators = [2, 4, 8, 16, 32, 64, 128, 256] 

    total_scores = []
    for estimators in l_estimators:
        # print(estimators)
        fold_accuracy = []
        regressor =  RandomForestRegressor(n_estimators= estimators, criterion='absolute_error', random_state=0)
        for train_fold, test_fold in cv.split(df_train):
            #División train test aleatoria
            f_train = df_train.loc[train_fold]
            f_test = df_train.loc[test_fold]
            # entrenamiento y ejecución del modelo
            regressor.fit(X = f_train.drop(['TASA_INCIDENCIA'], axis=1), y = f_train['TASA_INCIDENCIA'])
            y_pred = regressor.predict(X = f_test.drop(['TASA_INCIDENCIA'], axis = 1))
            # evaluación del modelo
            mae = mean_absolute_error(f_test['TASA_INCIDENCIA'], y_pred)
            fold_accuracy.append(mae)
        total_scores.append(sum(fold_accuracy)/len(fold_accuracy))

    best_est = l_estimators[np.argmin(total_scores)]
    # constructor
    regressor =  RandomForestRegressor(n_estimators= best_est, criterion='absolute_error', random_state=0)
    # fit and predict
    regressor.fit(X = df_train.drop(['TASA_INCIDENCIA'], axis=1), y = df_train['TASA_INCIDENCIA'])
    # save the model to disk
    filename = 'model_random_forest.pkl'
    pickle.dump(regressor, open(path_models + filename, 'wb'))

    y_pred = regressor.predict(X = df_test.drop(['TASA_INCIDENCIA'], axis = 1))

    mse_rfr = mean_squared_error(
            y_true  = df_test['TASA_INCIDENCIA'],
            y_pred  = y_pred,
            squared = True
           )
    rmse_rfr = mean_squared_error(
            y_true  = df_test['TASA_INCIDENCIA'],
            y_pred  = y_pred,
            squared = False
           )
    mae_rfr = mean_absolute_error(df_test['TASA_INCIDENCIA'], y_pred)
    r2_rfr = r2_score(df_test['TASA_INCIDENCIA'], y_pred)

    # print("")
    # print(f"El error (mse) de test es: {mse_rfr}")
    # print(f"El error (rmse) de test es: {rmse_rfr}")
    # print(f"El error (mae) de test es: {mae_rfr}")
    # print(f"El error (r2) de test es: {r2_rfr}")
    # print("")

    # ##################################################################################
    # ######################## Serie Temporal ForecasterAutoreg ########################
    # ##################################################################################
    print("Iniciando Serie Temporal ForecasterAutoreg")
    df = pd.read_csv(path_est + "data_refined.csv", keep_default_na=False, na_values="", sep=',')
    df['FECHA'] = pd.to_datetime(df.FECHA).dt.to_period('m')

    df = df.drop(columns=['PROVINCIA'])

    df = df.groupby(['FECHA']).agg(
        {'ALTITUD': 'mean', 'TEMP_MED': 'mean', 
         'PREC': 'mean', 'DIR': 'mean', 'VEL_MEDIA': 'mean', 'RACHA': 'mean', 
         'PRES_MIN': 'mean', 'SOL': 'mean', 'TASA_INCIDENCIA': 'sum'}).reset_index()

    df['FECHA'] = df['FECHA'].astype(str)
    df['FECHA'] = pd.to_datetime(df['FECHA'], format='%Y/%m/%d')
    df = df.set_index('FECHA')
    df = df.asfreq('MS')
    df = df.sort_index()

    steps = 12
    datos_train = df[:-steps]
    datos_test  = df[-steps:]

    # Crear y entrenar forecaster
    forecaster = ForecasterAutoreg(
                    regressor = RandomForestRegressor(n_estimators= 256, criterion='absolute_error', random_state=112),
                    lags = 10
                 )

    forecaster.fit(y=datos_train['TASA_INCIDENCIA'])
    # save the model to disk
    filename = 'model_forecasterAutoreg.pkl'
    pickle.dump(forecaster, open(path_models + filename, 'wb'))
    predicciones = forecaster.predict(steps=steps)

    mse_st = metrics.mean_squared_error(
                    y_true = datos_test['TASA_INCIDENCIA'],
                    y_pred = predicciones,
                    squared = True
                )
    rmse_st = metrics.mean_squared_error(
                    y_true = datos_test['TASA_INCIDENCIA'],
                    y_pred = predicciones,
                    squared = False
                )
    mae_st = metrics.mean_absolute_error(datos_test['TASA_INCIDENCIA'], predicciones)
    r2_st = metrics.r2_score(datos_test['TASA_INCIDENCIA'], predicciones)
    # print("")
    # print(f"El error (mse) de test : {mse_st}")
    # print(f"El error (rmse) de test es: {rmse_st}")
    # print(f"El error (mae) de test es: {mae_st}")
    # print(f"El error (r2) de test es: {r2_st}")
    # print("")

    data = [['Regresión Lineal Múltiple', 'R2', r2_rl], ['Regresión Lineal Múltiple', 'MAE', mae_rl], ['Regresión Lineal Múltiple', 'RMSE', rmse_rl],
            ['Random Forest Regresión', 'R2', r2_rfr], ['Random Forest Regresión', 'MAE', mae_rfr], ['Random Forest Regresión', 'RMSE', rmse_rfr],
            ['ForecasterAutoreg', 'R2', r2_st], ['ForecasterAutoreg', 'MAE', mae_st], ['ForecasterAutoreg', 'RMSE', rmse_st]]
    df_mestrics = pd.DataFrame(data, columns = ['Modelo', 'Métrica', 'Valor'])
    print(df_mestrics)
    df_mestrics.to_csv(path_metrics + 'metrics.csv', index=False)
    print("Etapa Datamining Finalizada")
    print("")

