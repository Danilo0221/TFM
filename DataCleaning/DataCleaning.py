import pandas as pd
import os
import numpy as np

'''
Limpieza base de datos de covid-19 en España por provincia
'''
path = os.getcwd()
path = path.replace("\\", "\\\\")
path = path.replace("\\\\", "/") + "/"
path_data = path + "Data/"
path_covid = path_data + "Data_Covid/"
df_mayor60 = pd.read_csv(path_covid + "casos_hosp_uci_def_sexo_edad_provres_60_mas.csv", keep_default_na=False, na_values="")
df_todaedad = pd.read_csv(path_covid + "casos_hosp_uci_def_sexo_edad_provres.csv", keep_default_na=False, na_values="")
df_covid = pd.concat([df_mayor60,df_todaedad]).drop_duplicates().reset_index(drop=True)
df_covid.columns = ["PROVINCIA_ISO", "SEXO", "GRUPO_EDAD", "FECHA", "NUM_CASOS", "NUM_HOSP", "NUM_UCI", "NUM_DEFU"]
df_covid["FECHA"] = pd.to_datetime(df_covid["FECHA"])
col_str = ["PROVINCIA_ISO", "SEXO", "GRUPO_EDAD"]
for x in col_str:
  df_covid[x] = df_covid[x].str.upper()
  df_covid[x] = df_covid[x].str.strip()
df_covid['PROVINCIA_ISO'] = df_covid['PROVINCIA_ISO'].replace(['NC'], 'NA')
  
print(df_covid.shape)
'''
Limpieza base de datos climatologica de España por provincia
'''
path_cli = path_data + "Data_Climatologica/Diaria/"
path_ref = path_data + "Data_Referencia/"
path_est = path_data + "Estandarizada/"

dir_list = os.listdir(path_cli)
df_clima = pd.DataFrame()
for i in dir_list:
    dfi = pd.read_json(path_cli + i, encoding="latin")
    df_clima = pd.concat([df_clima, dfi], ignore_index=True)
    
df_clima.columns = ['FECHA', 'INDICATIVO', 'NOMBRE', 'PROVINCIA', 'ALTITUD', 'TEMP_MED', 'PREC', 'TEMP_MIN', 'HORA_TEMP_MIN',
              'TEMP_MAX', 'HORA_TEMP_MAX', 'DIR', 'VEL_MEDIA', 'RACHA', 'HORA_RACHA', 'PRES_MAX', 'HORA_PRES_MAX', 
              'PRES_MIN', 'HORA_PRES_MIN', 'SOL']
df_clima.fillna(method="ffill", inplace=True)
df_clima.fillna(method="bfill", inplace=True)
df_clima['PREC'] = df_clima['PREC'].replace(['Ip'], '0,0')
df_clima["FECHA"] = pd.to_datetime(df_clima["FECHA"])
df_clima["TEMP_MED"] = df_clima["TEMP_MED"].str.replace(",",".").astype(float)
df_clima["PREC"] = df_clima["PREC"].str.replace(",",".").astype(float)
df_clima["TEMP_MIN"] = df_clima["TEMP_MIN"].str.replace(",",".").astype(float)
df_clima["TEMP_MAX"] = df_clima["TEMP_MAX"].str.replace(",",".").astype(float)
df_clima["VEL_MEDIA"] = df_clima["VEL_MEDIA"].str.replace(",",".").astype(float)
df_clima["RACHA"] = df_clima["RACHA"].str.replace(",",".").astype(float)
df_clima["SOL"] = df_clima["SOL"].str.replace(",",".").astype(float)
df_clima["PRES_MAX"] = df_clima["PRES_MAX"].str.replace(",",".").astype(float)
df_clima["PRES_MIN"] = df_clima["PRES_MIN"].str.replace(",",".").astype(float)
df_clima["HORA_TEMP_MAX"] = df_clima["HORA_TEMP_MAX"].str.split(":", expand=True)[0] # Validar el replace 24 a 00
df_clima['HORA_TEMP_MAX'] = df_clima['HORA_TEMP_MAX'].replace(['24'], '00')
df_clima["HORA_TEMP_MIN"] = df_clima["HORA_TEMP_MIN"].str.split(":", expand=True)[0]
df_clima['HORA_TEMP_MIN'] = df_clima['HORA_TEMP_MIN'].replace(['24'], '00')
df_clima["HORA_PRES_MAX"] = df_clima["HORA_PRES_MAX"].str.split(":", expand=True)[0]
df_clima['HORA_PRES_MAX'] = df_clima['HORA_PRES_MAX'].replace(['24'], '00')
df_clima["HORA_PRES_MIN"] = df_clima["HORA_PRES_MIN"].str.split(":", expand=True)[0]
df_clima['HORA_PRES_MIN'] = df_clima['HORA_PRES_MIN'].replace(['24'], '00')
df_clima = df_clima.drop(columns=['INDICATIVO', 'NOMBRE'])

df_clima["HORA_RACHA"] = df_clima["HORA_RACHA"].str.split(":", expand=True)[0]
df_clima['HORA_RACHA'] = df_clima['HORA_RACHA'].replace(['24'], '00')
df_clima['HORA_RACHA'] = df_clima['HORA_RACHA'].replace(['79'], df_clima[df_clima["PROVINCIA"] == list(df_clima[df_clima["HORA_RACHA"] == "79"]["PROVINCIA"])[0]]["HORA_RACHA"].mode())
df_clima['HORA_RACHA'] = df_clima['HORA_RACHA'].replace(['72'], df_clima[df_clima["PROVINCIA"] == list(df_clima[df_clima["HORA_RACHA"] == "72"]["PROVINCIA"])[0]]["HORA_RACHA"].mode())
df_clima['HORA_RACHA'] = df_clima['HORA_RACHA'].replace(['80'], df_clima[df_clima["PROVINCIA"] == list(df_clima[df_clima["HORA_RACHA"] == "80"]["PROVINCIA"])[0]]["HORA_RACHA"].mode())
iso_list = list(df_clima[df_clima["HORA_RACHA"] == "75"]["PROVINCIA"].unique())
df_clima["HORA_RACHA"] = np.where((df_clima["PROVINCIA"] == iso_list[0]) & (df_clima["HORA_RACHA"] == "75"), df_clima[df_clima["PROVINCIA"] == iso_list[0]]["HORA_RACHA"].mode(), df_clima["HORA_RACHA"])
df_clima["HORA_RACHA"] = np.where((df_clima["PROVINCIA"] == iso_list[1]) & (df_clima["HORA_RACHA"] == "75"), df_clima[df_clima["PROVINCIA"] == iso_list[1]]["HORA_RACHA"].mode(), df_clima["HORA_RACHA"])

'''
Union dataframe clima y cod_iso_provincias
'''

df_iso = pd.read_csv(path_ref + 'cod_iso_provincias.csv', encoding="utf-8", keep_default_na=False)
df_clima_iso = df_clima.merge(df_iso, how='inner', on='PROVINCIA')
print(df_clima_iso.shape)

'''
Union de los dataframes clima_iso y covid
'''

df_total = df_covid.merge(df_clima_iso, how='inner', on=['FECHA', 'PROVINCIA_ISO'])
print(df_total.shape)
#df_total.to_csv(path_est + 'data_total.csv', index=False)
df_total.to_parquet(path = path_est + 'data_total', engine = 'auto', compression ='snappy', index=None)

# Para saber cual información no cruza entre los megre anteriores.
#df_no=pd.merge(df,df_clima_iso,on=['FECHA', 'PROVINCIA_ISO'],how="outer",indicator=True)
#df_no=df_no[df_no['_merge']=='left_only']
#df_no.to_csv(path_est + 'data_no_cruce.csv', index=False)
