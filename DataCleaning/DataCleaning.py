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
df = pd.concat([df_mayor60,df_todaedad]).drop_duplicates().reset_index(drop=True)
df.columns = ["PROVINCIA_ISO", "SEXO", "GRUPO_EDAD", "FECHA", "NUM_CASOS", "NUM_HOSP", "NUM_UCI", "NUM_DEFU"]
df["FECHA"] = pd.to_datetime(df["FECHA"])
col_str = ["PROVINCIA_ISO", "SEXO", "GRUPO_EDAD"]
for x in col_str:
  df[x] = df[x].str.upper()
  df[x] = df[x].str.strip()
df['PROVINCIA_ISO'] = df['PROVINCIA_ISO'].replace(['NC'], 'NA')
  
print(df.shape)
'''
Limpieza base de datos climatologica de España por provincia
'''
path_cli = path_data + "Data_Climatologica/Diaria/"
path_ref = path_data + "Data_Referencia/"
path_est = path_data + "Estandarizada/"

dir_list = os.listdir(path_cli)
result = pd.DataFrame()
for i in dir_list:
    dfi = pd.read_json(path_cli + i, encoding="latin")
    result = pd.concat([result, dfi], ignore_index=True)
    
result.columns = ['FECHA', 'INDICATIVO', 'NOMBRE', 'PROVINCIA', 'ALTITUD', 'TEMP_MED', 'PREC', 'TEMP_MIN', 'HORA_TEMP_MIN',
              'TEMP_MAX', 'HORA_TEMP_MAX', 'DIR', 'VEL_MEDIA', 'RACHA', 'HORA_RACHA', 'PRES_MAX', 'HORA_PRES_MAX', 
              'PRES_MIN', 'HORA_PRES_MIN', 'SOL']
result.fillna(method="ffill", inplace=True)
result.fillna(method="bfill", inplace=True)
result['PREC'] = result['PREC'].replace(['Ip'], '0,0')
result["FECHA"] = pd.to_datetime(result["FECHA"])
result["TEMP_MED"] = result["TEMP_MED"].str.replace(",",".").astype(float)
result["PREC"] = result["PREC"].str.replace(",",".").astype(float)
result["TEMP_MIN"] = result["TEMP_MIN"].str.replace(",",".").astype(float)
result["TEMP_MAX"] = result["TEMP_MAX"].str.replace(",",".").astype(float)
result["VEL_MEDIA"] = result["VEL_MEDIA"].str.replace(",",".").astype(float)
result["RACHA"] = result["RACHA"].str.replace(",",".").astype(float)
result["SOL"] = result["SOL"].str.replace(",",".").astype(float)
result["PRES_MAX"] = result["PRES_MAX"].str.replace(",",".").astype(float)
result["PRES_MIN"] = result["PRES_MIN"].str.replace(",",".").astype(float)
result["HORA_TEMP_MAX"] = result["HORA_TEMP_MAX"].str.split(":", expand=True)[0]
result['HORA_TEMP_MAX'] = result['HORA_TEMP_MAX'].replace(['24'], '00')
result["HORA_TEMP_MIN"] = result["HORA_TEMP_MIN"].str.split(":", expand=True)[0]
result['HORA_TEMP_MIN'] = result['HORA_TEMP_MIN'].replace(['24'], '00')
result["HORA_PRES_MAX"] = result["HORA_PRES_MAX"].str.split(":", expand=True)[0]
result['HORA_PRES_MAX'] = result['HORA_PRES_MAX'].replace(['24'], '00')
result["HORA_PRES_MIN"] = result["HORA_PRES_MIN"].str.split(":", expand=True)[0]
result['HORA_PRES_MIN'] = result['HORA_PRES_MIN'].replace(['24'], '00')
result["HORA_RACHA"] = result["HORA_RACHA"].str.split(":", expand=True)[0]
result['HORA_RACHA'] = result['HORA_RACHA'].replace(['24'], '00')
result['HORA_RACHA'] = result['HORA_RACHA'].replace(['79'], result[result["PROVINCIA_ISO"] == list(result[result["HORA_RACHA"] == "79"]["PROVINCIA_ISO"])[0]]["HORA_RACHA"].mode())
result['HORA_RACHA'] = result['HORA_RACHA'].replace(['72'], result[result["PROVINCIA_ISO"] == list(result[result["HORA_RACHA"] == "72"]["PROVINCIA_ISO"])[0]]["HORA_RACHA"].mode())
result['HORA_RACHA'] = result['HORA_RACHA'].replace(['80'], result[result["PROVINCIA_ISO"] == list(result[result["HORA_RACHA"] == "80"]["PROVINCIA_ISO"])[0]]["HORA_RACHA"].mode())
iso_list = list(result[result["HORA_RACHA"] == "75"]["PROVINCIA_ISO"].unique())
result["HORA_RACHA"] = np.where((result["PROVINCIA_ISO"] == iso_list[0]) & (result["HORA_RACHA"] == "75"), result[result["PROVINCIA_ISO"] == iso_list[0]]["HORA_RACHA"].mode(), result["HORA_RACHA"])
result["HORA_RACHA"] = np.where((result["PROVINCIA_ISO"] == iso_list[1]) & (result["HORA_RACHA"] == "75"), result[result["PROVINCIA_ISO"] == iso_list[1]]["HORA_RACHA"].mode(), result["HORA_RACHA"])
result = result.drop(columns=['INDICATIVO', 'NOMBRE'])
df_iso = pd.read_csv(path_ref + 'cod_iso_provincias.csv', encoding="utf-8", keep_default_na=False)
result_iso = result.merge(df_iso, how='inner', on='PROVINCIA')

print(result_iso.shape)

'''
Union de los dos dataframes
'''

df_total = df.merge(result_iso, how='inner', on=['FECHA', 'PROVINCIA_ISO'])
print(df_total.shape)
#df_total.to_csv(path_est + 'data_total.csv', index=False)
df_total.to_parquet(path = path_est + 'data_total', engine = 'auto', compression ='snappy', index=None)

# Para saber cual información no cruza entre los megre anteriores.
#df_no=pd.merge(df,result_iso,on=['FECHA', 'PROVINCIA_ISO'],how="outer",indicator=True)
#df_no=df_no[df_no['_merge']=='left_only']
#df_no.to_csv(path_est + 'data_no_cruce.csv', index=False)
