import pandas as pd
import os

'''
Limpieza base de datos de covid-19 en España por provincia
'''
path = os.getcwd()
path = path.replace("\\", "\\\\")
path = path.replace("\\\\", "/")
path = "/".join(path.split("/")[:-1]) + "/Data/Data_Covid/"
df_mayor60 = pd.read_csv(path + "casos_hosp_uci_def_sexo_edad_provres_60_mas.csv")
df_todaedad = pd.read_csv(path + "casos_hosp_uci_def_sexo_edad_provres.csv")
df = pd.concat([df_mayor60,df_todaedad]).drop_duplicates().reset_index(drop=True)
df = df[df["provincia_iso"].notna()]
df.columns = ["PROVINCIA_ISO", "SEXO", "GRUPO_EDAD", "FECHA", "NUM_CASOS", "NUM_HOSP", "NUM_UCI", "NUM_DEFU"]
df["FECHA"] = pd.to_datetime(df["FECHA"])
col_str = ["PROVINCIA_ISO", "SEXO", "GRUPO_EDAD"]
for x in col_str:
  df[x] = df[x].str.upper()
  df[x] = df[x].str.strip()
  

'''
Limpieza base de datos climatologica de España por provincia
'''
path_cli = os.getcwd()
path_cli = path_cli.replace("\\", "\\\\")
path_cli = path_cli.replace("\\\\", "/")
path_cli = "/".join(path_cli.split("/")[:-1]) + "/Data/Data_Climatologica/Diaria/"

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
result = result.drop(columns=['unidad_generadora', 'periodicidad', 'descripcion', 'formato', 'copyright', 'notaLegal', 'campos'])
result = result.drop(columns=['INDICATIVO', 'NOMBRE'])

