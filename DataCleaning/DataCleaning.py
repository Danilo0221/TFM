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
path = os.getcwd()
path = path.replace("\\", "\\\\")
path = path.replace("\\\\", "/")
path = "/".join(path.split("/")[:-1]) + "/Data/Data_Climatologica/"

