import pandas as pd
import numpy as np
import os

path = os.getcwd()
path = path.replace("\\", "\\\\")
path = path.replace("\\\\", "/") + "/"
path_data = path + "Data/"
path_est = path_data + "Estandarizada/"
path_ref = path_data + "Data_Referencia/"

df = pd.read_parquet(path = path_est + 'data_total', engine = 'auto')
df = df.drop(columns=['PROVINCIA_ISO','HORA_TEMP_MIN', 'HORA_TEMP_MAX', 'HORA_RACHA', 'HORA_PRES_MAX', 'HORA_PRES_MIN', 'NUM_HOSP', 'NUM_UCI', 'NUM_DEFU', 'GRUPO_EDAD', 'SEXO'])
#df['MONTH'] = pd.DatetimeIndex(df['FECHA']).month
#df['WEEK'] = df['FECHA'].dt.isocalendar().week.astype(int)
df['FECHA'] = pd.to_datetime(df.FECHA).dt.to_period('m')

df = df.groupby(['FECHA', 'PROVINCIA']).agg(
    {'NUM_CASOS':'sum', 'ALTITUD': 'mean', 'TEMP_MED': 'mean', 
     'PREC': 'mean', 'TEMP_MIN': 'mean', 'TEMP_MAX': 'mean', 'DIR': 'mean', 'VEL_MEDIA': 'mean', 'RACHA': 'mean', 
     'PRES_MAX': 'mean', 'PRES_MIN': 'mean', 'SOL': 'mean'}).reset_index()

# Se utiliza la geometrica 
df_pob = pd.read_csv(path_ref + "pobla_prov.csv", encoding="latin", keep_default_na=False, na_values="", sep=';')
df_pob = df_pob.drop(columns=['Sexo','Edad (año a año)', 'Españoles/Extranjeros'])
df_pob["Provincias"] = df_pob["Provincias"].str.slice(start=3)
df_pob = df_pob[df_pob["Año"] >= 2013]
df_pob["Total"] = df_pob["Total"].str.replace(".","").astype(int)
df_pob = df_pob.assign(FEC_POB_INI = lambda x: (x['Año'] + 1))
df_pob["Provincias"] = df_pob["Provincias"].str.upper()
df_pob["Provincias"] = df_pob["Provincias"].str.strip()
df_pob.columns = ["PROVINCIA", "YEAR_ACUM", "TOTAL_POB", "YEAR"]
df_pob['PROVINCIA'] = df_pob['PROVINCIA'].replace(['ALMERÍA'], 'ALMERIA')
df_pob['PROVINCIA'] = df_pob['PROVINCIA'].replace(['ARABA/ÁLAVA'], 'ARABA/ALAVA')
df_pob['PROVINCIA'] = df_pob['PROVINCIA'].replace(['ALICANTE/ALACANT'], 'ALICANTE')
df_pob['PROVINCIA'] = df_pob['PROVINCIA'].replace(['ÁVILA'], 'AVILA')
df_pob['PROVINCIA'] = df_pob['PROVINCIA'].replace(['BALEARS, ILLES'], 'ILLES BALEARS')
df_pob['PROVINCIA'] = df_pob['PROVINCIA'].replace(['CÁCERES'], 'CACERES')
df_pob['PROVINCIA'] = df_pob['PROVINCIA'].replace(['CÁDIZ'], 'CADIZ')
df_pob['PROVINCIA'] = df_pob['PROVINCIA'].replace(['CASTELLÓN/CASTELLÓ'], 'CASTELLON')
df_pob['PROVINCIA'] = df_pob['PROVINCIA'].replace(['CÓRDOBA'], 'CORDOBA')
df_pob['PROVINCIA'] = df_pob['PROVINCIA'].replace(['CORUÑA, A'], 'A CORUÑA')
df_pob['PROVINCIA'] = df_pob['PROVINCIA'].replace(['JAÉN'], 'JAEN')
df_pob['PROVINCIA'] = df_pob['PROVINCIA'].replace(['LEÓN'], 'LEON')
df_pob['PROVINCIA'] = df_pob['PROVINCIA'].replace(['MÁLAGA'], 'MALAGA')
df_pob['PROVINCIA'] = df_pob['PROVINCIA'].replace(['PALMAS, LAS'], 'LAS PALMAS')
df_pob['PROVINCIA'] = df_pob['PROVINCIA'].replace(['RIOJA, LA'], 'LA RIOJA')
df_pob['PROVINCIA'] = df_pob['PROVINCIA'].replace(['SANTA CRUZ DE TENERIFE'], 'STA. CRUZ DE TENERIFE')
df_pob['PROVINCIA'] = df_pob['PROVINCIA'].replace(['VALENCIA/VALÈNCIA'], 'VALENCIA')

d = {}
num_year = 2022 - 2019
prov = list(df_pob['PROVINCIA'].unique())
for i in prov:
    poblacion_inicial = df_pob[(df_pob["PROVINCIA"] == i) & (df_pob["YEAR_ACUM"] == 2019)]["TOTAL_POB"].values[0]
    poblacion_final = df_pob[(df_pob["PROVINCIA"] == i) & (df_pob["YEAR_ACUM"] == 2022)]["TOTAL_POB"].values[0]
    tasa_crecimiento_anual = ((poblacion_final / poblacion_inicial) ** (1 / num_year) - 1)
    tasa_crecimiento_mensual = (1 + tasa_crecimiento_anual) ** (1 / 12) - 1
    d[i] = tasa_crecimiento_mensual
    
tasa_prov = pd.DataFrame([d]).transpose()
tasa_prov.reset_index(inplace = True)
tasa_prov.rename(columns={0:"TASA_MENSUAL", "index": "PROVINCIA"}, inplace=True)

df_prov_fec = df[["PROVINCIA", "FECHA"]]
df_prov_fec = df_prov_fec.drop_duplicates().sort_values(by=['PROVINCIA', 'FECHA'])
df_prov_fec = df_prov_fec.assign(YEAR = lambda x: (x['FECHA'].astype(str).str.slice(stop=4).astype(int)))
DF_PROV = df_prov_fec.merge(df_pob, how='left', on=['PROVINCIA', 'YEAR'])
DF_PROV = DF_PROV.merge(tasa_prov, how='inner', on=['PROVINCIA'])
DF_PROV = DF_PROV.assign(POB_MEN = lambda x: (
    round(x["TOTAL_POB"] * (1 + x["TASA_MENSUAL"]) ** (x['FECHA'].astype(str).str.slice(start=5).astype(int))).astype(int)))
DF_PROV = DF_PROV.drop(columns=['YEAR', 'YEAR_ACUM', 'TOTAL_POB', 'TASA_MENSUAL'])

df = df.merge(DF_PROV, how='inner', on=["PROVINCIA", "FECHA"])
df = df.assign(TASA_INCIDENCIA = lambda x: (round((x["NUM_CASOS"] / x["POB_MEN"]) * 100000,2)))
df = df.drop(columns=['NUM_CASOS', 'POB_MEN'])

df = df.drop(columns=["TEMP_MIN", "TEMP_MAX", "PRES_MAX"])
df.to_csv(path_est + 'data_refined.csv', index=False)

