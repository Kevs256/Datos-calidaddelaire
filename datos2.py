import math
from re import X
import pandas as pd
import matplotlib.pyplot as plot
import matplotlib
from matplotlib.figure import Figure
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import csv

# los datos ya fueron obtenidos por los expertos en la materia por medio de sensores
# en una ciudada contaminada
# de manera practica no poseemos ningun experto en la materia,
# pero suponemos que todos los datos fueron bien tomados y guardan una relacion logica

# en el archivo de datos vienen datos vacios al final de la hoja que fueron eliminados
# en el archivho se lee "Los valores faltantes se etiquetan con el valor -200"
# que nos indican sobre valores que faltaron o se dañaron

file = "AirQualityUCI.csv"
datacsv = pd.read_csv(file, sep=';', decimal=",")

column = ['Date',
          'Time',
          'CO(GT)',
          'PT08.S1(CO)',
          'NMHC(GT)',
          'C6H6(GT)',
          'PT08.S2(NMHC)',
          'NOx(GT)',
          'PT08.S3(NOx)',
          'NO2(GT)',
          'PT08.S4(NO2)',
          'PT08.S5(O3)',
          'T',
          'RH',
          'AH',
          'vacia1',
          'vacia2']
# hay dos columnas vacias al final de la hoja de datos
datacsv.columns = column

# lista numerica, columnas que contienen datos numericos posibles de conmutar
columnNumber = ['CO(GT)',
                'PT08.S1(CO)',
                'C6H6(GT)',
                'PT08.S2(NMHC)',
                'NOx(GT)',
                'PT08.S3(NOx)',
                'NO2(GT)',
                'PT08.S4(NO2)',
                'PT08.S5(O3)',
                'T',
                'RH',
                'AH']

datacsv = datacsv.drop(columns=['NMHC(GT)'])
# eliminaremos todas las filas que contengan un dato en -200 para no afectar el modelo
# este meotodo retorna la lista de indices a eliminar

# como la columna NMHC(GT) contiene aprox 8500 datos en -200 nnos indica que el sensor
# se ha dañado, entonces se ha decidido eliminar dicha columna de datos


def eliminar200():
    eliminar200 = []
    for val in datacsv.columns:
        for i in range(len(datacsv)):
            if (datacsv[val][i] == -200):
                eliminar200.append(i)
    return (list(set(eliminar200)))

# ahora encontraremos los datos atipicos de este dataset
# con esta funcion encontramos los cuartiles 1, 3 y el rango intercuartil


def quantiles(column):
    newlist = []
    qr1 = (datacsv[column].quantile(q=0.25))
    qr3 = (datacsv[column].quantile(q=0.75))
    iqr = qr3-qr1
    newlist.append(qr1)
    newlist.append(qr3)
    newlist.append(iqr)
    return newlist

# genera la lista de indices que se deben eliminar de una columna, recibe la lista anteriorir de qr's


def todelete(column, newlist):
    deleted = []
    for i in range(0, len(datacsv[column])):
        if (datacsv[column][i] < newlist[0]-(1.5*newlist[2])):
            deleted.append(i)
        if (datacsv[column][i] > newlist[1] + (1.5*newlist[2])):
            deleted.append(i)
    return deleted

# eliminaremos todos los atipicos de todas las columnas para poder generar un buen modelo


def todeletelist(listaColumnas):
    deletedtotal = []
    listasumada = []
    for i in range(len(listaColumnas)):
        newlist = quantiles(listaColumnas[i])
        listasumada = todelete(listaColumnas[i], newlist) + listasumada
    deletedtotal = list(set(listasumada))
    return deletedtotal

# elimina los indices, la lista a eliminar


def deleting(todelete, datacsv):
    datacsv2 = datacsv.drop(datacsv.index[todelete])
    return datacsv2


# procedemos a eliminar los valores en -200 para poder ver claramente la distribucion de los datos
# una vez programados los metodos procedemos a eliminar los -200
lista200 = eliminar200()
datacsv = deleting(lista200, datacsv)

#################################################################################################
#################################################################################################
#################################################################################################

# pero hay que transformar los datos antes de eliminar los atipicos

# de acuerdo al excel donde pruebo cada una de las columnas en las 4
# posibles operaciones de transfromacion las mejores fueron las siguientes (anexar foto)
# raiz para NO2(GT), PT08.S4(NO2), PT08.S5(O3), AH
# 4 datos de entrada
# sin embargo sin transformacion funcionan 2 variables, y en logaritmo una mas

# 'PT08.S1(CO)'
# 'C6H6(GT)'
# 'PT08.S2(NMHC)'
# 'NOx(GT)'
# 'PT08.S3(NOx)'
# 'NO2(GT)'
# 'PT08.S4(NO2)'
# 'PT08.S5(O3)'
# 'T'
# 'RH'
# 'AH'
# X = datacsv[['RH']]

# X=pow(np.array(X),2)
# X=pow(np.array(X),0.5)
# X=np.log10(X)
# X=np.exp(np.array(X))

# fig = plot.figure()
# figura = plot.hist(X)
# plot.show()
# plot.subplots()

# una vez terminada la prueba visual de las variables entrada procedemos a construir en dataframe
# construimos el nuevo dataframe de las cuatro variables que arrojaron los mejores resultados
# unimos el dataframe con la variable de salida para limpiarlos conjuntamente 

Z = datacsv[['RH']]

Y = datacsv[['CO(GT)']]

X = datacsv[['NO2(GT)','PT08.S4(NO2)','PT08.S5(O3)','AH']]
X = pow(np.array(X),0.5)
X = pd.DataFrame(X)

W = datacsv[['PT08.S1(CO)']]
W = np.log10(W)
W = pd.DataFrame(W)

R = datacsv[['PT08.S2(NMHC)']]
R = pow(np.array(R),0.5)
R = pd.DataFrame(R)

P = datacsv[['PT08.S3(NOx)']]
P = np.log10(P)
P = pd.DataFrame(P)

L = datacsv[['PT08.S4(NO2)']]


column = ['NO2(GT)','PT08.S4(NO2)','PT08.S5(O3)','AH']
X.columns=column

datacsv = X.assign(RH = Z.values)
datacsv = datacsv.assign(PT8S1CO = W.values)
datacsv = datacsv.assign(PT8S2NMHC = R.values)
datacsv = datacsv.assign(PT8S3NOX = P.values)
datacsv = datacsv.assign(PT8S4NO2 = L.values)
datacsv = datacsv.assign(CO = Y.values)

# procedemos a limpiar el dataframe nuevo
columnNumber = ['NO2(GT)','PT08.S4(NO2)','PT08.S5(O3)','AH','RH','PT8S1CO','PT8S2NMHC','PT8S3NOX','PT8S4NO2','CO']
listaAtipicos=todeletelist(columnNumber)
datacsv = deleting(listaAtipicos, datacsv)

#procedemos a crear el modelo
X = datacsv[['NO2(GT)','PT08.S4(NO2)','PT08.S5(O3)','AH','RH','PT8S1CO','PT8S2NMHC','PT8S3NOX','PT8S4NO2']]
Y = datacsv[['CO']]

resultados=[]

for i in range(11):
    X_train, X_test, y_train, y_test = train_test_split(
                                        X,
                                            Y,
                                            train_size   = 0.5,
                                        )
    modelo = LinearRegression()
    modelo.fit(X = np.array(X_train), y = y_train)
    datosobtenidos = modelo.predict(X = np.array(X_test))
    r2 = r2_score(y_true  = y_test, y_pred  = datosobtenidos)
    resultados.append(r2)

print(np.mean(r2))
