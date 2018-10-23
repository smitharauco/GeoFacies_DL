import numpy as np
import pandas as pd
import math
import re

def ReadObsFile(obsevation=''):

    valid, obs, value, mean, data, t_acu= [], [], [], [], [], []

    fin = open(obsevation, "r")
    lno = 0
    cont = 0
    for line in fin:

        cols = line.strip()
        cols = re.split('\t|\\t|\\\t|\\\\t|\\\\\t| |  |   |     |      |       |        |          |          |-|--|---|_|__|___',cols)
        data.append(cols[:])

        if(cols[0] != "TIME"):
            valid.append(float(cols[0]))
            value.append(float(cols[1]))
            mean.append(float(cols[2]))
        else:
            t_acu.append(int(cols[1]))  

        lno += 1
    fin.close()

    # Verificar se a quantidade de observações para cada tempo é igual 
    dt = pd.DataFrame(data)
    m = 1
    aux = 0
    while (dt.iloc[m,0] != 'TIME'):
        aux += 1
        m += 1
        if m == len(dt):
            break
    first = aux

    cont = 0
    i = 0
    observation_information = []

    for i in range(len(t_acu)):
        aux = 0
        n = i
        while dt.iloc[n+cont,0] != "TIME":
            aux += 1
            n += 1

        observation_information.append(int(aux))
        i += 1
        cont += aux

    observation_information[0] = first
    observation_information = np.array(observation_information)

    if len((np.unique(observation_information))) != 1:
        return 'A quantidade de observações para cada tempo estipulado é diferente'
    
    else:
        vector_R = []
        obs=[]
        for i in range(len(valid)):
            if valid[i] == 1.0:
                vector_R.append((float(mean[i]))**2)
                obs.append((float(value[i]))**2)

        R = np.diag(vector_R)

        elements_number = int(len(valid)/len(t_acu))

        act = np.array(valid).reshape((len(t_acu),elements_number))
        production = np.array(value).reshape((len(t_acu),elements_number))
        return np.array(obs).reshape((len(obs),1)), R, np.array(t_acu), act, production


