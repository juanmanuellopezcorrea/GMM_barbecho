#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 11:56:25 2020

@author: Juancho08
"""




import GMM_class as cl
import os
import pickle 

dir_orignal_img ="/Users/Juancho08/Desktop/Barbercho/GMM/SORHA_Juan/Original/"  # imagenes de entrenamiento
dir_manual_mask="/Users/Juancho08/Desktop/Barbercho/GMM/SORHA_Juan/Mascaras/"  # mascaras manuales
dir_img="/Users/Juancho08/Desktop/Barbercho/GMM/SORHA_Juan/Img_predicciones/"   # las que predice
dir_gmm="/Users/Juancho08/Desktop/Barbercho/GMM/SORHA_Juan/Best_prob/" # mask predichas
lstcomponent_maleza=[1,2,3,4]#,5,6,7,8]
lstcomponent_suelo=[1,2,3,4]#],5,6,7,8]
covarianza=["tied","full"]#"full"
for i in lstcomponent_maleza:
    for j in lstcomponent_suelo:
        for a in covarianza:
            models=cl.GMM_malezas.trainingGMM_rastrojo_maleza(dir_orignal_img, dir_manual_mask,i,j,a)
            models_maleza=models[0]    
            models_rastrojo=models[1]
            name=str(i)+"M"+str(j)+"S"+str(a)
            if not os.path.exists(dir_gmm+"modelos"):
                os.mkdir(dir_gmm+"modelos")	
                print('directorio creado: ' + dir_gmm+"modelos")
                
            path_gmm = os.path.join(dir_gmm, "modelos/")
	       
            pickle.dump(models_rastrojo, open(path_gmm+name+"suelo", 'wb'))
            pickle.dump(models_maleza, open(path_gmm+name+"maleza", 'wb'))
            write_dir_test=dir_gmm+name
            segmentaciones=cl.GMM_malezas.prediction(dir_img,models_maleza,models_rastrojo,write_dir_test)
            print("imagen guardada")


# Ver como guardar los modelos nico lo hace en su script



