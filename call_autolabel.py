# -*- coding: utf-8 -*-

import autolabel_dire as autolabel
import pandas as pd
import os

#Variables


porcentaje= 1

split=[4,3]

weed_label="ELEIN_Romi"

best_segmentation="1M2Sfull"





paht='/Users/Juancho08/Desktop/Barbercho/GMM/'

if not os.path.exists(paht+weed_label+"/"+"prediction_mask/"):
                os.mkdir(paht+weed_label+"/"+"prediction_mask/")	
                os.mkdir(paht+weed_label+"/"+"prediction_mask/"+best_segmentation+"/")
                print('directorio creado:' +paht+weed_label+"/"+"prediction_mask/"+best_segmentation+"/")

write_dir_test=paht+ weed_label+"/prediction_mask/"+best_segmentation+"/"

dir_gmm_malezas=paht+ weed_label+'/Best_model/modelos/'+ best_segmentation+"maleza"

dir_gmm_rastrojo=paht+ weed_label+'/Best_model/modelos/'+ best_segmentation+"suelo"

path_csv_write=paht+ weed_label+'/prediction_mask/'+best_segmentation+"/"+weed_label+ "_" +best_segmentation+".csv"




df = pd.read_csv("/Users/Juancho08/Desktop/Barbercho/CSV/database.csv")
images_dir_1=df.loc[df["Label"] == weed_label] 
images_dir=images_dir_1["path_image"].tolist()

a=autolabel.self_labeled(porcentaje,split,images_dir,dir_gmm_malezas,dir_gmm_rastrojo,write_dir_test,weed_label,path_csv_write)

