# -*- coding: utf-8 -*-




"""
La función devuelve un csv con los nombres de los patch y la correspondiente a su especie de Maleza
"""

#Importamos las bibliotecas necesarias 
import pandas as pd
import pickle
import numpy as np
import cv2



# Funciones


def sliding_window_logico(image, stepSize_X,stepSize_y, windowSize):
	"""Función generadora que lleva adelante un sliding windows sin recortar la foto.

	Parámetros:
	image: numpy array que contiene la imagen completa
	stepSize: cantidad de píxeles que se movera la ventana tanto en horizontal como en vertical
	windowSize: dupla que contiene el ancho y alto de la ventana en píxeles (ancho,alto)

	Salidas:
	Por cada iteración genera una tupla con los siguientes elementos:
		-x1: coordenada en x de la esquina inferior izquierda del parche
		-y1: coordenada en y de la esquina inferior izquierda del parche
		-x2: coordenada en x de la esquina superior derecha del parche
		-y2: coordenada en y de la esquina superior derecha del parche

	"""
    
	for y in range(0, image.shape[0], stepSize_y):
		for x in range(0, image.shape[1], stepSize_X):
			yield (x, y, x + windowSize[1], y + windowSize[0])
            

def prediction(img_dir_test,gmm_malezas,gmm_rastrojo,write_dir_test):   
    count=0
    img=cv2.imread(img_dir_test)
    imgLUV=cv2.cvtColor(img, cv2.COLOR_BGR2LUV) # Trasformamos la imágen al espacio de color  LUV
    imgUV=imgLUV[:,:,1:3]  # le quitamos el L dejando solo UV
    imgUV=imgUV.reshape(((-1,2)))
    # Obtenemos  la prob de cada pixel a pertenecer a cada componente del GMM  para cada modelo     
    prob_suelo=gmm_rastrojo.predict_proba(imgUV)
    prob_maleza=gmm_malezas.predict_proba(imgUV)
    # Compara las probabilidades de cada pixel para cada componente y para cada modelo.
    df_suelo= np.amax(prob_suelo,1 ) # elige  la mayor prob en la fila
    df_maleza= np.amax(prob_maleza,1)
    idclases=df_maleza>df_suelo # En cada pixel compara con si df malezas es más grande que df suelo pone un True si no un False
    idmask=np.zeros(idclases.shape) #crea una matriz de 0
    idmask[np.where(idclases==True)]=0 # Donde hay un 0 en la matrix lo reemplaza por 0 si en el mismo pixel de la matriz idclasse tiene true
    idmask[np.where(idclases==False)]=255
    # Reconstruimos la imagen 
    original_shape=img.shape
    segmented=idmask.reshape(original_shape[0],original_shape[1])
    name_img=img_dir_test.split("/")[-1]
    cv2.imwrite(write_dir_test+(name_img[:-4])+'.png',segmented)
    
        
    print("Predición Gaussian Mixtures Malezas finalizada")
    return segmented 


def self_labeled(porcentaje,split,images_dir,dir_gmm_malezas,dir_gmm_rastrojo,write_dir_test,weed_label,path_csv_write):
    
    
    """
    Funcion que devuelve un dataframe 7 columnas con la siguiente información luego de realizat un scanning windows sobre una imágen.
    X1: cordenadas X minima del patch
    y1: cordenadas y minima del patch
    X1: cordenadas X máxima del patch
    y1: cordenadas y máxima del patch
    name_img: nombre de la imagen origen del patch
    id_patch: número identificador dentro del patch
    label: maleza a la que pertenezce el patch / si pertenece al suelo pero proviene de las imagenes de esa maleza
               
    """
    
    #Cargamos los modelos
    gmm_malezas=pickle.load(open(dir_gmm_malezas, 'rb'))
    gmm_rastrojo= pickle.load(open(dir_gmm_rastrojo, 'rb'))
    df_patches=pd.DataFrame({"X1":[],"y1":[],"X2":[],"y2":[],"id_patch":[],"label":[]}) 
   
    
    for img in images_dir:
    
            
        # Generar la máscara
       
        
        mask=prediction(img,gmm_malezas,gmm_rastrojo,write_dir_test)
        
        
        
        
        # Aplicar scanning windows generar los patches 
        
        stepSize_y=int(mask.shape[0]/split[0])
        stepSize_X=int(mask.shape[1]/split[1])
        windowSize=[stepSize_y,stepSize_X]
        patches_mask=sliding_window_logico(mask, stepSize_X,stepSize_y, windowSize)
        
        df_cord=pd.DataFrame(patches_mask)  # convierto al generator salido de scanning widows a df
        
        
        df_cord=df_cord.rename(columns={0:"X1",1:"y1",2:"X2",3:"y2"}) # renombro las columns así se llama iguales y se puede concat
        
        # creo mi df para id add todas los patch
        
        #df_patches = pd.concat([df_patches, df_cord], axis=0)# concateno las coord con los índices de los patch
       

        # concateno los patch
        n_patches=df_cord["X1"].count() # creo un objeto que me almacene len del df
        my_list = list(range(0, n_patches)) #genero una lista in range de la la len del df
        df_id_patch=pd.Series(my_list)
        name_img=img.split("/")[-1]
        
         
        ####### aplicamos las comparación y creamos un csv con los resultados #########
        lst_labels=[]
        for row in df_cord.itertuples():
            patch=mask[int(row.y1):int(row.y2),int(row.X1):int(row.X2)] # genero  los patch
            dimesions=patch.shape
            
            size_patch=dimesions[0]*dimesions[1]
            
            n_pixeles_malezas = np.count_nonzero(patch == 255)
            threshold=porcentaje*size_patch/100 
            if n_pixeles_malezas > threshold:
                lst_labels.append(weed_label)
                
            else: 
                lst_labels.append("SOIL")
                     
        df_labels=pd.Series(lst_labels)
        
        df_cord["id_patch"]=df_id_patch
        df_cord["original_image"]=name_img
        df_cord["label"]=df_labels
        df_cord["name_patch"]=str(weed_label)+"_"+df_id_patch.astype(str)+"_"+str(split[0])+"X"+str(split[1])+"_"+str(porcentaje)+"_"+"_"+str(name_img)
        df_patches = pd.concat([df_patches, df_cord], axis=0)
        df_patches.to_csv(path_csv_write)
    return df_patches 


"""

# Variables


porcentaje= 1

split=[4,3]

#path="/Users/Juancho08/Desktop/Barbercho/Dataset_labels/PAIDE_Juan/"
#images_dir=[path+"8.jpg",path+"5.jpg", path+"6.jpg",path+"7.jpg"]
weed_label="PAIDE_Juan"
write_dir_test= '/Users/Juancho08/Desktop/Barbercho/GMM/PAIDE_Romi/Best_model/'
dir_gmm_malezas='/Users/Juancho08/Desktop/Barbercho/GMM/PAIDE_Romi/Best_model/modelos/1M7Stiedmaleza'
dir_gmm_rastrojo= '/Users/Juancho08/Desktop/Barbercho/GMM/PAIDE_Romi/Best_model/modelos/1M7Stiedsuelo'
path_csv_write='/Users/Juancho08/Desktop/Barbercho/GMM/PAIDE_Romi/Best_model/'+weed_label+".csv"

weed_label="PAIDE_Romi"



df = pd.read_csv("/Users/Juancho08/Desktop/Barbercho/CSV/database.csv")
images_dir_1=df.loc[df["Label"] == weed_label] 
images_dir=images_dir_1["path_image"].tolist()

a=self_labeled(porcentaje,split,images_dir,dir_gmm_malezas,dir_gmm_rastrojo,write_dir_test,weed_label,path_csv_write)



"""