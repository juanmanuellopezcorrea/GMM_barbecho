#!/usr/bin/env python3# -*- coding: utf-8 -*-"""Created on Thu May  7 11:56:25 2020@author: Juancho08"""import cv2import osimport globimport numpy as npfrom sklearn.mixture import GaussianMixture as GMMimport matplotlib.pyplot as pltclass GMM_malezas:        def _init_ (self,dir_original,dir_mask,dir_img):        self.dir_original=dir_original        self.dir_mask=dir_mask        self.dir_img=dir_img            def trainingGMM_rastrojo_maleza(dir_orignal_img, dir_manual_mask,n_component_malezas,n_component_rastrojo,covarianza):                 data_paht_original=os.path.join(dir_orignal_img,'*g')        files_original=glob.glob(data_paht_original)        data_original=[]        for f1 in files_original:            img=cv2.imread(f1)            imgLUV=cv2.cvtColor(img, cv2.COLOR_BGR2LUV) # Trasformamos la imágen al espacio de color  LUV            imgUV=imgLUV[:,:,1:3]  # le quitamos el L dejando solo UV            data_original.append(imgUV)        # Importamos las mascaras para entrenar al GMM        data_paht_mascara=os.path.join(dir_manual_mask,'*g')        files_mascara=glob.glob(data_paht_mascara)        data_mascara=[]        for f1 in files_mascara:            img=cv2.imread(f1)# intentar leer en gris para disminuir el costo computacional            img[np.where(img>220)]=255            img[np.where(img<20)]=0            data_mascara.append(img)                #Extraemos los pixeles de las malezas,del rastrojo, e indeseados                lst_px_maleza=[]        lst_px_suelo=[]        lst_px_indeseados=[]        color_maleza=[255,255,255]        color_suelo=[0,0,0]                for (img_original,img_mask) in zip(data_original,data_mascara):                        for (fila_imagen,fila_mask) in zip(img_original,img_mask):                                for (px_imagen, px_mask) in zip(fila_imagen,fila_mask):                    if np.array_equal(px_mask,color_maleza):                        lst_px_maleza.append(np.array(px_imagen))                                              elif np.array_equal(px_mask,color_suelo):                        lst_px_suelo.append(np.array(px_imagen))                    else:                        lst_px_indeseados.append(np.array(px_imagen))                                        # Entrenamos el GMM para malezas        gmm_model_maleza = GMM (n_components=n_component_malezas,covariance_type=covarianza).fit(lst_px_maleza)        # Entrenamos el GMM para suelo y guardamos        gmm_model_suelo = GMM (n_components=n_component_rastrojo,covariance_type=covarianza).fit(lst_px_suelo)        # Graficar               UM=[]        VM=[]            for i in lst_px_maleza:            UM.append(i[:1])            VM.append(i[1:])                    US=[]        VS=[]            for i in lst_px_suelo:            US.append(i[:1])            VS.append(i[1:])                plt.scatter(UM,VM,s=50,c='r')         plt.scatter(US,VS,s=15,c='b')         plt.title("Ditribucion espacial de pixeles en el espacio de color UV")        plt.xlabel("U")        plt.ylabel("V")                 #plt.axis([40, 160, 0, 0.03]) # la amplitud del eje x e y        plt.grid(True)           	# Almacena el modelo	    	        return gmm_model_maleza,gmm_model_suelo,plt                    def prediction(img_dir_test,gmm_malezas,gmm_rastrojo,write_dir_test):                   import cv2        import os        import glob        import numpy as np                data_paht_test=os.path.join(img_dir_test,'*g')        files_original=glob.glob(data_paht_test)        data_test=[]        count=0        for f1 in files_original:            img=cv2.imread(f1)            imgLUV=cv2.cvtColor(img, cv2.COLOR_BGR2LUV) # Trasformamos la imágen al espacio de color  LUV            imgUV=imgLUV[:,:,1:3]  # le quitamos el L dejando solo UV            imgUV=imgUV.reshape(((-1,2)))            data_test.append(imgUV)            # Obtenemos  la prob de cada pixel a pertenecer a cada componente del GMM  para cada modelo                 prob_suelo=gmm_rastrojo.predict_proba(imgUV)            prob_maleza=gmm_malezas.predict_proba(imgUV)            # Compara las probabilidades de cada pixel para cada componente y para cada modelo.            df_suelo= np.amax(prob_suelo,1 ) # elige  la mayor prob en la fila            df_maleza= np.amax(prob_maleza,1)            idclases=df_maleza>df_suelo # En cada pixel compara con si df malezas es más grande que df suelo pone un True si no un False            idmask=np.zeros(idclases.shape) #crea una matriz de 0            idmask[np.where(idclases==True)]=0 # Donde hay un 0 en la matrix lo reemplaza por 0 si en el mismo pixel de la matriz idclasse tiene true            idmask[np.where(idclases==False)]=255            # Reconstruimos la imagen             original_shape=img.shape            segmented=idmask.reshape(original_shape[0],original_shape[1])            count=count+1            cv2.imwrite(write_dir_test+str(count)+'.png',segmented)                    print("Predición Gaussian Mixtures Malezas finalizada")                  