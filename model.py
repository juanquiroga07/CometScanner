import pandas as pd
import segmentation_models_pytorch as smp
import numpy as np
import torch.nn as nn

class ModeloUnet(nn.Module):

    def __init__(self, ENCODER='resnet50', ENCODER_WEIGHTS='imagenet', ACTIVATION='sigmoid'):

        super(ModeloUnet, self).__init__()  # Llama al constructor de la clase base

        self.ENCODER = ENCODER
        self.ENCODER_WEIGHTS = ENCODER_WEIGHTS
        self.ACTIVATION = ACTIVATION

        # Cargar el archivo CSV con la información de las clases
        class_dict = pd.read_csv("/home/omicas/Documentos/Juanes/Entrenamiento/INTERFAZ/label_class_dict.csv")

        # Obtener nombres de las clases y valores RGB
        self.class_names = class_dict['name'].tolist()

        self.class_rgb_values = class_dict[['r']].values.tolist()

        # Get RGB values of required classes
        select_class_indices = [self.class_names.index(cls.lower()) for cls in self.class_names]
        select_class_rgb_values =  np.array(self.class_rgb_values)[select_class_indices]

        self.class_rgb_values = select_class_rgb_values

        # Configuración del modelo
        self.CLASSES = self.class_names  # Utilizar las clases obtenidas del CSV

        # Cargar el modelo U-Net++
        self.modelo_segmentacion = smp.UnetPlusPlus(
            self.ENCODER,
            encoder_weights=self.ENCODER_WEIGHTS,
            classes=len(self.CLASSES),
            activation=self.ACTIVATION,
        )

    def obtener_clases(self):
        return self.class_names
    