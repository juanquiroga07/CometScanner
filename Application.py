import os
import sys
import cv2
import pandas as pd
import albumentations as album
import torch
import numpy as np
import tifffile as tiff
import segmentation_models_pytorch as smp
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsDropShadowEffect, QFileDialog, QLabel, QVBoxLayout, QWidget, QInputDialog, QMenu, QAction, QProgressBar, QMessageBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QMetaObject
from PyQt5.QtGui import QColor, QPixmap, QImage, QMovie
from PyQt5 import QtWidgets
from PyQt5.uic import loadUi
from hmax import FastHybridDome
from collections import deque
from model import ModeloUnet
from io import BytesIO

class BuildingsDataset(torch.utils.data.Dataset):
    
    """Massachusetts Buildings Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_rgb_values (list): RGB values of select classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    def __init__(
            self,
            image_path,
            class_rgb_values=None,
            augmentation=None,
            preprocessing=None,
    ):

        self.image_path = image_path
        self.class_rgb_values = class_rgb_values
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):

        
        image = cv2.cvtColor(cv2.imread(self.image_path), cv2.COLOR_BGR2RGB)


        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image)
            image = sample['image']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image)
            image = sample['image']

        return image

    def __len__(self):
        # return length of
        return 1


# Set device: `cuda` or `cpu`
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar el modelo preentrenado
# Crear una instancia de la clase ModeloUnet
modelo_unet_instancia = ModeloUnet(ENCODER='resnet50', ENCODER_WEIGHTS='imagenet', ACTIVATION='sigmoid')

ENCODER='resnet50'
ENCODER_WEIGHTS='imagenet'
ACTIVATION='sigmoid'

# Obtener las clases de la instancia
CLASSES = modelo_unet_instancia.obtener_clases()

CLASSES_RGB = modelo_unet_instancia.class_rgb_values

# Cargar el modelo preentrenado
ruta_checkpoint = "/home/omicas/Documentos/Juanes/Entrenamiento/INTERFAZ/best_model_checkpoint.pth"
modelo_checkpoint = torch.load(ruta_checkpoint, map_location=torch.device('cpu'))
modelo_unet_instancia.modelo_segmentacion.load_state_dict(modelo_checkpoint['model_state_dict'])
modelo_unet_instancia.modelo_segmentacion.eval()

# Perform reverse one-hot-encoding on labels / preds
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image

    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified
        class key.
    """
    x = np.argmax(image, axis = -1)
    return x

# Perform colour coding on the reverse-one-hot outputs
def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x

# Funciones de transformación
def get_validation_augmentation():
    # Add sufficient padding to ensure image is divisible by 32
    test_transform = [
        album.PadIfNeeded(min_height=1536, min_width=1536, always_apply=True, border_mode=0),
        album.augmentations.geometric.resize.LongestMaxSize(max_size=256, interpolation=1, always_apply=True, p=1), # make images smaller

    ]
    return album.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callable): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose"""
    
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor))

    return album.Compose(_transform)

preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

class ImageProcessingThread(QThread):
    processed_image = pyqtSignal(np.ndarray)
    progress_updated = pyqtSignal(int)

    def __init__(self):
        super(ImageProcessingThread, self).__init__()
        self.image = None
        self.fast_hybrid_dome = None

    def set_fast_hybrid_dome(self, fast_hybrid_dome):
        self.fast_hybrid_dome = fast_hybrid_dome

    def set_image(self, imagen):
        self.image = imagen

    def run(self):
        try:
            # Supongamos que tienes una función batch_process que procesa la imagen por lotes
            batch_size = 10  # Ajusta esto según tu implementación
            total_batches = len(self.image) // batch_size

            # Inicializar result antes del bucle
            result = []

            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = (batch_idx + 1) * batch_size

                # Procesar el lote actual
                batch_result = self.fast_hybrid_dome.run(self.image[start_idx:end_idx])

                # Añadir el resultado del lote actual a la lista
                result.append(batch_result)

                # Emitir la señal de progreso después de cada lote
                progress_value = int((batch_idx + 1) / total_batches * 100)
                self.progress_updated.emit(progress_value)

                # Permitir que la interfaz gráfica se actualice
                QApplication.processEvents()

            # Concatenar las imágenes procesadas
            final_result = np.concatenate(result, axis=0)

            # Emitir la señal con la imagen procesada final
            self.processed_image.emit(final_result)

            # Emitir la señal de progreso al 100%
            self.progress_updated.emit(100)

        except Exception as e:
            print(f"Error en el hilo de procesamiento: {str(e)}")


class ImageViewer(QWidget):
    def __init__(self, label_3):
        super(ImageViewer, self).__init__()

        self.image_label = QLabel()
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        self.setLayout(layout)

        self.image = None

        self.label_3 = label_3 

    def show_image(self, img):
        if len(img.shape) == 2:
            # La imagen es en escala de grises
            height, width = img.shape
            bytes_per_line = width
        elif len(img.shape) == 3:
            # La imagen es en color
            height, width, _ = img.shape
            bytes_per_line = 3 * width
        else:
            raise ValueError("La imagen tiene una forma inesperada")

        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)

        # Resto del código para mostrar la imagen...

        pixmap = QPixmap.fromImage(q_img)

        # Obtener el tamaño máximo deseado (puedes ajustar estos valores según tus necesidades)
        max_width = 900
        max_height = 850

        # Escalar la imagen si excede el tamaño máximo
        if pixmap.width() > max_width or pixmap.height() > max_height:
            pixmap = pixmap.scaled(max_width, max_height, Qt.KeepAspectRatio)

        # Establecer el QPixmap en el QLabel label_3
        self.label_3.setPixmap(pixmap)
        self.label_3.setAlignment(Qt.AlignCenter)


class VentanaPrincipal(QMainWindow):
    def __init__(self):
        super(VentanaPrincipal, self).__init__()
        loadUi('COMETS.ui', self)

        self.fileName = None
        self.test_dataset = None
        self.nombre_imagen_sin_extension = None
        self.datos_cometas = None
        self.imagen = None
        self.processed_image = None
        self.result = None

        # Buscar la barra de progreso en el diseño de la interfaz
        self.progress_bar = self.findChild(QProgressBar, 'progressBar')  # Reemplaza 'nombre_de_tu_barra_de_progreso'

        # sombra de los widgets
        self.sombra_frame(self.stackedWidget)
        self.sombra_frame(self.frame_superior)
        self.sombra_frame(self.bt_uno)
        self.sombra_frame(self.bt_dos)
        self.sombra_frame(self.bt_tres)
        self.sombra_frame(self.bt_cuatro)
        
        # control barra de titulos
        self.bt_minimizar.clicked.connect(self.showMinimized)
        self.bt_maximizar.clicked.connect(self.toggle_maximized)
        self.bt_cerrar.clicked.connect(self.close)

        # eliminar barra y de titulo - opacidad
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setWindowOpacity(1)

        # SizeGrip
        self.gripSize = 10
        self.grip = QtWidgets.QSizeGrip(self)
        self.grip.resize(self.gripSize, self.gripSize)

        # mover ventana
        self.frame_superior.mouseMoveEvent = self.mover_ventana

        # acceder a las paginas
        self.bt_uno.clicked.connect(self.pagina_uno)
        self.bt_dos.clicked.connect(self.pagina_dos)
        self.bt_tres.clicked.connect(self.pagina_tres)
        self.bt_cuatro.clicked.connect(self.pagina_cuatro)

        # Conecta el botón bt_uno a un nuevo método
        self.bt_uno.clicked.connect(self.abrir_dialogo_imagen)

        # Conecta el botón bt_tres al método procesar_imagen
        self.bt_dos.clicked.connect(self.procesar_imagen)

        # Conecta el botón bt_dos al método seleccionar_carpeta_destino
        self.bt_tres.clicked.connect(self.seleccionar_carpeta_destino)

        self.fast_hybrid_dome = None

        self.image_processing_thread = ImageProcessingThread()

        # Establecer la imagen
        self.image_processing_thread.set_image(np.zeros((1, 1)))
        
        # Crea una nueva instancia de ImageViewer (puede ser None inicialmente)
        self.viewer = ImageViewer(np.zeros((1, 1)))

        self.label_3 = self.findChild(QLabel, 'label_3')  # Asigna el QLabel label_3 como un atributo

        # Crear una nueva instancia de ImageViewer
        self.viewer = ImageViewer(self.label_3)

        # Crear una nueva instancia de ImageViewer para la página uno
        self.viewer_pagina_uno = ImageViewer(self.label_2)

        # Crear una nueva instancia de ImageViewer para la página dos
        self.viewer_pagina_tres = ImageViewer(self.label_4)

        # Crear una nueva instancia de ImageViewer para la página tres
        self.viewer_pagina_cuatro = ImageViewer(self.label_5)

        # Llamar a la función para mostrar la página uno y el GIF al inicio
        self.pagina_uno()

        # Conecta directamente al método show_image de la instancia viewer
        self.image_processing_thread.processed_image.connect(self.viewer.show_image)

        self.image_processing_thread.progress_updated.connect(self.update_progress_bar)

        self.image_processing_thread.processed_image.connect(self.handle_processed_image)

        self.image_processing_thread.finished.connect(self.image_processing_finished)  # Conectar la señal finished  # Conectar la señal finished

        # Conectar el método actualizar_tamanos_gif al evento resizeEvent
        self.resizeEvent = lambda event: self.actualizar_tamanos_gif()

    def actualizar_tamanos_gif(self):
        # Actualizar el tamaño del QLabel de la página uno
        self.label_2.setFixedSize(self.width(), self.height())

        # Actualizar el tamaño del QLabel de la página tres
        self.label_4.setFixedSize(self.width(), self.height())

        # Actualizar el tamaño del QLabel de la página cuatro
        self.label_5.setFixedSize(self.width(), self.height())

    def update_progress_bar(self, progress):
        self.progressBar.setValue(progress)  # Actualizar el valor de la barra de progreso

    def toggle_maximized(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    # creamos el metodo sombra
    def sombra_frame(self, frame):
        sombra = QGraphicsDropShadowEffect(self)
        sombra.setBlurRadius(30)
        sombra.setXOffset(8)
        sombra.setYOffset(8)
        sombra.setColor(QColor(0, 0, 0, 255))
        frame.setGraphicsEffect(sombra)

    # SizeGrip
    def resizeEvent(self, event):
        rect = self.rect()
        self.grip.move(rect.right() - self.gripSize, rect.bottom() - self.gripSize)

    # mover ventana
    def mousePressEvent(self, event):
        if not self.isMaximized():
            self.clickPosition = event.globalPos()
        else:
            self.clickPosition = event.pos()

    def mover_ventana(self, event):
        if not self.isMaximized():
            if event.buttons() == Qt.LeftButton:
                self.move(self.pos() + event.globalPos() - self.clickPosition)
                self.clickPosition = event.globalPos()
                event.accept()
        else:
            super().mouseMoveEvent(event)
    
    # Animacion de paginas
    def pagina_uno(self):
        self.stackedWidget.setCurrentWidget(self.page_uno)
        #self.animacion_paginas()
        self.label_2 = self.findChild(QLabel, 'label_2')

        # Obtener la ruta del GIF desde la propiedad "toolTip" del QLabel
        # Especificar directamente la ruta del GIF
        gif_path = "comet.gif"

        if gif_path:
            # Crear un objeto QMovie y asignarlo al QLabel
            self.gif_movie = QMovie(gif_path)
            self.label_2.setMovie(self.gif_movie)

            # Iniciar el QMovie
            self.gif_movie.start()
        
        #self.label_2.show()

    def pagina_dos(self):
        self.stackedWidget.setCurrentWidget(self.page_dos)
        #self.animacion_paginas()

    def pagina_tres(self):
        self.stackedWidget.setCurrentWidget(self.page_tres)
        #self.animacion_paginas()
        self.label_4 = self.findChild(QLabel, 'label_4')

        # Obtener la ruta del GIF desde la propiedad "toolTip" del QLabel
        # Especificar directamente la ruta del GIF
        gif_path_2 = "cells.gif"

        if gif_path_2:
            # Crear un objeto QMovie y asignarlo al QLabel
            self.gif_movie_2 = QMovie(gif_path_2)
            self.label_4.setMovie(self.gif_movie_2)

            # Iniciar el QMovie
            self.gif_movie_2.start()

    def pagina_cuatro(self):
        # Cambiar la página en self.stackedWidget
        self.stackedWidget.setCurrentWidget(self.page_cuatro)

        # Obtener el botón que activó la acción
        button = self.bt_cuatro

        # Crear y mostrar el menú alineado al botón
        menu = self.crear_menu_videos()
        menu.exec_(button.mapToGlobal(button.rect().bottomLeft())) 

        #self.animacion_paginas()
        self.label_5 = self.findChild(QLabel, 'label_5')

        # Obtener la ruta del GIF desde la propiedad "toolTip" del QLabel
        # Especificar directamente la ruta del GIF
        gif_path_3 = "videos.gif"

        if gif_path_3:
            # Crear un objeto QMovie y asignarlo al QLabel
            self.gif_movie_3 = QMovie(gif_path_3)
            self.label_5.setMovie(self.gif_movie_3)

            # Iniciar el QMovie
            self.gif_movie_3.start()
    
    def crear_menu_videos(self):
        menu_videos = QMenu(self)

        # Agrega acciones (videos) al menú
        video1_action = QAction("Video 1", self)
        video1_action.triggered.connect(lambda: self.abrir_video("https://www.youtube.com/watch?v=O202APsAjzw"))
        menu_videos.addAction(video1_action)

        video2_action = QAction("Video 2", self)
        video2_action.triggered.connect(lambda: self.abrir_video("https://www.youtube.com/watch?v=oNP5Xe4aYWk"))
        menu_videos.addAction(video2_action)

        video3_action = QAction("Video 3", self)
        video3_action.triggered.connect(lambda: self.abrir_video("https://www.youtube.com/watch?v=jFKNGZPRnSg&t=396s"))
        menu_videos.addAction(video3_action)

        # Agrega más videos según sea necesario...

        return menu_videos
    
    def abrir_video(self, video_url):
        # Aquí puedes abrir el enlace del video. Podrías usar algún navegador web o tu reproductor de videos favorito.
        # Por ejemplo, para abrirlo en el navegador predeterminado:
        import webbrowser
        webbrowser.open(video_url)

    def abrir_dialogo_imagen(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        fileName, _ = QFileDialog.getOpenFileName(self, "Select image", "", "Images (*.png *.jpg *.bmp *.jpeg *.tif);;All Files (*)", options=options)
        
        if fileName:
        
            self.fileName = fileName  # Almacena fileName como un atributo de instancia
            #self.bt_dos.setEnabled(True)  # Habilita el botón "Run"
            imagen = cv2.imread(self.fileName)
           
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

            self.imagen = imagen

            # Obtener valor de hdomo
            hdomo, ok = QInputDialog.getInt(self, "Adjust hdomo", "Enter the value of hdomo:", value=200, min=0, max=255)
            if ok:
                self.fast_hybrid_dome = FastHybridDome(dome_height=hdomo)

                self.image_processing_thread.image = imagen
                self.image_processing_thread.set_fast_hybrid_dome(self.fast_hybrid_dome)
                self.progressBar.setValue(0)  # Reiniciar la barra de progreso
                self.image_processing_thread.start()

                self.viewer.show_image(imagen)

                 # Crear la instancia de BuildingsDataset aquí, después de que self.fileName se haya establecido
                test_dataset = BuildingsDataset(
                    image_path=self.fileName,
                    augmentation=get_validation_augmentation(),
                    preprocessing=get_preprocessing(preprocessing_fn),
                    class_rgb_values=CLASSES_RGB,
                )

                for idx in range(len(test_dataset)):
                    image = test_dataset[idx]
                    
                    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
                    
                    # Asegurarse de que los pesos del modelo y la entrada estén en el mismo dispositivo
                    modelo_unet_instancia.to(DEVICE)
                    x_tensor = x_tensor.to(DEVICE).to(torch.float)

                    # Realizar la predicción
                    with torch.no_grad():
                        mascara_predicha = modelo_unet_instancia.modelo_segmentacion(x_tensor)

                    mascara_predicha = mascara_predicha.detach().squeeze().cpu().numpy()
                    mascara_predicha = np.transpose(mascara_predicha, (1, 2, 0))
                    mascara_predicha = colour_code_segmentation(reverse_one_hot(mascara_predicha), CLASSES_RGB)

                # Convertir a arreglos NumPy
                image_np = np.array(image)
                mascara_predicha_np = np.array(mascara_predicha)

                # Escalar los valores a 8 bits (0-255)
                image_np_scaled = ((image_np - image_np.min()) / (image_np.max() - image_np.min()) * 255).astype(np.uint8)
                mascara_predicha_np_scaled = ((mascara_predicha_np - mascara_predicha_np.min()) / (mascara_predicha_np.max() - mascara_predicha_np.min()) * 255).astype(np.uint8)

                # Convertir la imagen a formato TIFF en memoria
                with BytesIO() as imagen_tiff_bytes:
                    tiff.imwrite(imagen_tiff_bytes, image_np_scaled)
                    imagen_tiff_bytes.seek(0)  # Rebobinar el puntero de bytes al principio
                    imagen_tiff = tiff.imread(imagen_tiff_bytes)

                # Convertir la máscara a formato TIFF en memoria
                with BytesIO() as mascara_tiff_bytes:
                    tiff.imwrite(mascara_tiff_bytes, mascara_predicha_np_scaled)
                    mascara_tiff_bytes.seek(0)  # Rebobinar el puntero de bytes al principio
                    mascara_tiff = tiff.imread(mascara_tiff_bytes)
                    self.mascara_tiff = mascara_tiff
       
    def handle_processed_image(self, result):

        # Actualiza la imagen en la instancia de ImageViewer
        self.viewer.show_image(result)
        self.result = result
        
        # Actualizar la barra de progreso
        self.progressBar.setValue(100)
    
    def image_processing_finished(self):
        
    # Muestra un mensaje al usuario
    # Puedes usar QMessageBox para mostrar un cuadro de diálogo con un mensaje
        QMessageBox.information(self, "Completed process", "The h-max technique was successfully applied. Please press the 'RUN' button.")
    
    def procesar_imagen(self):

        # Verifica si se ha importado una imagen antes de procesar
        if self.fileName is None:
            QMessageBox.warning(self, "Error", "Please import an image before running the process.")
            return

        # Inicializar listas para almacenar datos de los cometas
        numeros_cometas = []
        area_total = []
        area_cabeza = []
        area_cola = []
        intensidad_cometa = []
        intensidad_cabeza = []
        intensidad_cola = []
        porcentaje_daño_cometa = []
        tipos_cometas = []

        # Lista de colores para asignar a cada tipo de cometa en formato BGR
        colores_cometas = {
            'No Damage': (255, 255, 255),         # Blanco en RGB
            'Slight': (0, 255, 255),          # Celeste en RGB
            'Moderate': (255, 255, 0),      # Amarillo en RGB
            'Serious': (255, 102, 0),         # Naranja en RGB
            'Critical': (255, 0, 0),         # Rojo en RGB
        }
        # Convertir imagen a color si es necesario
        if len(self.result.shape) == 2:
            self.result = cv2.cvtColor(self.result, cv2.COLOR_GRAY2BGR)
        
        # Redimensionar las imágenes al mismo tamaño si no tienen las mismas dimensiones
        if self.result.shape[:2] != self.mascara_tiff.shape:
            self.mascara_tiff = cv2.resize(self.mascara_tiff, (self.result.shape[1], self.result.shape[0]))

        # Etiquetar los cometas y contarlos
        _, etiquetas = cv2.connectedComponents(self.mascara_tiff.astype('uint8'))
        numero_cometas = etiquetas.max()
        

        for num_cometa in range(1, numero_cometas + 1):
            cometa_mascara = (etiquetas == num_cometa).astype('uint8')
        
            
            area_cabeza_cometa = cv2.countNonZero(cometa_mascara * (self.mascara_tiff == 255))
            area_cola_cometa = cv2.countNonZero(cometa_mascara * (self.mascara_tiff == 60))
            intensidad_cabeza_cometa = self.result[cometa_mascara * (self.mascara_tiff == 255) > 0].sum()
            intensidad_cola_cometa = self.result[cometa_mascara * (self.mascara_tiff == 60) > 0].sum()

            
            # La intensidad total del cometa es la suma de la intensidad de la cabeza y la cola
            intensidad_total_cometa = intensidad_cabeza_cometa + intensidad_cola_cometa
            area_total_cometa = area_cabeza_cometa + area_cola_cometa
            
            # Calcular el porcentaje de intensidad de la cola respecto a la intensidad total
            porcentaje_intensidad_cola = (intensidad_cola_cometa / intensidad_total_cometa) * 100

            # Redondear el porcentaje a dos decimales
            porcentaje_intensidad_cola = round(porcentaje_intensidad_cola, 2)
            
            # Clasificar cometas según el porcentaje de intensidad de la cola
            if porcentaje_intensidad_cola < 5:
                tipo_cometa = 'No Damage'
            elif 5 <= porcentaje_intensidad_cola < 30:
                tipo_cometa = 'Slight'
            elif 30 <= porcentaje_intensidad_cola < 60:
                tipo_cometa = 'Moderate'
            elif 60 <= porcentaje_intensidad_cola < 80:
                tipo_cometa = 'Serious'
            elif 80 <= porcentaje_intensidad_cola <= 100:
                tipo_cometa = 'Critical'

            # Agregar datos a las listas
            numeros_cometas.append(num_cometa)
            area_total.append(area_total_cometa)
            area_cabeza.append(area_cabeza_cometa)
            area_cola.append(area_cola_cometa)
            intensidad_cometa.append(intensidad_total_cometa)
            intensidad_cabeza.append(intensidad_cabeza_cometa)
            intensidad_cola.append(intensidad_cola_cometa)
            porcentaje_daño_cometa.append(porcentaje_intensidad_cola)
            tipos_cometas.append(tipo_cometa)
        

            # Dibujar contornos en la imagen original con el color asignado al tipo de cometa
            color_cometa = colores_cometas[tipo_cometa]
            contornos, _ = cv2.findContours(cometa_mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(self.result, contornos, -1, color_cometa, 2)
            
            # Obtener el centroide del cometa
            M = cv2.moments(cometa_mascara)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Dibujar el número del cometa en blanco
                cv2.putText(self.result, str(num_cometa), (cX - 10, cY - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Obtener el nombre de la imagen sin extensión
        nombre_imagen_sin_extension = os.path.splitext(os.path.basename(self.fileName))[0]
        self.nombre_imagen_sin_extension = nombre_imagen_sin_extension

        # Crear un DataFrame con los datos de los cometas
        datos_cometas = pd.DataFrame({
            'COMET NUMBER': numeros_cometas,
            'COMET AREA': area_total,
            'HEAD AREA': area_cabeza,
            'TAIL AREA': area_cola,
            'COMET INTENSITY': intensidad_cometa,
            'HEAD INTENSITY': intensidad_cabeza,
            'TAIL INTENSITY': intensidad_cola,
            'PERCENTAGE OF DNA DAMAGE (%)': porcentaje_daño_cometa,
            'TYPE OF DAMAGE': tipos_cometas
        })
        
        # Actualiza la imagen en el QLabel label_3
        pixmap = QPixmap.fromImage(QImage(self.result.data, self.result.shape[1], self.result.shape[0], self.result.shape[1] * 3, QImage.Format_RGB888))
        self.label_3.setPixmap(pixmap)

        # Actualiza la imagen en la instancia de ImageViewer
        self.viewer.show_image(self.result)

        self.datos_cometas = datos_cometas

        QMessageBox.information(self, "Results generated", "Select the folder where you want them to be saved using the 'OUTPUT DIRECTORY' button.")

    def seleccionar_carpeta_destino(self):

        # Verifica si se ha importado una imagen antes de seleccionar la carpeta de destino
        if self.fileName is None:
            QMessageBox.warning(self, "Error", "Please import an image before selecting the results folder.")
            return

        if self.imagen is not None:
            # Preguntar al usuario la carpeta de destino
            carpeta_resultados = QFileDialog.getExistingDirectory(self, "Select results folder")

            if carpeta_resultados:

                # Guardar la imagen con los cometas etiquetados en la carpeta seleccionada
                # Crear el nombre del archivo Excel usando el nombre de la imagen
                nombre_excel = f"Comet_results_{self.nombre_imagen_sin_extension}.xlsx"
                ruta_excel = os.path.join(carpeta_resultados, nombre_excel)
                nombre_imagen_etiquetada = f"Comets_{self.nombre_imagen_sin_extension}.tif"
                ruta_imagen_etiquetada = os.path.join(carpeta_resultados, nombre_imagen_etiquetada)
                tiff.imwrite(ruta_imagen_etiquetada, self.result)



                with pd.ExcelWriter(ruta_excel, engine='xlsxwriter') as writer:
                    self.datos_cometas.to_excel(writer, index=False, sheet_name='Cometas')

                # Imprimir mensaje de finalización
                
                QMessageBox.information(self, "Saved results", f"The results have been saved in: {carpeta_resultados}.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mi_app = VentanaPrincipal()
    mi_app.show()
    sys.exit(app.exec_())