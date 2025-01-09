import csv
import time
import cv2
import os
import numpy as np
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame  # Asegúrate de importar VideoFrame
import requests
from datetime import datetime

def write_to_csv(data):
    with open('deteccion_arma.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Tiempo', 'Precisión'])
        for row in data:
            writer.writerow(row)

# Almacena los datos en una lista en lugar de escribir directamente en el archivo
data_to_write = []

# Asegúrate de tener una carpeta para las imágenes
os.makedirs('imagenes_detectadas', exist_ok=True)

# Variable para controlar el tiempo entre detecciones
last_detection_time = 0
detection_cooldown = 5  # Tiempo de espera en segundos entre detecciones (ajusta según sea necesario)

def on_prediction_custom(predictions, frame):
    global last_detection_time
    current_time = time.time()  # Obtiene el tiempo actual en segundos
    
    print(f"Tipo de frame: {type(frame)}")
    
    if isinstance(predictions, dict) and "predictions" in predictions and isinstance(predictions["predictions"], list):
        for prediction in predictions["predictions"]:
            precision = prediction['confidence']
            print(f"Arma detectada - Tiempo: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}, Precisión: {precision}")
            
            # Verificar si ha pasado el tiempo suficiente desde la última detección
            if current_time - last_detection_time < detection_cooldown:
                print("Detección demasiado reciente, omitiendo solicitud.")
                return  # Si no ha pasado el tiempo suficiente, omite el envío

            last_detection_time = current_time  # Actualiza el tiempo de la última detección
            
            # Intentar acceder directamente a los datos del frame
            try:
                frame_data = frame.data  # Suponiendo que el atributo 'data' contiene el array de la imagen
            except AttributeError:
                try:
                    frame_data = frame.image  # Alternativa si 'image' es el atributo correcto
                except AttributeError:
                    print("No se pudo acceder a los datos de la imagen desde el objeto VideoFrame.")
                    return

            if isinstance(frame_data, np.ndarray):
                current_time_str = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
                image_filename = f"imagenes_detectadas/deteccion_{current_time_str}.jpg"
                cv2.imwrite(image_filename, frame_data)  # Guardar la imagen correctamente en formato JPG
                
                # Verificar que el archivo se guardó correctamente antes de enviarlo
                if os.path.exists(image_filename):
                    with open(image_filename, 'rb') as image_file:
                        headers = {
                            # Asegúrate de enviar cabeceras si es necesario
                        }
                        data = {
                            'precision': precision,
                            'object': 'SI',
                        }
                        files = {'file': (image_filename, image_file, 'image/jpeg')}  # Especifica el tipo MIME correcto
                        
                        response = requests.post('https://tesis-ups-cb4uz.ondigitalocean.app/api/reports/upload', files=files, data=data, headers=headers)
                        if response.status_code == 200:
                            print("Archivo subido exitosamente.")
                        else:
                            print(f"Error al subir el archivo: {response.status_code} - {response.text}")
                else:
                    print("No se pudo guardar la imagen correctamente.")
            else:
                print("Los datos del frame no son un array de NumPy, no se puede guardar la imagen.")
    else:
        print("Predicciones no válidas o vacías.")

# Inicializar la pipeline de inferencia con el callback personalizado
pipeline = InferencePipeline.init(
    model_id="detect_firearms/1",
    video_reference=0,
    on_prediction=on_prediction_custom,  
    confidence=0.75,
    iou_threshold=0.5,
    max_detections=10,
    mask_decode_mode="accurate",
    tradeoff_factor=0.2,
    active_learning_enabled=False,
    max_fps=30,
)

try:
    # Iniciar la pipeline
    pipeline.start()
    pipeline.join()
finally:
    # Asegurarse de que los datos se escriban en el archivo CSV después de que la pipeline termine
    write_to_csv(data_to_write)