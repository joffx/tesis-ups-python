import csv
import time
import cv2
import os
import numpy as np
from inference import get_model
import supervision as sv
from inference import InferencePipeline

# Crear o abrir el archivo CSV para guardar los datos
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

def on_prediction_custom(predictions, frame):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    print(f"Tipo de frame: {type(frame)}")
    
    if isinstance(predictions, dict) and "predictions" in predictions and isinstance(predictions["predictions"], list):
        try:
            frame_data = frame.data  # Suponiendo que el atributo 'data' contiene el array de la imagen
        except AttributeError:
            try:
                frame_data = frame.image  # Alternativa si 'image' es el atributo correcto
            except AttributeError:
                print("No se pudo acceder a los datos de la imagen desde el objeto VideoFrame.")
                return

        if isinstance(frame_data, np.ndarray):
            # Cargar el modelo preentrenado
            model = get_model(model_id="detect_firearms/1")

            # Ejecutar inferencia sobre la imagen
            results = model.infer(frame_data)[0]

            # Cargar los resultados en la API de Detecciones de Supervision
            detections = sv.Detections.from_inference(results)

            # Crear los anotadores de Supervision
            bounding_box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()

            # Anotar la imagen con los resultados de inferencia
            annotated_image = bounding_box_annotator.annotate(
                scene=frame_data, detections=detections)
            annotated_image = label_annotator.annotate(
                scene=annotated_image, detections=detections)

            # Guardar la imagen anotada
            image_filename = f"imagenes_detectadas/deteccion_{current_time.replace(':', '-')}.jpg"
            cv2.imwrite(image_filename, annotated_image)

            # Registrar la detección en el archivo CSV
            for detection in detections:
                precision = detection.confidence
                data_to_write.append([current_time, precision])
                
            # Mostrar la imagen con las anotaciones
            sv.plot_image(annotated_image)
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
