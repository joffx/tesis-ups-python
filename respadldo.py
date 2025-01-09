import csv
import time
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

def on_prediction_custom(predictions, frame):  
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    # Asegurarse de que predictions sea una lista de predicciones
    if isinstance(predictions, dict) and "predictions" in predictions and isinstance(predictions["predictions"], list):
        for prediction in predictions["predictions"]:
            precision = prediction['confidence']  # Precisión de la detección
            # data_to_write.append([current_time, precision])  # Guardar en la lista
            print(f"Arma detectada - Tiempo: {current_time}, Precisión: {precision}")
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
