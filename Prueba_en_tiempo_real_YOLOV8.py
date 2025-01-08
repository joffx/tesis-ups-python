import requests
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame
from inference.core.interfaces.stream.sinks import render_boxes
import time
# Función para manejar las predicciones y detectar si hay un arma de fuego
def on_prediction_custom(predictions):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    # Asegurarse de que predictions sea una lista de predicciones
    if isinstance(predictions, dict) and "predictions" in predictions and isinstance(predictions["predictions"], list):
        for prediction in predictions["predictions"]:
            precision = prediction['confidence']  # Precisión de la detección
            print(f"Arma detectada - Tiempo: {current_time}, Precisión: {precision}")
    else:
        print("Predicciones no válidas o vacías.")

# Inicializar la pipeline de inferencia con el callback personalizado y renderización de cuadros delimitadores
pipeline = InferencePipeline.init(
    # model_id="weapon-detection-db7n2/2",
    model_id="detect_firearms/1",
    video_reference=1,
    # on_prediction=on_prediction_custom,
    on_prediction=render_boxes,
    confidence=0.50,
    iou_threshold=0.5,
    max_detections=10,
    mask_decode_mode="accurate",
    tradeoff_factor=0.2,
    active_learning_enabled=False,
    max_fps=30,
)

# Iniciar la pipeline
pipeline.start()
pipeline.join()
