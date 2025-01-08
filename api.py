import time
import requests
from inference import InferencePipeline

def on_prediction_custom(predictions, frame):  
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    # Asegurarse de que predictions sea una lista de predicciones
    if isinstance(predictions, dict) and "predictions" in predictions and isinstance(predictions["predictions"], list):
        for prediction in predictions["predictions"]:
            precision = prediction['confidence']  # Precisión de la detección
            print(f"Arma detectada - Tiempo: {current_time}, Precisión: {precision}")
            
            # Preparar los datos para enviar a la API
            data = {
                "accuracy": precision,
                "date": current_time
            }
            
            # Enviar la solicitud POST a la API
            response = requests.post("https://coral-app-j5lxf.ondigitalocean.app/api/alerts/createAlert", json=data)
            
            # Verificar el resultado de la solicitud
            # if response.status_code == 200:
            #     print("Datos enviados exitosamente a la API.")
            # else:
            #     print(f"Error al enviar los datos a la API: {response.status_code} - {response.text}")
    else:
        print("Predicciones no válidas o vacías.")

# Inicializar la pipeline de inferencia con el callback personalizado
pipeline = InferencePipeline.init(
    # model_id="weapon-detection-db7n2/2",
    model_id="detect_firearms/3",
    video_reference=0,
    on_prediction=on_prediction_custom,
    confidence=0.45,
    max_detections=10,
    mask_decode_mode="accurate",
    tradeoff_factor=0.2,
    active_learning_enabled=False,
    max_fps=60,
)

# Iniciar la pipeline
pipeline.start()
pipeline.join()

# pip install --upgrade inference