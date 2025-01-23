#include <WiFi.h>
#include <HTTPClient.h>

// Configuración de WiFi
const char* ssid = "REDJOFFRE"; // Reemplaza con el nombre de tu red WiFi
const char* password = "0123456789"; // Reemplaza con la contraseña de tu red WiFi

// URL de la API
const char* serverName = "https://king-prawn-app-okmlu.ondigitalocean.app/api/reports/alertEsp32";

// Pin del zumbador
const int buzzerPin = 23;

void setup() {
  // Inicia el monitor serie
  Serial.begin(115200);
  
  // Configura el pin del zumbador como salida
  pinMode(buzzerPin, OUTPUT);
  
  // Conexión a la red WiFi
  WiFi.begin(ssid, password);
  Serial.print("Conectando a WiFi");
  
  // Espera hasta que se conecte
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println(" Conectado!");
}

void loop() {
  // Comprueba la conexión WiFi antes de enviar la solicitud
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;

    // Especifica la URL del servidor y la ruta del endpoint
    http.begin(serverName);

    // Enviar solicitud GET
    int httpResponseCode = http.GET();

    if (httpResponseCode > 0) {
      String payload = http.getString(); // Obtiene la respuesta del servidor
      Serial.println("Respuesta de la API: " + payload);
      
      if (payload == "2") {
        // Enciende el zumbador por 5 segundos
        digitalWrite(buzzerPin, HIGH);
        delay(10000);
        digitalWrite(buzzerPin, LOW);
        // Espera 5 segundos antes de continuar
        delay(60000);
      }
    } else {
      Serial.println("Error en la solicitud HTTP: " + String(httpResponseCode));
    }
    http.end(); // Finaliza la conexión
  } else {
    Serial.println("Error de conexión WiFi");
  }

  // Espera 5 segundos antes de la próxima consulta si no ha sonado el zumbador
  delay(5000);
}
