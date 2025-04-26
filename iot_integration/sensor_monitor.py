# iot_integration/sensor_monitor.py
import paho.mqtt.client as mqtt
import json
import time

# MQTT Configuration
BROKER = "iot.eclipse.org"
TOPIC = "farm/sensors"

def on_message(client, userdata, message):
    payload = json.loads(message.payload.decode())
    process_sensor_data(payload)

def process_sensor_data(data):
    # Integrate with AI models
    soil_moisture = data['soil_moisture']
    temperature = data['temperature']
    
    # Trigger irrigation if needed
    if soil_moisture < 0.3:
        activate_irrigation()
    
    # Store in database
    store_in_db(data)

client = mqtt.Client()
client.connect(BROKER)
client.subscribe(TOPIC)
client.on_message = on_message
client.loop_forever()