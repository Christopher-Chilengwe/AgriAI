# iot_integration/sensor_monitor.py
import paho.mqtt.client as mqtt
import json
import time
import logging
import os
from typing import Dict, Any
from datetime import datetime
from configparser import ConfigParser
from database import DBConnector  # Custom database module
from ai_integration import YieldPredictor  # Custom AI module
import ssl
import threading
import schedule
from queue import Queue
from retry import retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(threadName)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IoTMonitor:
    def __init__(self, config_path: str = 'config.ini'):
        self.config = self._load_config(config_path)
        self.message_queue = Queue(maxsize=1000)
        self.db = DBConnector(**self.config['database'])
        self.yield_predictor = YieldPredictor()
        self._init_mqtt_client()
        self._setup_heartbeat()

    def _load_config(self, config_path: str) -> ConfigParser:
        """Load configuration from INI file"""
        config = ConfigParser()
        config.read(config_path)
        return config

    def _init_mqtt_client(self):
        """Initialize secure MQTT client with TLS"""
        self.client = mqtt.Client(
            client_id=self.config['mqtt']['client_id'],
            transport="tcp",
            protocol=mqtt.MQTTv311
        )
        
        # Configure TLS
        self.client.tls_set(
            ca_certs=self.config['mqtt']['ca_cert'],
            certfile=self.config['mqtt']['client_cert'],
            keyfile=self.config['mqtt']['client_key'],
            tls_version=ssl.PROTOCOL_TLSv1_2
        )
        
        self.client.username_pw_set(
            self.config['mqtt']['username'],
            self.config['mqtt']['password']
        )
        
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

    @retry(tries=3, delay=5, logger=logger)
    def connect(self):
        """Connect to MQTT broker with retry"""
        logger.info("Connecting to MQTT broker...")
        self.client.connect(
            self.config['mqtt']['broker'],
            port=int(self.config['mqtt']['port']),
            keepalive=60
        )
        self.client.loop_start()

    def _on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback"""
        if rc == 0:
            logger.info("Connected to MQTT broker")
            client.subscribe(self.config['mqtt']['topic'], qos=1)
        else:
            logger.error(f"Connection failed with code {rc}")

    def _on_message(self, client, userdata, message):
        """Incoming message handler"""
        try:
            payload = json.loads(message.payload.decode())
            self.message_queue.put((payload, datetime.utcnow()))
            logger.debug(f"Queued message: {payload['sensor_id']}")
        except Exception as e:
            logger.error(f"Message processing error: {str(e)}")

    def _process_messages(self):
        """Process messages from queue with threading"""
        while True:
            payload, received_at = self.message_queue.get()
            try:
                self._validate_payload(payload)
                self._process_sensor_data(payload, received_at)
                self.message_queue.task_done()
            except Exception as e:
                logger.error(f"Failed to process message: {str(e)}")

    def _validate_payload(self, payload: Dict[str, Any]):
        """Validate sensor data structure"""
        required_fields = ['sensor_id', 'temperature', 'humidity', 'soil_moisture']
        if not all(field in payload for field in required_fields):
            raise ValueError("Invalid payload structure")
        
        if not (0 <= payload['soil_moisture'] <= 1):
            raise ValueError("Invalid soil moisture value")

    def _process_sensor_data(self, data: Dict[str, Any], timestamp: datetime):
        """Process and analyze sensor data"""
        logger.info(f"Processing data from {data['sensor_id']}")
        
        # Store raw data
        self.db.insert_reading(data, timestamp)
        
        # AI integration
        prediction = self.yield_predictor.predict({
            'temp': data['temperature'],
            'humidity': data['humidity'],
            'soil_moisture': data['soil_moisture'],
            'ndvi': data.get('ndvi', 0.75)
        })
        
        logger.info(f"Yield prediction: {prediction:.2f} kg/ha")
        
        # Smart irrigation logic
        self._manage_irrigation(data, prediction)
        
        # Alert system
        self._check_anomalies(data)

    def _manage_irrigation(self, data: Dict[str, Any], prediction: float):
        """Smart irrigation management with AI"""
        base_threshold = 0.3
        dynamic_threshold = base_threshold * (1 - prediction/10000)
        
        if data['soil_moisture'] < dynamic_threshold:
            self._activate_irrigation(data['sensor_id'], duration=60)
            logger.warning(f"Activating irrigation for {data['sensor_id']}")

    @retry(tries=3, delay=2, logger=logger)
    def _activate_irrigation(self, sensor_id: str, duration: int):
        """Activate irrigation system with retry"""
        # Implementation for hardware control
        # Could be MQTT publish to actuator topic
        logger.info(f"Activating irrigation for {sensor_id} ({duration}s)")
        
    def _check_anomalies(self, data: Dict[str, Any]):
        """Check for sensor anomalies using statistical models"""
        # Implementation could use moving averages or ML models
        if data['temperature'] > 40:
            self._send_alert(f"High temperature alert: {data['temperature']}°C")

    def _send_alert(self, message: str):
        """Send alert through multiple channels"""
        logger.warning(f"ALERT: {message}")
        # Implementation for email/SMS/webhook integration

    def _setup_heartbeat(self):
        """Configure periodic health checks"""
        schedule.every(10).minutes.do(self._send_heartbeat)

        def heartbeat_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(1)

        threading.Thread(
            target=heartbeat_scheduler,
            name="HeartbeatScheduler",
            daemon=True
        ).start()

    def _send_heartbeat(self):
        """Send system health status"""
        status = {
            'queue_size': self.message_queue.qsize(),
            'db_connections': self.db.connection_count,
            'timestamp': datetime.utcnow().isoformat()
        }
        logger.info(f"Heartbeat: {status}")
        self.client.publish(
            topic=f"{self.config['mqtt']['topic']}/heartbeat",
            payload=json.dumps(status),
            qos=1
        )

    def _on_disconnect(self, client, userdata, rc):
        """Handle unexpected disconnections"""
        logger.warning(f"Disconnected from broker (rc={rc})")
        if rc != 0:
            self.connect()

    def run(self):
        """Start monitoring system"""
        processor_threads = [
            threading.Thread(
                target=self._process_messages,
                name=f"MessageProcessor-{i}",
                daemon=True
            ) for i in range(4)
        ]
        
        for thread in processor_threads:
            thread.start()

        self.connect()
        logger.info("IoT Monitoring System started")

if __name__ == "__main__":
    monitor = IoTMonitor()
    monitor.run()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down monitoring system")
        monitor.client.loop_stop()
