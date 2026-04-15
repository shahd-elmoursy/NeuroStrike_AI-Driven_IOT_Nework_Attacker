import paho.mqtt.client as mqtt

BROKER = "192.168.0.134"
PORT = 1883
TOPIC = "hazards/#"

def on_connect(client, userdata, flags, rc, properties=None):
    print("Connected with result code", rc)
    client.subscribe(TOPIC)

def on_message(client, userdata, msg):
    print(f"Received: {msg.payload.decode()} on topic {msg.topic}")

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, 'rpi_client3')
client.on_connect = on_connect
print('connected to broker')
client.on_message = on_message

client.connect(BROKER, PORT)
client.loop_forever()
