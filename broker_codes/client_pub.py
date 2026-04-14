import time
import paho.mqtt.client as mqtt


def on_publish(client, userdata, mid, rc, properties=None):
    print("message published")


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, "rpi_client2") #this name should be unique
client.on_publish = on_publish
client.connect('192.168.0.134',1883)
# start a new thread
client.loop_start()

k=0
while True:
    k=k+1
    if(k>5):
        k=1 
        
    try:
        msg =str(k)
        pubMsg = client.publish(
            topic='rpi1/broadcast',
            payload=msg.encode('utf-8'),
            qos=0,
        )
        pubMsg.wait_for_publish()
        print(pubMsg.is_published())
    
    except Exception as e:
        print(e)
        
    time.sleep(2)

client.loop_stop()
