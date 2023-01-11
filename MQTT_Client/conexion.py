pip install paho-mqtt

pip install python-dotenv

from dotenv import dotenv_values
config = dotenv_values(".env")

import random
from paho.mqtt import client as mqtt_client
import json
from datetime import datetime

db_collection = 0

# Receive Data
# Connection Callback
cont=0
num=[]
def on_connect(client, userdata, flags, rc):
    print("temperatur  " + str(rc))
    client.subscribe("weatherStation")
# On Data Received Callback
def on_message(client, userdata, msg):
   global mensaje
   
   mensaje=json.loads(msg.payload)
   print(msg.topic + " " + str(msg.payload))
   print(mensaje)

   
   lol=mensaje["data"]["irrad"]
   num.insert(cont,lol)
   print(num)
   

# Leemos los datos y los agregamos a la lista
   
      

    

# Ahora mostremos las listas






      # Convert from json to python
   
    # Set date if not in data





#////////////////////////////////////////////###


# Generate a random client ID
client_id = f'python-mqtt-{random.randint(0, 10000)}'
# Create a new MQTT client instance
client = mqtt_client.Client(client_id=client_id)



client.username_pw_set(
  username="campusVerde",
   password="2m7eY8DDUU3MNQFcVN4LmDuvEv7a8LB2"
 )

# Start Connection
client.connect("campusverde.udenar.edu.co",1884, 60)

# Setup Callbacks
client.on_connect = on_connect  # Attach the on_connect function to the client
client.on_message = on_message  # Attach the on_message function to the client

# Start MQTT Client loop
while len(num)< 5:
  client.loop()
