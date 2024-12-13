from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from sensorVision import convert_bytes_to_img, computer_vision
from drone import QuadcopterController,degrees_to_radians,radians_to_degrees
import numpy as np


client = RemoteAPIClient(port=23000)
sim = client.require("sim")

visionSensorHandle = sim.getObject('/drone/visionSensor')


cv=computer_vision("C:\\Users\\Katana\\Desktop\\drone\\best (2).onnx",[640,640])
    

    
# Включаем пошаговый режим симуляции
sim.setStepping(True)

# Запускаем симуляцию
sim.startSimulation()


drone=QuadcopterController(sim,cv)

while (t := sim.getSimulationTime()) < 300:  # Остановим симуляцию через 30 секунд
    print(f'Simulation time: {t:.2f} [s]')
    
    
    # Считываем изображение
    bytes,resolution  = sim.getVisionSensorImg(visionSensorHandle)
    
    image=convert_bytes_to_img(bytes,resolution)

    drone.interception(image)
    
    
    
    print()
    sim.step()