from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from sensorVision import convert_bytes_to_img, computer_vision,getVisionSensorImg
from drone import QuadcopterController,degrees_to_radians,radians_to_degrees
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
from connection_to_sensorVision import ConnectTCPVisionSensor
import threading




client = RemoteAPIClient(port=23000)
sim = client.require("sim")

visionSensorHandle = sim.getObject('/drone/visionSensor')


cv=computer_vision("best (2).onnx",[640,640])
    

    
# Включаем пошаговый режим симуляции
sim.setStepping(True)

# Запускаем симуляцию
sim.startSimulation()



drone=QuadcopterController(sim,cv)

executor = ThreadPoolExecutor()

sensor=ConnectTCPVisionSensor()


async def image_producer(queue):
    """
    Задача, которая получает изображения с сенсора, обрабатывает их и помещает в очередь.
    """
    loop = asyncio.get_running_loop()
    
    while True:
        image = await loop.run_in_executor(executor, sensor.receive_image)
        await queue.put(image)
        

async def main():
    queue = asyncio.Queue()
    current_coord = [0, 0, 0, 0.4]


    # Запуск фоновой задачи для получения и обработки изображений
    producer_task = asyncio.create_task(image_producer(queue))

    while (t := sim.getSimulationTime()) < 300:  # Остановим симуляцию через 30 секунд
        # print(f'Simulation time: {t:.2f} [s]')

        if not queue.empty():
            image = await queue.get()
            queue.task_done()
            
            # result=cv.detect(image)
            
            cv.visual(image,[])
            
            # drone.interception(result)
            
    
        drone.update_angles(*current_coord)
        sim.step()  # Следующий шаг симуляции
        await asyncio.sleep(0)  # Позволяет другим задачам выполняться

    # Завершение работы ThreadPoolExecutor
    executor.shutdown(wait=True)

# Запуск основной функции
asyncio.run(main())