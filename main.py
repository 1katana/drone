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
        result = await loop.run_in_executor(executor, camera)
        await queue.put(result)
    
    
def camera():
    
    image = sensor.receive_image()
    
    result=cv.detect(image)
    
    cv.visual(image,result)
    
    return result

async def main():
    queue = asyncio.Queue()
    current_coord = [[0, 0, 0.3], drone.get_orientation()[0]]

    time_not_target_start=-1

    # Запуск фоновой задачи для получения и обработки изображений
    producer_task = asyncio.create_task(image_producer(queue))

    while (t := sim.getSimulationTime()) < 300:  # Остановим симуляцию через 30 секунд
        # print(f'Simulation time: {t:.2f} [s]')

        if not queue.empty():
            result = await queue.get()

            queue.task_done()

            if len(result)==0:
                
                if time_not_target_start!=-1:
                    if t-time_not_target_start>3:
                        current_coord=drone.find_drone_use_target()
                else:
                    time_not_target_start=t
            
            else:
                time_not_target_start=-1
                current_coord=drone.interception_use_target(result)
         
        drone.control_use_target( *current_coord)

        # drone.update_angles(*current_coord)
        sim.step()  # Следующий шаг симуляции
        await asyncio.sleep(0)  # Позволяет другим задачам выполняться

    # Завершение работы ThreadPoolExecutor
    executor.shutdown(wait=True)

# Запуск основной функции
asyncio.run(main())