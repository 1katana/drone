import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

import asyncio

def getVisionSensorImg(sim,visionSensorHandle):
    
    bytes,resolution = sim.getVisionSensorImg(visionSensorHandle)
    return bytes,resolution


def convert_bytes_to_img(bytes: bytes, resolution: list):
    """
    Конвертирует байты из датчика изображения в NumPy массив с преобразованием в float32.

    :param bytes: Байтовые данные изображения
    :param resolution: Список из двух элементов [ширина, высота]
    :return: NumPy массив изображения в формате float32
    """
    if bytes is not None and resolution:
        # Конвертация из байтов в список, а затем в NumPy массив
        image_array = np.frombuffer(bytes, dtype=np.uint8)

        image_array = image_array.reshape((resolution[1], resolution[0], 3))  # Высота, ширина, RGB
        
        # Переворот изображения по вертикали (если требуется)
        image_array = np.flip(image_array, axis=0)
        
        # image_array = image_array.astype(np.float32) / 255.0
        
        # plt.imshow(image_array)
        # plt.axis('off')
        # plt.show()
        image = cv2.cvtColor(image_array , cv2.COLOR_RGB2BGR)
        return image
    else:
        print("Ошибка: изображение не получено или данные некорректны.")
        return None


class computer_vision():
    
    def __init__(self,net_path: str,input_size:list, fov_deg=60):
        
        self.net=cv2.dnn.readNet(model=net_path)
        self.input_size=input_size
        self.fov_deg=fov_deg
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print(cv2.cuda.getCudaEnabledDeviceCount())
        

    
    def process_yolo_output(self,output, confidence_threshold=0.4, iou_threshold=0.4):
        """
        Обрабатывает вывод YOLO модели.
        
        :param output: Вывод модели [1, 8, 8400]
        :param confidence_threshold: Порог уверенности для фильтрации предсказаний.
        :param nms_threshold: Порог для подавления перекрытий.
        :param input_size: Размер входного изображения (ширина, высота).
        :return: Список объектов в формате [class_id, confidence, x1, y1, x2, y2].
        """
        # Убираем первую ось (размер батча)
        predictions = output[0]  # Форма: [8, 8400]
        
        # Транспонируем массив для удобства (если нужно)
        predictions = predictions.T  # Форма: [8400, 8]
        
        boxes = []
        confidences = []
        class_ids = []

        for pred in predictions:
            # Первые 4 элемента — вероятности классов
            class_scores = pred[4:]
            class_id = np.argmax(class_scores)
            class_confidence = class_scores[class_id]

            # Условие фильтрации по порогу уверенности
            if class_confidence > confidence_threshold:
                # Следующие 4 элемента — координаты бокса (x_center, y_center, width, height)
                x_center, y_center, width, height = pred[:4]

                # Добавляем результат
                boxes.append([x_center, y_center, width, height])
                confidences.append(float(class_confidence))
                class_ids.append(class_id)

        # print(class_ids,confidences,boxes)
        
        # Применяем NMS
        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes,
            scores=confidences,
            score_threshold=confidence_threshold,
            nms_threshold=iou_threshold,
        )

        results = []
        for i in indices:
            # Добавляем только боксы, прошедшие NMS
            x_center, y_center, width, height = boxes[i]
            results.append([class_ids[i], confidences[i], x_center, y_center, width, height])

        return results
    
    def visual(self,image,result):
        # Создаем изменяемую копию изображения
        image_copy = image.copy()

        
        # Отображение якорных рамок (bounding boxes)
        for detection in result:
            # Рамки: (x, y) - координаты центра, (w, h) - размеры
            x, y, w, h = detection[2:]  # Пример: первые 4 значения это координаты центра и размеры
            confidence = detection[1]    # Доверие для рамки
            class_id = int(detection[0])  # Идентификатор класса
            
            # Преобразование координат рамки для корректного отображения
            x1, y1 = int(x-(w/2)), int(y-(h/2))
            x2, y2 = int(x + (w/2)), int(y + (h/2))
            
            # Отображение рамки на копии изображения
            cv2.rectangle(image_copy, (x1,y1), (x2,y2), (0, 255, 0), 2)  # Зеленая рамка для объектов
            cv2.putText(image_copy, f'Class {class_id} Conf: {confidence:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # Отображение изображения с рамками
        cv2.imshow("Detections", image_copy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
    
        
    def detect(self,image):
        
        # img=cv2.imread("C:\\Users\\Katana\\Desktop\\drone\\64dde2f302e8bd0cb20df32a.jpg")
        
        blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=self.input_size, mean=(0, 0, 0), swapRB=True, crop=False)
        
        
        
        # Передача изображения в сеть
        self.net.setInput(blob)

        # Прямой проход (inference)
        outputs = self.net.forward()
        
        
        result=self.process_yolo_output(output=outputs)
        # print(result)
        
        return result
    
    def calculate_drone_angles(self, target_coords, drone_orientation):
        """
        Calculate the yaw, pitch, and roll angles for the drone to aim the camera at the target.

        Parameters:
        - image_size: Tuple (width, height) representing the image size (e.g., (1024, 1024)).
        - target_coords: Tuple (x, y) representing the coordinates of the target on the image.
        - drone_orientation: Tuple (yaw, pitch, roll) representing the drone's orientation in degrees.
        - fov_deg: Field of view of the camera in degrees (default is 60 degrees).

        Returns:
        - yaw (degrees): Horizontal rotation angle of the drone.
        - pitch (degrees): Vertical rotation angle of the drone.
        - roll (degrees): Roll angle of the drone.
        """
        if len(target_coords)==0:
            return drone_orientation
        
        index=0
        maxSquare=0
        
        for i in range(len(target_coords)):
            square=target_coords[i][-1]*target_coords[i][-2]
            if square>maxSquare:
                index = i 
                maxSquare=square

        target=target_coords[index]
            
        
        # Расчет угла на пиксель
        width, height = self.input_size
        fov_rad = math.radians(self.fov_deg)
        angle_per_pixel = fov_rad / (width / 2)

        # Координаты цели на изображении
        x_target, y_target = target[2:4]

        # Центр изображения
        x_center = width / 2
        y_center = height / 2

        # Расчет углов рыскания и тангажа относительно изображения
        yaw_image = (x_target - x_center) * angle_per_pixel
        pitch_image = (y_target - y_center) * angle_per_pixel

        # Учитываем ориентацию дрона (в частности, крен и тангаж)
        yaw, pitch, roll = drone_orientation
        print("цель: ",yaw_image,pitch_image)
        # Возвращаем вычисленные углы
        return yaw-yaw_image, pitch, roll