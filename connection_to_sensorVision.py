import socket
import numpy as np
import cv2
import time

class ConnectTCPVisionSensor:
    def __init__(self, HOST='127.0.0.1', PORT=20001):
        self.HOST = HOST
        self.PORT = PORT
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        while True:
            try:
                result = self.client_socket.connect_ex((self.HOST, self.PORT))
                if result == 0:
                    print("Соединение установлено успешно")
                    break
                else:
                    print(f"Ошибка соединения: {result}")

            except Exception as e:
                print(f"Исключение при подключении: {e}")
                time.sleep(1)
        print("Соединение установлено")

    def receive_image(self):
        try:
            # Отправляем серверу команду READY
            self.client_socket.sendall(b"READY")

            # Получаем заголовок
            header = b""
            while b"|" not in header:
                byte = self.client_socket.recv(1)
                if not byte:
                    raise ConnectionError("Соединение закрыто удаленной стороной")
                header += byte

            # Разбираем заголовок: "width,height|"
            header = header.decode('utf-8')
            width, height = map(int, header.strip('|').split(','))

            # Получаем изображение в байтах
            image_size = width * height * 3  # RGB
            image_data = b""
            while len(image_data) < image_size:
                packet = self.client_socket.recv(4096)
                if not packet:
                    raise ConnectionError("Соединение закрыто удаленной стороной")
                image_data += packet

            # Преобразуем байты в массив и изображение
            if len(image_data) == image_size:
                image = np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, 3))
                
                # Переворачиваем изображение (по вертикали)
                image = cv2.flip(image, 0)

                # Конвертируем из RGB в BGR для OpenCV
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                return image
            else:
                print("Получено неполное изображение!")
                return None
        except Exception as e:
            print(f"Ошибка при получении изображения: {e}")
            return None

    def close(self):
        try:
            self.client_socket.close()
            print("Соединение закрыто")
        except Exception as e:
            print(f"Ошибка при закрытии соединения: {e}")

            
def main():
    sensor = ConnectTCPVisionSensor()  # Устанавливаем соединение один раз

    try:
        while True:
            image = sensor.receive_image()
            if image is not None:
                cv2.imshow("Vision Sensor Image", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                print("Изображение не получено, повторная попытка...")
    except KeyboardInterrupt:
        print("Завершение работы по сигналу прерывания")
    finally:
        sensor.close()  # Закрываем соединение
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()