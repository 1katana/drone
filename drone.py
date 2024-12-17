from coppeliasim_zmqremoteapi_client import RemoteAPIClient
from propeller import Propeller
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
from sensorVision import computer_vision


def radians_to_degrees(radians):
    return radians * (180 / math.pi)

def degrees_to_radians(degrees):
    return degrees * (math.pi / 180)

def normalize_angle(angle):
    """
    Нормализует угол в диапазон [-180, 180].
    """
    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle

class QuadcopterController:
    def __init__(self, sim,cv:computer_vision,droneTarget=True):
        """
        Инициализация класса управления квадрокоптером.

        :param sim: Объект симуляции (API CoppeliaSim).
        """
        self.sim = sim
        self.drone = sim.getObject('/drone')
        
        self.droneTarget= sim.getObject('/droneTarget')
        
        self.sim.setObjectPosition(self.droneTarget,self.sim.getObjectPosition(self.drone))
        
        self.sim.setObjectOrientation(self.droneTarget,[0,0,self.get_orientation()[0]])
        
        self.cv=cv

        self.propellers = [
            Propeller(sim, 0),
            Propeller(sim, 1),
            Propeller(sim, 2),
            Propeller(sim, 3)
        ]

        # Параметры ПИД-регулятора для высоты
        self.pParam_height = 2
        self.iParam_height = 0
        self.dParam_height = 0
        self.vParam_height = -2
        self.cumul_height_error = 0.0
        self.last_height_error = 0.0


        # Параметры ПИД-регулятора для крена (roll)
        self.pParam_roll = -0.25
        self.iParam_roll = -0.005
        self.dParam_roll = -0.7

        # Параметры ПИД-регулятора для тангажа (pitch)
        self.pParam_pitch = -0.25
        self.iParam_pitch = -0.005
        self.dParam_pitch = -0.75

        # Параметры ПИД-регулятора для рыскания (yaw)
        self.pParam_yaw = 0.25
        self.iParam_yaw = 0.001
        self.dParam_yaw = 2.5
        
        

        self.cumul_roll_error = 0.0
        self.last_roll_error = 0.0


        self.cumul_pitch_error = 0.0
        self.last_pitch_error = 0.0



        self.cumul_yaw_error = 0.0
        self.last_yaw_error = 0.0
        self.last_yaw=0.0
        self.cumul_yaw_error = 0.0
        self.max_thrust = 6

    def get_pos(self):
        pos=self.sim.getObjectPosition(self.drone)
        # print("height: ",height)
        return pos

    def control_use_target(self, target_pos,target_yaw):
        self.sim.setObjectPosition(self.droneTarget,target_pos)
        euler = list(self.get_orientation())
        euler[0]=target_yaw
        self.sim.setObjectOrientation(self.droneTarget,[0,0,target_yaw])

    def get_orientation(self):
        euler = self.sim.getObjectOrientation(self.drone)
        
        return self.sim.alphaBetaGammaToYawPitchRoll(*euler)
    
    def get_height(self):
        
        height=self.sim.getObjectPosition(self.drone)[2]
        # print("height: ",height)
        return height

    def update_angles(self, target_yaw,target_pitch,target_roll, target_height):
        """
        Обновление состояния квадрокоптера для управления углами.

        :param target_roll: Целевой угол крена.
        :param target_pitch: Целевой угол тангажа.
        :param target_yaw: Целевой угол рыскания.
        :param thrust: Тяга, рассчитанная для достижения целевой высоты.
        """
        # Получаем линейную скорость
        l = self.sim.getObjectVelocity(self.drone)[0]
        

        # Вертикальная стабилизация с использованием ПИД-регулятора
        current_height = self.get_height()
        error = target_height - current_height
        self.cumul_height_error += error
        derivative_error = error - self.last_height_error

        p_term = self.pParam_height * error
        i_term = self.iParam_height * self.cumul_height_error
        d_term = self.dParam_height * derivative_error

        thrust = 5.45 + p_term + i_term + d_term + l[2] * self.vParam_height

        self.last_height_error = error
        
        # Управление угловыми положениями с использованием ПИД-регуляторов

        yaw, pitch, roll=self.get_orientation()
            
        # print("Euler Angles: ", yaw, pitch, roll,current_height)

        roll_error = target_roll - roll
        pitch_error = target_pitch - pitch
        yaw_error = target_yaw - yaw

        # print("error: ",yaw_error,pitch_error,roll_error,error)

        self.cumul_roll_error += roll_error
        self.cumul_pitch_error += pitch_error
        self.cumul_yaw_error += yaw_error

        derivative_roll_error = roll_error - self.last_roll_error
        derivative_pitch_error = pitch_error - self.last_pitch_error
        derivative_yaw_error = yaw_error - self.last_yaw_error

        p_roll_term = self.pParam_roll * roll_error
        i_roll_term = self.iParam_roll * self.cumul_roll_error
        d_roll_term = self.dParam_roll * derivative_roll_error

        p_pitch_term = self.pParam_pitch * pitch_error
        i_pitch_term = self.iParam_pitch * self.cumul_pitch_error
        d_pitch_term = self.dParam_pitch * derivative_pitch_error

        p_yaw_term = self.pParam_yaw * yaw_error
        i_yaw_term = self.iParam_yaw * self.cumul_yaw_error
        d_yaw_term = self.dParam_yaw * derivative_yaw_error

        roll_corr = p_roll_term + i_roll_term + d_roll_term
        pitch_corr = p_pitch_term + i_pitch_term + d_pitch_term
        yaw_corr = p_yaw_term + d_yaw_term + i_yaw_term

        self.last_roll_error = roll_error
        self.last_pitch_error = pitch_error
        self.last_yaw_error = yaw_error

        # Определение скоростей двигателей
        self.propellers[0].handle_propeller(thrust * (1 - roll_corr + pitch_corr + yaw_corr))
        self.propellers[1].handle_propeller(thrust * (1 - roll_corr - pitch_corr - yaw_corr))
        self.propellers[2].handle_propeller(thrust * (1 + roll_corr - pitch_corr + yaw_corr))
        self.propellers[3].handle_propeller(thrust * (1 + roll_corr + pitch_corr - yaw_corr))

        return roll_error, pitch_error, yaw_error
    
    def get_velicity_yaw(self):

        return self.sim.getObjectVelocity(self.drone)[1][2]
        
        
    def interception_use_target(self,result) -> tuple[float, float, float, float]:
        

        speed=0.1
            
            
        drone_pos,yaw=self.cv.calculate_drone_target(result,self.get_pos(),self.get_orientation()[0])
        
        drone_pos[0] += speed * math.cos(yaw)
        drone_pos[1] += speed * math.sin(yaw)
        
        
        # self.update_angles(yaw, pitch, roll, self.get_height())
        
        return drone_pos,yaw

    def find_drone_use_target(self):
        # Получаем текущую высоту
        current_height = self.get_height()
        
        # Устанавливаем границы высоты
        min_height = 0.2
        max_height = 3.0
        
        # Ограничиваем высоту в допустимых пределах
        target_height = max(min_height, min(current_height, max_height))
        
        # Обновляем угол рыскания с небольшим поворотом
        if self.get_velicity_yaw()>0.8:
            target_yaw = -0.2
        else:
            target_yaw = 0.2
        
        
        
        return [[0, 0, target_height],  self.get_orientation()[0]+target_yaw ]

    
    
    
    def interception(self,result) -> tuple[float, float, float, float]:
        

        if len(result)==0:
            self.cumul_height_error=0
            self.cumul_pitch_error=0
            self.cumul_roll_error=0
            self.cumul_yaw_error=0
            return self.find_drone()
        else:
            
            
            yaw, pitch, roll, height=self.cv.calculate_drone_angles(result,self.get_orientation(),self.get_height())
            
            # self.update_angles(yaw, pitch, roll, self.get_height())
            
            return yaw, pitch, roll, height
    
    def find_drone(self):
        # Получаем текущую высоту
        current_height = self.get_height()
        
        # Устанавливаем границы высоты
        min_height = 0.1
        max_height = 3.0
        
        # Ограничиваем высоту в допустимых пределах
        target_height = max(min_height, min(current_height-0.05, max_height))
        
        # Обновляем угол рыскания с небольшим поворотом
        target_yaw = self.get_orientation()[0] + 0.2
        
        return target_yaw, 0, 0, target_height
        # # Обновляем параметры дрона
        # self.update_angles(target_yaw, 0, 0, target_height)
        
        