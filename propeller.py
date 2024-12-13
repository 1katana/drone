from coppeliasim_zmqremoteapi_client import RemoteAPIClient
import numpy as np
import math
import random

class Propeller:
    def __init__(self, sim, index: int):
        """
        Инициализация класса пропеллера.

        :param sim: Объект симуляции (API CoppeliaSim).
        :param index: Индекс пропеллера (0-3).
        """
        self.sim = sim
        self.index = index

        # Параметры частиц
        self.particle_density = 8500
        self.particle_size = 0.005
        self.particle_count_per_second = 430
        self.particle_scattering_angle = 30
        self.simulate_particles = True

        # Инициализация объектов
        self.propeller_respondable = sim.getObject(f'/drone/propeller[{index}]/respondable')
        self.propeller_joint = sim.getObject(f'/drone/propeller[{index}]/respondable/body/joint')
        self.particle_object = -1

        if self.simulate_particles:
            particle_type = (
                sim.particle_roughspheres |
                sim.particle_cyclic |
                sim.particle_respondable1to4 |
                sim.particle_respondable5to8 |
                sim.particle_ignoresgravity
            )
            self.particle_object = sim.addParticleObject(
                particle_type,
                self.particle_size,
                self.particle_density,
                [2, 1, 0.2, 3, 0.4],  # параметры поведения частиц
                0.5,  # время жизни частиц
                50,   # максимальное количество частиц
                [0.3, 0.7, 1]  # цвет частиц
            )

        # Для контроля частиц
        self.not_full_particles = 0

    def handle_propeller(self, particle_velocity):
        """
        Управление вращением пропеллера и генерация частиц.

        :param particle_velocity: Скорость частиц.
        """
        # Установка позиции шарнира пропеллера
        t = self.sim.getSimulationTime()
        self.sim.setJointPosition(self.propeller_joint, t * 10)

        # Получение матрицы трансформации
        m = np.array(self.sim.getObjectMatrix(self.propeller_respondable)).reshape(3, 4)

        # Генерация частиц
        ts=self.sim.getSimulationTimeStep()
        required_particle_count = self.particle_count_per_second * ts + self.not_full_particles
        self.not_full_particles = required_particle_count % 1
        required_particle_count = int(required_particle_count)

        for _ in range(required_particle_count):
            x, y = random.uniform(-1, 1), random.uniform(-1, 1)
            if x**2 + y**2 <= 1:
                if self.simulate_particles:
                    pos = np.array([x * 0.08, y * 0.08, -self.particle_size * 0.6, 1.0])
                    dir = np.array([
                        pos[0] + random.uniform(-1, 1) * math.tan(math.radians(self.particle_scattering_angle) / 2) * particle_velocity,
                        pos[1] + random.uniform(-1, 1) * math.tan(math.radians(self.particle_scattering_angle) / 2) * particle_velocity,
                        pos[2] - particle_velocity * (1 + 0.2 * random.uniform(-0.5, 0.5)),
                        0.0
                    ])

                    # Применение матрицы трансформации
                    pos = np.dot(m[:, :3], pos[:3]) + m[:, 3]
                    dir = np.dot(m[:, :3], dir[:3])

                    # Добавление частицы
                    item_data = list(pos) + list(dir)
                    self.sim.addParticleObjectItem(self.particle_object, item_data)

        # Применение реактивной силы
        particle_volume = (4 / 3) * math.pi * (self.particle_size / 2) ** 3
        total_exerted_force = required_particle_count * particle_volume * self.particle_density * particle_velocity / ts
        force = np.array([0, 0, total_exerted_force])
        torque = np.array([0, 0, (1 - (self.index % 2) * 2) * 0.002 * particle_velocity])

        # Применение трансформации силы и момента
        force = np.dot(m[:, :3], force)
        torque = np.dot(m[:, :3], torque)

        # Добавление силы и момента
        self.sim.addForceAndTorque(self.propeller_respondable, force.tolist(), torque.tolist())