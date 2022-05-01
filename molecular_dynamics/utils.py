import numpy as np
import matplotlib.pyplot as plt

class particle_cloud:
    def __init__(self, M, L, T):
        self._number_of_particles = M
        self._length_of_box = L
        self._temperature = T
        self._shape = np.array([L, L, L])

        self._perfect_particles_1d = self.perfect_particles_1d(self._number_of_particles)
        self._perfect_number_of_particles = self.perfect_cube(self._perfect_particles_1d)

        self._ghost_particles = self._perfect_number_of_particles - self._number_of_particles

        self._axis_1d = self.linspace_axis_1d()

    def perfect_particles_1d(self, N):
        return np.floor(N ** (1/3)) + 1

    def perfect_cube(self, perfect_particles_1d): 
        return perfect_particles_1d ** 3

    def linspace_axis_1d(self):
        start = (self._length_of_box / (self._perfect_particles_1d + 1)) / 2
        stop = self._length_of_box - start
        num = self._perfect_particles_1d
        return np.linspace(start, stop, num, dtype= float)

    def get_array(self):
        # todo
        # return array of perfect cube with cords of linspace_axis_1d
        # remove number of ghost particles randomly
        # shift each particle coord with random distribution with distance of value start as sigma * 5
        

    
    
        
    

