import numpy as np
import matplotlib.pyplot as plt
from itertools import product

class particle_cloud:
    def __init__(self, M, L, T):
        self._number_of_particles = M
        self._length_of_box = L
        self._temperature = T
        self._shape = np.array([L, L, L])

        self._perfect_particles_1d = self.perfect_particles_1d(self._number_of_particles)
        self._perfect_number_of_particles = self.perfect_cube(self._perfect_particles_1d)

        self._ghost_particles = self._perfect_number_of_particles - self._number_of_particles

        self._particle_box_length = (self._length_of_box / (self._perfect_particles_1d + 1)) / 2


        self._axis_1d = self.linspace_axis_1d()

    def perfect_particles_1d(self, N):
        return int(np.floor(N ** (1/3)) + 1)

    def perfect_cube(self, perfect_particles_1d): 
        return int(perfect_particles_1d ** 3)

    def linspace_axis_1d(self):
        start = self._particle_box_length
        stop = self._length_of_box - start
        num = self._perfect_particles_1d
        return np.linspace(start, stop, num)

    def get_meshgrid(self):
        # todo
        arr = np.meshgrid(self._axis_1d,self._axis_1d,self._axis_1d)
#        pos = np.vstack(map(np.ravel, arr)).T
 
        return arr

    def get_array(self):
        # array of perfect cube with cords of linspace_axis_1d
        arr = np.array(list(product(self._axis_1d,self._axis_1d,self._axis_1d)))
        # remove number of ghost particles randomly
        print(f"arr len before pop: {len(arr)}")
        print(f"ghost particles: {self._ghost_particles}")
        arr = arr.tolist()
        for _ in range(self._ghost_particles):
            random_index = np.random.randint(0, len(arr))
            arr.pop(random_index)
        print(f"arr len after pop: {len(arr)}")
        # shift each particle coord with random distribution with distance of value start as sigma * 5
        for i in range(len(arr)):
            for j in range(len(arr[i])):
                shift_border = self._particle_box_length * 0.3
                shift = np.random.uniform((-1)*shift_border, shift_border) # A single value
                arr[i][j] += shift
        return arr    
    




    def get_array_dev(self):
        # todo
        # return array of perfect cube with cords of linspace_axis_1d
        print(len(self._axis_1d))
        arr = np.meshgrid(self._axis_1d,self._axis_1d,self._axis_1d)
        print(len(arr))
        
        # remove number of ghost particles randomly
        print(f"arr len before pop: {len(arr)}")
        print(f"ghost particles: {self._ghost_particles}")
        for _ in range(self._ghost_particles):
            random_index = np.random.randint(0, len(arr))
            arr.pop(random_index)
        # shift each particle coord with random distribution with distance of value start as sigma * 5
        print(f"arr len after pop: {len(arr)}")
        return arr

    
    
        
    

