import random
import logging

import aiomas

from creamas import CreativeAgent
import numpy as np


class CooperationAgent(CreativeAgent):

    def __init__(self, *args, **kwargs):
        self._pos = kwargs.pop('pos', (0, 0))
        self._map = kwargs.pop('map', np.zeros((20, 20)))
        super().__init__(*args, **kwargs)
        #print("Spawned on {}".format(self.pos))

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, npos):
        if self.in_map(npos):
            self._pos = npos
        else:
            raise ValueError("Cannot set position outside the map.")

    @property
    def map(self):
        return self._map

    @map.setter
    def map(self, nmap):
        if not type(nmap) == np.ndarray:
            raise ValueError("Map must be numpy.ndarray, got {}.".format(type(nmap)))
        self._map = nmap

    @aiomas.expose
    def get_pos(self):
        return self.pos

    @aiomas.expose
    def update_map(self, nmap):
        self.map = nmap

    def in_map(self, pos):
        return 0 <= pos[0] < self.map.shape[0] and 0 <= pos[1] < self.map.shape[1]

    def move_random(self):
        dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        possible_npos = []
        for dir in dirs:
            npos = (self.pos[0] + dir[0], self.pos[1] + dir[1])
            if self.in_map(npos):
                possible_npos.append(npos)
        return random.choice(possible_npos)

    @aiomas.expose
    async def act(self, *args, **kwargs):
        nmap = kwargs.pop('map', None)
        if nmap is not None:
            self.map = nmap
        self.pos = self.move_random()
        #print("Moved to {}".format(self.pos))
        return self.pos
