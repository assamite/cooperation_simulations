import asyncio
import random
import time
import os

import aiomas
from creamas import Environment, MultiEnvironment, EnvManager, MultiEnvManager
from creamas.util import run, wait_tasks

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

from serializers import get_serializers


class MapEnvironment(Environment):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.name = "MapEnvironment"

    def update_maps(self, nmap):
        for agent in self.get_agents(addr=False):
            agent.update_map(nmap)


class MapEnvManager(EnvManager):

    @aiomas.expose
    def update_maps(self, nmap):
        self.env.update_maps(nmap)


class MapMultiEnvironment(MultiEnvironment):

    def __init__(self, *args, **kwargs):
        self._map = kwargs.pop('map', None)
        self._n_agents = kwargs.pop('n_agents', 1000)
        super().__init__(*args, **kwargs)

        self.step = 0
        self.max_steps = 0

        self.fig = None
        self.im = None
        self.ani = None
        #self.im_vmin = -self._n_agents
        #self.im_vmax = self._n_agents
        self.im_vmin = -10
        self.im_vmax = int(np.sqrt(self._n_agents))

    async def update_maps(self, nmap):
        self._map = nmap
        async def slave_task(addr, nmap):
            r_manager = await self.env.connect(addr, timeout=5)
            return await r_manager.update_maps(nmap)

        tasks = []
        for i, addr in enumerate(self.addrs):
            task = asyncio.ensure_future(slave_task(addr, nmap))
            tasks.append(task)

        return await wait_tasks(tasks)

    def write_map(self, iteration):
        fig, ax = plt.subplots()
        ax.imshow(self._map, vmin=self.im_vmin, vmax=self.im_vmax, interpolation=None)
        ax.set_title("Map")
        fig.tight_layout()
        plt.axis('off')
        plt.savefig(os.path.join('images', "{:0>6}.png".format(iteration)))
        plt.close()

    def run_animation(self, steps):
        self.max_steps += steps

        self.fig = plt.figure(figsize=(6, 6))
        plt.axis('off')
        plt.tight_layout()
        self.im = plt.imshow(self._map, vmin=self.im_vmin, vmax=self.im_vmax, interpolation=None, animated=True)
        self.ani = animation.FuncAnimation(self.fig, self.animation_step, interval=0, blit=True)
        plt.show()

    def animation_step(self, *args, **kwargs):
        t1 = time.monotonic()
        self.step += 1
        if self.step > self.max_steps:
            # it takes a while for the animation loop to notice the event
            self.ani.event_source.stop()
        else:
            ret = run(self.trigger_all())
            self.im.set_data(self._map)

        print("Step {} in {:.4f} seconds.".format(self.step, time.monotonic()-t1))
        return self.im,

    async def trigger_all(self, *args, **kwargs):
        kwargs['map'] = self._map
        ret = await super().trigger_all(*args, **kwargs)
        self._map[:, :] = self.im_vmin
        for r in ret:
            if self._map[r] <= 0:
                self._map[r] = 1
            else:
                self._map[r] += 1

        return ret


addr = ('localhost', 5555)
env_kwargs = {'extra_serializers': get_serializers(), 'codec': aiomas.MsgPack}

map_size = 50
n_agents = 1000
agent_map = np.zeros((map_size, map_size))
menv = MapMultiEnvironment(addr,
                           env_cls=Environment,
                           mgr_cls=MultiEnvManager,
                           logger=None,
                           map=agent_map,
                           n_agents=n_agents,
                           **env_kwargs)

# Define slave environments and their arguments
n_slaves = 4
slave_addrs = [('localhost', 5556+i) for i in range(n_slaves)]
slave_env_cls = MapEnvironment
slave_mgr_cls = MapEnvManager
slave_kwargs = [{'extra_serializers': get_serializers(), 'codec': aiomas.MsgPack} for _ in range(n_slaves)]

# Spawn the actual slave environments
run(menv.spawn_slaves(slave_addrs, slave_env_cls, slave_mgr_cls, slave_kwargs))

# Wait that all the slaves are ready, if you need to do some other
# preparation before environments' return True for their is_ready-method, then
# change check_ready=False
run(menv.wait_slaves(10, check_ready=True))

# Set host managers for the slave environments
ret = run(menv.set_host_managers())

# Check that the multienvironment is ready (this double checks that the slaves are ready).
ret = run(menv.is_ready())

if not ret:
    raise RuntimeWarning("Not all the slave environments report to be ready before continuing the execution.")

# Spawn the agents to slave environments.
all_agent_pos = []
tasks = []
t1 = time.monotonic()
for _ in range(n_agents):
    agent_pos = (random.randrange(map_size), random.randrange(map_size))
    agent_map[agent_pos[0], agent_pos[1]] += 1
    agent_kwargs = {'pos': agent_pos, 'map': agent_map}
    run(menv.spawn("agents:CooperationAgent", **agent_kwargs))

print("Spawned {} in {:.3f} seconds.".format(n_agents, time.monotonic()-t1))
#print(agent_map)
run(menv.update_maps(agent_map))


# Run the agent animation for a number of steps
n_steps = 100

t1 = time.monotonic()
menv.run_animation(steps=n_steps)
#print("Actual average time per step: {:.3f}.".format((time.monotonic()-t1) / n_steps))

# Destroy the environment to free the resources
menv.destroy(as_coro=False)

