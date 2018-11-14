import gym
import numpy as np

class PytorchImage(gym.ObservationWrapper):
    def __init__(self, env):
        super(PytorchImage, self).__init__(env)
	# we check current shape of observations in environment
        current_shape = self.observation_space.shape
        # we change order of dimensions - so last one (-1) becomes first
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(current_shape[-1], current_shape[0], current_shape[1]), dtype=np.float32)

    def observation(self, observation):
        # and finally we change order of dimensions for every single observation
        # here transpose method could be also used
        return np.swapaxes(observation, 2, 0)