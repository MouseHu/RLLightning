import numpy as np
import os
import pickle as pkl


class StackReplayBuffer(object):
    def __init__(self, args, component):
        self.args = args
        self.component = component
        buffer_size = int(args.buffer_size)

        self.capacity = buffer_size
        self.curr_capacity = 0
        self.pointer = 0

        self.obs_space = component.env.observation_space
        self.action_shape = component.env.action_space.n
        self.rews_scale = args.rews_scale
        self.frames = args.frames

        self.begin_obs = dict()
        self.frame_buffer = np.empty((buffer_size,) + self.obs_space.shape[1:], np.uint8)
        self.action_buffer = np.empty([buffer_size, 1], np.int)
        self.episode_steps = np.zeros((buffer_size,), np.int)
        self.reward_buffer = np.empty((buffer_size,), np.float32)
        self.done_buffer = np.empty((buffer_size,), np.bool)

        self.end_points = []
        self.next_id = -1 * np.ones(buffer_size)
        self.prev_id = [[] for _ in range(buffer_size)]

        self.sequence = []

    def _get_ring_obs(self, idx, num_frame):
        begin_idx = (idx + num_frame) % self.capacity
        if begin_idx < idx:
            obs = np.concatenate([self.frame_buffer[idx:], self.frame_buffer[:begin_idx]], axis=0)
        else:
            obs = self.frame_buffer[idx:begin_idx, :, :]
        obs = obs[::-1, :, :]
        return obs

    def get_obs(self, idxes):
        return np.array([self._get_obs(idx) for idx in idxes])

    def _get_obs(self, idx):
        if self.episode_steps[idx] + 1 < self.frames:
            # need to append first
            first_idx = (idx + self.episode_steps[idx]) % self.capacity
            try:
                begin_obs = self.begin_obs[first_idx][self.episode_steps[idx]:-1, :, :]
            except KeyError:
                print(idx, self.episode_steps[idx], first_idx)
                print(list(self.begin_obs.keys()))
                raise KeyError
            end_obs = self._get_ring_obs(idx, self.episode_steps[idx] + 1)
            obs = np.concatenate([begin_obs, end_obs], axis=0)
        else:
            obs = self._get_ring_obs(idx, self.frames)
        return obs

    def clean(self):
        buffer_size, obs_space, action_shape = self.capacity, self.obs_space, self.action_shape
        self.curr_capacity = 0
        self.pointer = 0

        self.frame_buffer = np.empty((buffer_size,) + obs_space.shape[:-1], np.float32)
        self.begin_obs = dict()
        self.action_buffer = np.empty((buffer_size,) + action_shape, np.uint8)
        self.reward_buffer = np.empty((buffer_size,), np.float32)
        self.done_buffer = np.empty((buffer_size,), np.bool)

    def squeeze(self, obses):
        return np.array([(obs - self.obs_space.low) / (self.obs_space.high - self.obs_space.low) for obs in obses])

    def unsqueeze(self, obses):
        return np.array([obs * (self.obs_space.high - self.obs_space.low) + self.obs_space.low for obs in obses])

    def save(self, filedir):
        save_dict = {"frame_buffer": self.frame_buffer, "begin_obs": self.begin_obs,
                     "reward_buffer": self.reward_buffer, "done_buffer": self.done_buffer,
                     "curr_capacity": self.curr_capacity, "capacity": self.capacity}
        with open(os.path.join(filedir, "episodic_memory.pkl"), "wb") as memory_file:
            pkl.dump(save_dict, memory_file)

    def insert(self, obs, action, next_id=-1):

        index = self.pointer
        self.pointer = (self.pointer + 1) % self.capacity

        if self.curr_capacity >= self.capacity:
            # Clean up old entry
            if index in self.end_points:
                self.end_points.remove(index)
                for prev_id in self.prev_id[index]:
                    if prev_id not in self.end_points:
                        self.end_points.append(prev_id)
            self.prev_id[index] = []
            self.next_id[index] = -1
            if self.episode_steps[index] == 0:
                del self.begin_obs[index]
            self.episode_steps[index] = 0
        else:
            self.curr_capacity = min(self.capacity, self.curr_capacity + 1)
        # Store new entry
        self.frame_buffer[index] = obs[-1, :, :]
        self.action_buffer[index] = action

        if next_id >= 0:
            self.next_id[index] = next_id
            if index not in self.prev_id[next_id]:
                self.prev_id[next_id].append(index)

        return index

    def add_sequence(self, sequence):
        # print(sequence)
        next_id = -1
        episode_steps = len(sequence)
        ids = []
        for i, transition in enumerate(reversed(sequence)):
            obs, a, r, done = transition
            # print(np.mean(z))
            episode_steps = episode_steps - 1
            r = np.clip(r, -self.rews_scale, self.rews_scale)
            current_id = self.insert(obs, a, next_id)
            ids.append(current_id)
            if done:
                self.end_points.append(current_id)
            if episode_steps == 0:
                self.begin_obs[current_id] = obs
            self.frame_buffer[current_id] = obs[-1, :, :]
            self.reward_buffer[current_id] = r
            self.done_buffer[current_id] = done
            self.episode_steps[current_id] = episode_steps
            next_id = int(current_id)

        # for i, transition in enumerate(reversed(sequence)):
        #     obs, a, r, done = transition
        #     diff = obs - self.get_obs([ids[i]])[0]
        #     assert np.sum(abs(diff)) == 0, np.sum(abs(diff))

        return

    def add(self, obs_t, action, reward, done, obs_tp1):
        self.sequence.append((obs_t, action, reward, done))
        # print(done)
        if done:
            self.sequence.append((obs_tp1, action, reward, done))
            self.add_sequence(self.sequence)
            self.sequence = []

    def sample(self, batch_size):
        # Draw such that we always have a proceeding element
        if self.curr_capacity < batch_size + len(self.end_points):
            return None

        batch_idxs = []
        batch_idxs_next = []
        while len(batch_idxs) < batch_size:
            rnd_idx = np.random.randint(0, self.curr_capacity)
            if self.next_id[rnd_idx] == -1:
                continue
            else:
                batch_idxs_next.append(self.next_id[rnd_idx])
                batch_idxs.append(rnd_idx)

        batch_idxs = np.array(batch_idxs).astype(np.int)
        batch_idxs_next = np.array(batch_idxs_next).astype(np.int)

        obs0_batch = self.get_obs(batch_idxs)
        obs1_batch = self.get_obs(batch_idxs_next)
        action_batch = self.action_buffer[batch_idxs]
        reward_batch = self.reward_buffer[batch_idxs]
        terminal1_batch = self.done_buffer[batch_idxs]

        return (np.array(obs0_batch).astype(np.float32) / 255.0,
                np.array(action_batch).squeeze(),
                np.array(reward_batch).squeeze(),
                np.array(terminal1_batch).squeeze(),
                np.array(obs1_batch).astype(np.float32) / 255.0
                )
