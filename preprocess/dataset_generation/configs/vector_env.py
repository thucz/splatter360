
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import multiprocessing as mp
from multiprocessing.connection import Connection

# import billiard as mp
# from billiard.connection import Connection

from queue import Queue
from threading import Thread
from typing import Any, Callable, Iterable, List, Optional, Set, Tuple, Union

import gym
import numpy as np
# from gym.spaces.dict_space import Dict as SpaceDict#gym==0.10.9
from gym.spaces.dict import Dict as SpaceDict#gym=0.23.

import habitat
from habitat.config import Config
from habitat.core.env import Env, Observations
from habitat.core.logging import logger
from habitat.core.utils import tile_images

from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.core.simulator import ShortestPathPoint
from habitat.core.logging import logger
from habitat.utils.geometry_utils import quaternion_to_list
from habitat.datasets.utils import get_action_shortest_path

STEP_COMMAND = "step"
RESET_COMMAND = "reset"
RENDER_COMMAND = "render"
CLOSE_COMMAND = "close"
OBSERVATION_SPACE_COMMAND = "observation_space"
ACTION_SPACE_COMMAND = "action_space"
CALL_COMMAND = "call"
NAVIGABLE_COMMAND = 'navigate'
OBSERVATIONS = 'observations'
AGENT_STATE_COMMAND = 'get_agent_state'
SET_AGENT_STATE_COMMAND = 'set_agent_state'
GET_SHORTEST_PATH_COMMAND = 'get_action_shortest_path'
GET_CURRENT_SCENE_COMMAND = 'get_current_scene'
def _make_env_fn(
    config: Config, dataset: Optional[habitat.Dataset] = None, rank: int = 0
) -> Env:
    r"""Constructor for default mhabitat Env.

    Args:
        config: configuration for environment.
        dataset: dataset for environment.
        rank: rank for setting seed of environment

    Returns:
        ``Env``/``RLEnv`` object
    """
    habitat_env = Env(config=config, dataset=dataset)
    habitat_env.seed(config.SEED + rank)
    # import pdb;pdb.set_trace()
    print("\n\n\n")
    print("Note:")
    print("habitat_sim:", habitat_env._sim)
    print("\n\n\n")

    return habitat_env


class VectorEnv:
    r"""Vectorized environment which creates multiple processes where each
    process runs its own environment. All the environments are synchronized
    on step and reset methods.

    Args:
        make_env_fn: function which creates a single environment. An
            environment can be of type Env or RLEnv
        env_fn_args: tuple of tuple of args to pass to the make_env_fn.
        auto_reset_done: automatically reset the environment when
            done. This functionality is provided for seamless training
            of vectorized environments.
        multiprocessing_start_method: the multiprocessing method used to
            spawn worker processes. Valid methods are
            ``{'spawn', 'forkserver', 'fork'}`` ``'forkserver'`` is the
            recommended method as it works well with CUDA. If
            ``'fork'`` is used, the subproccess  must be started before
            any other GPU useage.
    """

    observation_spaces: SpaceDict
    action_spaces: SpaceDict
    _workers: List[Union[mp.Process, Thread]]
    _is_waiting: bool
    _num_envs: int
    _auto_reset_done: bool
    _mp_ctx: mp.context.BaseContext
    _connection_read_fns: List[Callable[[], Any]]
    _connection_write_fns: List[Callable[[Any], None]]

    def __init__(
        self,
        make_env_fn: Callable[..., Env] = _make_env_fn,
        env_fn_args: Tuple[Tuple] = None,
        auto_reset_done: bool = True,
        multiprocessing_start_method: str = "forkserver",
    ) -> None:

        self._is_waiting = False
        self._is_closed = True
        self.env_fn_args = env_fn_args
        # self.env = make_env_fn(*env_fn_args)

        assert (
            env_fn_args is not None and len(env_fn_args) > 0
        ), "number of environments to be created should be greater than 0"

        self._num_envs = len(env_fn_args)

        assert multiprocessing_start_method in self._valid_start_methods, (
            "multiprocessing_start_method must be one of {}. Got '{}'"
        ).format(self._valid_start_methods, multiprocessing_start_method)
        self._auto_reset_done = auto_reset_done
        self._mp_ctx = mp.get_context(multiprocessing_start_method)
        self._workers = []
        (
            self._connection_read_fns,
            self._connection_write_fns,
        ) = self._spawn_workers(  # noqa
            env_fn_args, make_env_fn
        )

        self._is_closed = False

        for write_fn in self._connection_write_fns:
            write_fn((OBSERVATION_SPACE_COMMAND, None))
        self.observation_spaces = [
            read_fn() for read_fn in self._connection_read_fns
        ]
        for write_fn in self._connection_write_fns:
            write_fn((ACTION_SPACE_COMMAND, None))
        self.action_spaces = [
            read_fn() for read_fn in self._connection_read_fns
        ]
        self._paused = []

    @property
    def num_envs(self):
        r"""
        Returns:
             number of individual environments.
        """
        return self._num_envs - len(self._paused)

    @staticmethod
    def _worker_env(
        connection_read_fn: Callable,
        connection_write_fn: Callable,
        env_fn: Callable,
        env_fn_args: Tuple[Any],
        auto_reset_done: bool,
        child_pipe: Optional[Connection] = None,
        parent_pipe: Optional[Connection] = None,
    ) -> None:
        """process worker for creating and interacting with the environment.
        """
        env = env_fn(*env_fn_args)
        # env = self.env
        if parent_pipe is not None:
            parent_pipe.close()
        try:
            command, data = connection_read_fn()
            # import ipdb;ipdb.set_trace()
            while command != CLOSE_COMMAND:
                if command == STEP_COMMAND:
                    # different step methods for mhabitat.RLEnv and mhabitat.Env
                    if isinstance(env, habitat.RLEnv) or isinstance(
                        env, gym.Env
                    ):
                        # mhabitat.RLEnv
                        observations, reward, done, info = env.step(data)
                        if auto_reset_done and done:
                            observations = env.reset()
                        connection_write_fn((observations, reward, done, info))
                    elif isinstance(env, habitat.Env):
                        # mhabitat.Env
                        observations = env.step(data)
                        if auto_reset_done and env.episode_over:
                            observations = env.reset()
                        connection_write_fn(observations)
                    else:
                        raise NotImplementedError

                elif command == RESET_COMMAND:
                    observations = env.reset()
                    connection_write_fn(observations)

                elif command == RENDER_COMMAND:
                    connection_write_fn(env.render(*data[0], **data[1]))

                elif (
                    command == OBSERVATION_SPACE_COMMAND
                    or command == ACTION_SPACE_COMMAND
                ):
                    connection_write_fn(getattr(env, command))

                elif command == CALL_COMMAND:
                    function_name, function_args = data
                    if function_args is None or len(function_args) == 0:
                        result = getattr(env, function_name)()
                    else:
                        result = getattr(env, function_name)(*function_args)
                    connection_write_fn(result)
                elif command == NAVIGABLE_COMMAND:
                    location = env.sim.sample_navigable_point()
                    connection_write_fn(location)
                elif command == OBSERVATIONS:
                    position, rotation = data
                    observations = env.sim.get_observations_at(position=position,
                                                                    rotation=rotation,
                                                                    keep_agent_at_new_pose=True)
                    connection_write_fn((observations))
                elif command == AGENT_STATE_COMMAND:
                    agent_state = env.sim.get_agent_state().sensor_states['depth']
                    rotation = np.array([agent_state.rotation.w, agent_state.rotation.x, agent_state.rotation.y,
                                         agent_state.rotation.z])
                    connection_write_fn((agent_state.position, rotation))
                    # raise Exception # for debug
                elif command == SET_AGENT_STATE_COMMAND:
                    # agent_state = env.sim.set_agent_state().sensor_states['depth']
                    # rotation = np.array([agent_state.rotation.w, agent_state.rotation.x, agent_state.rotation.y,
                    #                      agent_state.rotation.z])
                    # connection_write_fn()
                    position, rotation, reset_sensors = data
                    status = env.sim.set_agent_state(position=position, rotation=rotation, reset_sensors=reset_sensors)
                    connection_write_fn((status))
                elif command == GET_SHORTEST_PATH_COMMAND:
                    # data: 
                    source_position, source_rotation, goal_position, success_distance, max_episode_steps = data


                    # env.sim.reset()
                    # env.sim.set_agent_state(source_position, source_rotation)
                    # follower = ShortestPathFollower(env.sim, success_distance, False)
                    # shortest_path = []
                    # step_count = 0

                    # action = follower.get_next_action(goal_position)
                    # while (
                    #     action is not HabitatSimActions.STOP and step_count < max_episode_steps
                    # ):
                    #     state = env.sim.get_agent_state()
                    #     shortest_path.append(
                    #         ShortestPathPoint(
                    #             state.position.tolist(),
                    #             quaternion_to_list(state.rotation),
                    #             action,
                    #         )
                    #     )
                    #     env.sim.step(action)
                    #     step_count += 1
                    #     action = follower.get_next_action(goal_position)

                    # if step_count == max_episode_steps:
                    #     logger.warning("Shortest path wasn't found.")

                    shortest_path = get_action_shortest_path(
                                env.sim,
                                source_position=source_position,
                                source_rotation=source_rotation,
                                goal_position=goal_position,
                                success_distance=success_distance,
                                max_episode_steps=max_episode_steps,
                            )
                    connection_write_fn((shortest_path))
                elif command == GET_CURRENT_SCENE_COMMAND:
                    index = data
                    # print("env._config.SIMULATOR.SCENE:", env._config.SIMULATOR.SCENE)                
                    current_scene = env._config.SIMULATOR.SCENE #.split("/")[-2]

                    connection_write_fn((current_scene))
                    









                else:
                    raise NotImplementedError

                command, data = connection_read_fn()

            if child_pipe is not None:
                child_pipe.close()
        except KeyboardInterrupt:
            logger.info("Worker KeyboardInterrupt")
        finally:
            env.close()

    def _spawn_workers(
        self,
        env_fn_args: Iterable[Tuple[Any, ...]],
        make_env_fn: Callable[..., Env] = _make_env_fn,
    ) -> Tuple[List[Callable[[], Any]], List[Callable[[Any], None]]]:
        parent_connections, worker_connections = zip(
            *[self._mp_ctx.Pipe(duplex=True) for _ in range(self._num_envs)]
        )
        self._workers = []
        for worker_conn, parent_conn, env_args in zip(
            worker_connections, parent_connections, env_fn_args
        ):
            ps = self._mp_ctx.Process(
                target=self._worker_env,
                args=(
                    worker_conn.recv,
                    worker_conn.send,
                    make_env_fn,
                    env_args,
                    self._auto_reset_done,
                    worker_conn,
                    parent_conn,
                ),
            )
            self._workers.append(ps)
            ps.daemon = True
            ps.start()
            worker_conn.close()
        return (
            [p.recv for p in parent_connections],
            [p.send for p in parent_connections],
        )

    def reset(self):
        r"""Reset all the vectorized environments

        Returns:
            list of outputs from the reset method of envs.
        """
        self._is_waiting = True
        for write_fn in self._connection_write_fns:
            write_fn((RESET_COMMAND, None))

        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        self._is_waiting = False
        return results

    def reset_at(self, index_env: int):
        r"""Reset in the index_env environment in the vector.

        Args:
            index_env: index of the environment to be reset

        Returns:
            list containing the output of reset method of indexed env.
        """
        self._is_waiting = True
        self._connection_write_fns[index_env]((RESET_COMMAND, None))
        results = [self._connection_read_fns[index_env]()]
        self._is_waiting = False
        return results

    def step_at(self, index_env: int, action: int):
        r"""Step in the index_env environment in the vector.

        Args:
            index_env: index of the environment to be stepped into
            action: action to be taken

        Returns:
            list containing the output of step method of indexed env.
        """
        self._is_waiting = True
        self._connection_write_fns[index_env]((STEP_COMMAND, action))
        results = [self._connection_read_fns[index_env]()]
        self._is_waiting = False
        return results

    def async_step(self, actions: List[int]) -> None:
        r"""Asynchronously step in the environments.

        Args:
            actions: actions to be performed in the vectorized envs.
        """
        self._is_waiting = True
        for write_fn, action in zip(self._connection_write_fns, actions):
            write_fn((STEP_COMMAND, action))

    def wait_step(self) -> List[Observations]:
        r"""Wait until all the asynchronized environments have synchronized.
        """
        observations = []
        for read_fn in self._connection_read_fns:
            observations.append(read_fn())
        self._is_waiting = False
        return observations

    def step(self, actions: List[int]):
        r"""Perform actions in the vectorized environments.

        Args:
            actions: list of size _num_envs containing action to be taken
                in each environment.

        Returns:
            list of outputs from the step method of envs.
        """
        self.async_step(actions)
        return self.wait_step()

    def get_observations_at(self, index: int, position: List[float], rotation: List[float]):
        self._is_waiting = True
        self._connection_write_fns[index]((OBSERVATIONS, (position, rotation)))
        observations = self._connection_read_fns[index]()
        # import ipdb;ipdb.set_trace()
        # obs = observations.copy()
        # normalized_rgb = obs["rgb"]
        # cv2.imwrite("rgb_test.jpg", normalized_rgb)


        self._is_waiting = False
        return observations

    def sample_navigable_point(self, index: int):
        self._is_waiting = True
        self._connection_write_fns[index]((NAVIGABLE_COMMAND,None))
        locations = self._connection_read_fns[index]()
        self._is_waiting = False
        return locations


    def get_agent_state(self, index: int):
        self._is_waiting = True
        self._connection_write_fns[index]((AGENT_STATE_COMMAND,None))
        cameras = self._connection_read_fns[index]()
        self._is_waiting = False
        return cameras
    
    # todo: set_agent_state;
    def set_agent_state(self, index: int,
        position: List[float],
        rotation: List[float],
        reset_sensors: bool = True):

        self._is_waiting = True
        self._connection_write_fns[index]((SET_AGENT_STATE_COMMAND,  (position, rotation, reset_sensors)))
        status = self._connection_read_fns[index]()
        self._is_waiting = False
        return status
    
    def get_action_shortest_path(self, index: int,
        source_position: List[float],
        source_rotation: List[float],
        goal_position: List[float],
        success_distance: float,
        max_episode_steps: int):
        self._is_waiting = True
        self._connection_write_fns[index]((GET_SHORTEST_PATH_COMMAND, (source_position, source_rotation, goal_position, success_distance, max_episode_steps )))
        shortest_path = self._connection_read_fns[index]()
        self._is_waiting = False
        return shortest_path
    def get_current_scene(self, index: int):
        self._is_waiting = True
        self._connection_write_fns[index]((GET_CURRENT_SCENE_COMMAND, (index)))
        current_scene = self._connection_read_fns[index]()
        self._is_waiting = False
        return current_scene


    def close(self) -> None:
        if self._is_closed:
            return

        if self._is_waiting:
            for read_fn in self._connection_read_fns:
                read_fn()

        for write_fn in self._connection_write_fns:
            write_fn((CLOSE_COMMAND, None))

        for _, _, write_fn, _ in self._paused:
            write_fn((CLOSE_COMMAND, None))

        for process in self._workers:
            process.join()

        for _, _, _, process in self._paused:
            process.join()

        self._is_closed = True

    def pause_at(self, index: int) -> None:
        r"""Pauses computation on this env without destroying the env. This is
        useful for not needing to call steps on all environments when only
        some are active (for example during the last episodes of running
        eval episodes).

        Args:
            index: which env to pause. All indexes after this one will be
                shifted down by one.
        """
        if self._is_waiting:
            for read_fn in self._connection_read_fns:
                read_fn()
        read_fn = self._connection_read_fns.pop(index)
        write_fn = self._connection_write_fns.pop(index)
        worker = self._workers.pop(index)
        self._paused.append((index, read_fn, write_fn, worker))

    def resume_all(self) -> None:
        r"""Resumes any paused envs.
        """
        for index, read_fn, write_fn, worker in reversed(self._paused):
            self._connection_read_fns.insert(index, read_fn)
            self._connection_write_fns.insert(index, write_fn)
            self._workers.insert(index, worker)
        self._paused = []

    def call_at(
        self,
        index: int,
        function_name: str,
        function_args: Optional[List[Any]] = None,
    ) -> Any:
        r"""Calls a function (which is passed by name) on the selected env and
        returns the result.

        Args:
            index: which env to call the function on.
            function_name: the name of the function to call on the env.
            function_args: optional function args.

        Returns:
            result of calling the function.
        """
        self._is_waiting = True
        self._connection_write_fns[index](
            (CALL_COMMAND, (function_name, function_args))
        )
        result = self._connection_read_fns[index]()
        self._is_waiting = False
        return result

    def call(
        self,
        function_names: List[str],
        function_args_list: Optional[List[Any]] = None,
    ) -> List[Any]:
        r"""Calls a list of functions (which are passed by name) on the
        corresponding env (by index).

        Args:
            function_names: the name of the functions to call on the envs.
            function_args_list: list of function args for each function. If
                provided, len(function_args_list) should be as long as
                len(function_names).

        Returns:
            result of calling the function.
        """
        self._is_waiting = True
        if function_args_list is None:
            function_args_list = [None] * len(function_names)
        assert len(function_names) == len(function_args_list)
        func_args = zip(function_names, function_args_list)
        for write_fn, func_args_on in zip(
            self._connection_write_fns, func_args
        ):
            write_fn((CALL_COMMAND, func_args_on))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        self._is_waiting = False
        return results

    def render(
        self, mode: str = "human", *args, **kwargs
    ) -> Union[np.ndarray, None]:
        r"""Render observations from all environments in a tiled image.
        """
        for write_fn in self._connection_write_fns:
            write_fn((RENDER_COMMAND, (args, {"mode": "rgb", **kwargs})))
        images = [read_fn() for read_fn in self._connection_read_fns]
        tile = tile_images(images)
        if mode == "human":
            import cv2

            cv2.imshow("vecenv", tile[:, :, ::-1])
            cv2.waitKey(1)
            return None
        elif mode == "rgb_array":
            return tile
        else:
            raise NotImplementedError

    @property
    def _valid_start_methods(self) -> Set[str]:
        return {"forkserver", "spawn", "fork"}

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ThreadedVectorEnv(VectorEnv):
    r"""Provides same functionality as ``VectorEnv``, the only difference is it
    runs in a multi-thread setup inside a single process. ``VectorEnv`` runs
    in a multi-proc setup. This makes it much easier to debug when using
    ``VectorEnv`` because you can actually put break points in the environment
    methods. It should not be used for best performance.
    """

    def _spawn_workers(
        self,
        env_fn_args: Iterable[Tuple[Any, ...]],
        make_env_fn: Callable[..., Env] = _make_env_fn,
    ) -> Tuple[List[Callable[[], Any]], List[Callable[[Any], None]]]:
        parent_read_queues, parent_write_queues = zip(
            *[(Queue(), Queue()) for _ in range(self._num_envs)]
        )
        self._workers = []
        for parent_read_queue, parent_write_queue, env_args in zip(
            parent_read_queues, parent_write_queues, env_fn_args
        ):
            thread = Thread(
                target=self._worker_env,
                args=(
                    parent_write_queue.get,
                    parent_read_queue.put,
                    make_env_fn,
                    env_args,
                    self._auto_reset_done,
                ),
            )
            self._workers.append(thread)
            thread.daemon = True
            thread.start()
        return (
            [q.get for q in parent_read_queues],
            [q.put for q in parent_write_queues],
        )