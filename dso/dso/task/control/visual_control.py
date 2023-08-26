
import gym
import numpy as np
from dso.vision_module.vision_observer import VisionObserver
from dso.task.control.control import Action, REWARD_SCALE, REWARD_SEED_SHIFT, create_decision_tree_tokens
from dso.functions import create_tokens
from dso.library import Library
from dso.program import Program
from dso.functions import create_tokens
import dso.task.control.utils as U
from dso.task import HierarchicalTask
    
class VisionControlTask(HierarchicalTask):
    """
    Class for the control task. Discrete objects are expressions, which are
    evaluated by directly using them as control policies in a reinforcement
    learning environment.
    """

    def __init__(self, function_set, env, action_spec, algorithm=None,
                 anchor=None, n_episodes_train=5, n_episodes_test=1000,
                 success_score=None, protected=False, env_kwargs=None,
                 fix_seeds=False, episode_seed_shift=0, reward_scale=True,
                 decision_tree_threshold_set=None, ref_action=None):
        """
        Parameters
        ----------

        function_set : list
            List of allowable functions.

        env : str
            Name of Gym environment, e.g. "Pendulum-v0" or "my_module:MyEnv-v0".

        action_spec : list
            List of action specifications: None, "anchor", or a list of tokens.

        algorithm : str or None
            Name of algorithm corresponding to anchor path, or None to use
            default anchor for given environment.

        anchor : str or None
            Path to anchor model, or None to use default anchor for given
            environment.

        n_episodes_train : int
            Number of episodes to run during training.

        n_episodes_test : int
            Number of episodes to run during testing.

        success_score : float
            Episodic reward considered to be "successful." A Program will have
            success=True if all n_episodes_test episodes achieve this score.

        protected : bool
            Whether or not to use protected operators.

        env_kwargs : dict
            Dictionary of environment kwargs passed to gym.make().

        fix_seeds : bool
            If True, environment uses the first n_episodes_train seeds for
            reward and the next n_episodes_test seeds for evaluation. This makes
            the task deterministic.

        episode_seed_shift : int
            Training episode seeds start at episode_seed_shift * 100 +
            REWARD_SEED_SHIFT. This has no effect if fix_seeds == False.

        reward_scale : list or bool
            If list: list of [r_min, r_max] used to scale rewards. If True, use
            default values in REWARD_SCALE. If False, don't scale rewards.

        decision_tree_threshold_set : list
            A set of constants {tj} for constructing nodes (xi < tj) in decision
            trees.
        """

        super(HierarchicalTask).__init__()

        # Set member variables used by member functions
        self.n_episodes_train = n_episodes_train
        self.n_episodes_test = n_episodes_test
        self.success_score = success_score
        self.fix_seeds = fix_seeds
        self.episode_seed_shift = episode_seed_shift
        self.stochastic = not fix_seeds
        
        # Create the environment
        env_name = env
        if env_kwargs is None:
            env_kwargs = {}
        self.env = gym.make(env_name, **env_kwargs)

        # HACK: Wrap pybullet envs in TimeFeatureWrapper
        # TBD: Load the Zoo hyperparameters, including wrapper features, not just the model.
        # Note Zoo is not implemented as a package, which might make this tedious
        if "Bullet" in env_name:
            self.env = U.TimeFeatureWrapper(self.env)
        
        self.action = Action(self.env.action_space)

        # Determine reward scaling
        if isinstance(reward_scale, list):
            assert len(reward_scale) == 2, "Reward scale should be length 2: \
                                            min, max."
            self.r_min, self.r_max = reward_scale
        elif reward_scale:
            if env_name in REWARD_SCALE:
                self.r_min, self.r_max = REWARD_SCALE[env_name]
            else:
                raise RuntimeError("{} has no default values for reward_scale. \
                                   Use reward_scale=False or specify \
                                   reward_scale=[r_min, r_max]."
                                   .format(env_name))
        else:
            self.r_min = self.r_max = None

        # Set the library (do this now in case there are symbolic actions)
        test_env = gym.make(env_name, **env_kwargs)
        test_env.reset()
        test_frame = test_env.render(mode='rgb_array')
        test_observer = VisionObserver()
        test_observation = test_observer.start(test_frame)
        test_env.close()
        n_input_var = test_observation.shape[0]
        self.n_input_var = n_input_var
        if self.action.is_discrete or self.action.is_multi_discrete:
            print("WARNING: The provided function_set will be ignored because "\
                  "action space of {} is {}.".format(env_name, self.env.action_space))
            tokens = create_decision_tree_tokens(n_input_var, decision_tree_threshold_set, 
                                                 self.env.action_space, ref_action)
        else:
            tokens = create_tokens(n_input_var, function_set, protected,
                                   decision_tree_threshold_set)
        self.library = Library(tokens)
        Program.library = self.library

        # Configuration assertions
        assert len(self.env.observation_space.shape) == 1, \
               "Only support vector observation spaces."
        n_actions = self.action.n_actions
        assert n_actions == len(action_spec), "Received spec for {} action \
               dimensions; expected {}.".format(len(action_spec), n_actions)
        if not self.action.is_multi_discrete:
            assert (len([v for v in action_spec if v is None]) <= 1), \
                   "No more than 1 action_spec element can be None."
        assert int(algorithm is None) + int(anchor is None) in [0, 2], \
               "Either none or both of (algorithm, anchor) must be None."

        # Generate symbolic policies and determine action dimension
        self.action.set_action_spec(action_spec, algorithm, anchor, env_name)
        
        # Define name based on environment and learned action dimension
        self.name = env_name
        if self.action.action_dim is not None:
            self.name += "_a{}".format(self.action.action_dim)

    def run_episodes(self, p, n_episodes, evaluate):
        """Runs n_episodes episodes and returns each episodic reward."""

        # Run the episodes and return the average episodic reward
        r_episodes = np.zeros(n_episodes, dtype=np.float64) # Episodic rewards for each episode
        for i in range(n_episodes):
            # During evaluation, always use the same seeds
            if evaluate:
                self.env.seed(i)
            elif self.fix_seeds:
                seed = i + (self.episode_seed_shift * 100) + REWARD_SEED_SHIFT
                self.env.seed(seed)
            self.env.reset()
            frame = self.env.render(mode='rgb_array')
            observer = VisionObserver()
            observation = observer.start(frame)

            done = False
            while not done:
                if observation.shape[0] < self.n_input_var:
                    comp = np.zeros((self.n_input_var - observation.shape[0]))
                    observation = np.concatenate((observation, comp))
                action = self.action(p, observation)
                obs, r, done, _ = self.env.step(action)
                frame = self.env.render(mode='rgb_array')
                observation = observer.step(frame)
                r_episodes[i] += r

        return r_episodes

    def reward_function(self, p, optimizing=False):

        # Run the episodes
        r_episodes = self.run_episodes(p, self.n_episodes_train, evaluate=False)

        # print("program:", p)
        # print("r_episodes:", r_episodes)

        # Return the mean
        r_avg = np.mean(r_episodes)

        # Scale rewards to [0, 1]
        if self.r_min is not None:
            r_avg = (r_avg - self.r_min) / (self.r_max - self.r_min)

        return r_avg

    def evaluate(self, p):

        # Run the episodes
        r_episodes = self.run_episodes(p, self.n_episodes_test, evaluate=True)

        # Compute eval statistics
        r_avg_test = np.mean(r_episodes)
        success_rate = np.mean(r_episodes >= self.success_score)
        success = success_rate == 1.0

        info = {
            "r_avg_test" : r_avg_test,
            "success_rate" : success_rate,
            "success" : success
        }
        return info
