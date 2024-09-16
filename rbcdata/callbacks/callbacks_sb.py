import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy


class ExampleCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``, to show what's possible.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

        def _on_training_start(self) -> None:
            """
            This method is called before the first rollout starts.
            """
            pass

        def _on_rollout_start(self) -> None:
            """
            A rollout is the collection of environment interactions
            using the current policy.
            """
            pass

        def _on_step(self) -> bool:
            """
            This method will be called by the model after each call to `env.step()`.

            :return: (bool) If the callback returns False, training is aborted early.
            """
            return True

        def _on_rollout_end(self) -> None:
            """
            This event is triggered before updating the policy.
            """
            pass

        def _on_training_end(self) -> None:
            """
            This event is triggered before exiting the `learn()` method.
            """
            pass


class LogNusseltCallback(BaseCallback):
    """
    Used to log the nusselt number during training (and evaluation?)

    :param freq: The frequency at which to log the nusselt number.
    """

    def __init__(self, freq: int, verbose=0):
        super().__init__(verbose)
        self.freq = freq
        self.nusselts = []
        # self.logger.record_mean   TODO used for logging.

    def _on_step(self) -> bool:
        if self.n_calls % self.freq == 0:
            # TODO MS does not really seem to be doing what it should. Check again. These
            # locals are not always up to date it seems...
            infos = self.locals["infos"]
            nusselt_averages = np.zeros(len(infos))
            for i, info in enumerate(infos):
                nusselt_averages[i] = info["nusselt_avg"]
            self.logger.info(
                f'Timestep {self.num_timesteps}. Mean Nusselt over last {info["nusselt_len"]} \
                    steps in {len(nusselt_averages)} environments: {np.mean(nusselt_averages)}.\
                    Std. among environments: {np.std(nusselt_averages)}'
            )

        return True

    def _on_rollout_end(self) -> None:
        pass


class EvalCallback(BaseCallback):
    """
    Callback for evaluating an agent. It will evaluate the agent at the end of each episode.

    :param eval_env: The environment to evaluate the agent on.
    :param callback_on_new_best: A callback to call when a new best model is found.
    :param callback_after_eval: A callback to call after the evaluation.
    :param n_eval_episodes: The number of episodes to evaluate the agent on.
    :param best_model_save_path: The path to save the best model.
    :param log_path: The path to save the log.
    :param eval_freq: The frequency at which to evaluate the agent.
    :param deterministic: Whether to use a deterministic policy.
    :param render: Whether to render the environment.
    """

    def __init__(
        self,
        eval_env,
        callback_on_new_best,
        callback_after_eval,
        n_eval_episodes,
        eval_freq,
        deterministic,
        render,
        verbose=0,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.callback_on_new_best = callback_on_new_best
        self.callback_after_eval = callback_after_eval
        self.n_eval_episodes = n_eval_episodes
        self.best_mean_reward = -np.inf
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        self.render = render

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self.logger.info(f"Step {self.num_timesteps}. Evaluating model...")
            mean_reward, std_reward = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
                render=self.render,
            )
            self.logger.info(f"Mean evaluation reward: {mean_reward:.2f} +/- {std_reward:.2f}")
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.logger.info(f"New best model found with mean reward: {mean_reward:.2f}.")
                # self.model.save(self.best_model_save_path)
                # model saving already handled by callback_on_new_best
                self.callback_on_new_best.on_step()
            self.callback_after_eval.on_step()

        return True

    def _on_rollout_end(self) -> None:
        pass