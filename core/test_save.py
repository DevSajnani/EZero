import os
import ray
import time
import torch

import numpy as np
import core.ctree.cytree as cytree

from tqdm.auto import tqdm
from torch.cuda.amp import autocast as autocast
from core.mcts import MCTS
from core.game import GameHistory
from core.utils import select_action, prepare_observation_lst


@ray.remote(num_gpus=0.25)
def _test_save(config, shared_storage):
    test_model = config.get_uniform_network()
    best_test_score = float('-inf')
    episodes = 0
    while True:
        counter = ray.get(shared_storage.get_counter.remote())
        if counter >= config.training_steps + config.last_steps:
            time.sleep(30)
            break
        if counter >= config.test_interval * episodes:
            episodes += 1
            test_model.set_weights(ray.get(shared_storage.get_weights.remote()))
            test_model.eval()

            test_score, eval_steps, _ = test_save(config, test_model, counter, config.test_episodes, config.device, False, save_video=False)
            mean_score = test_score.mean()
            std_score = test_score.std()
            print('Start evaluation at step {}.'.format(counter))
            if mean_score >= best_test_score:
                best_test_score = mean_score
                torch.save(test_model.state_dict(), config.model_path)

            test_log = {
                'mean_score': mean_score,
                'std_score': std_score,
                'max_score': test_score.max(),
                'min_score': test_score.min(),
            }

            shared_storage.add_test_log.remote(counter, test_log)
            print('Training step {}, test scores: \n{} of {} eval steps.'.format(counter, test_score, eval_steps))

        time.sleep(30)

class SaveFeatures():
     activations=None
     def __init__(self, layer, activations):
         self.hook = layer.register_forward_hook(self.hook_fn)
         self.activations = activations
     def hook_fn(self, module, input, output): self.activations.append(output.detach().cpu())
     def remove(self): self.hook.remove()

#attaches a target layer of the model and file folder to the saved activations
class target_layer():
    def __init__(self, layer, folder):
        self.layer = layer
        self.folder = folder
        #store activations as a list of pytorch tensors, then when it's big, save to disk and increment the chunk number
        self.activations = []
        self.chunk_number = 0


def save_chunk(targ_lay):
    #save stored activations to disk at folder/chunk number.pt
    dataset = torch.cat(targ_lay.activations, dim=0).to('cpu')
    os.makedirs(targ_lay.folder, exist_ok=True)
    with open(targ_lay.folder+"/"+str(targ_lay.chunk_number)+".pt","wb") as f:
        torch.save(dataset, f)
    #wipe stored activations
    targ_lay.activations = []
    targ_lay.chunk_number += 1


def test_save(config, model, counter, test_episodes, device, render, save_video=False, final_test=False, use_pb=False):
    """evaluation test
    Parameters
    ----------
    model: any
        models for evaluation
    counter: int
        current training step counter
    test_episodes: int
        number of test episodes
    device: str
        'cuda' or 'cpu'
    render: bool
        True -> render the image during evaluation
    save_video: bool
        True -> save the videos during evaluation
    final_test: bool
        True -> this test is the final test, and the max moves would be 108k/skip
    use_pb: bool
        True -> use tqdm bars
    """
    model.to(device)
    model.eval()
    save_path = os.path.join(config.exp_path, 'recordings', 'step_{}'.format(counter))

    with torch.no_grad():
        # new games
        envs = [config.new_game(seed=i, save_video=save_video, save_path=save_path, test=True, final_test=final_test,
                              video_callable=lambda episode_id: True, uid=i) for i in range(test_episodes)]


        max_episode_steps = envs[0].get_max_episode_steps()
        if use_pb:
            pb = tqdm(np.arange(max_episode_steps), leave=True)
        # initializations
        init_obses = [env.reset() for env in envs]
        dones = np.array([False for _ in range(test_episodes)])
        game_histories = [GameHistory(envs[_].env.action_space, max_length=max_episode_steps, config=config) for _ in range(test_episodes)]
        for i in range(test_episodes):
            game_histories[i].init([init_obses[i] for _ in range(config.stacked_observations)])

        step = 0
        ep_ori_rewards = np.zeros(test_episodes)
        ep_clip_rewards = np.zeros(test_episodes)

        #TODO: Initialize target_layers for all the layers we want to hook, remove test prints
        hooked_layers = set()
        hooked_layers.add(target_layer(model.projection[6],"proj/6"))

        # loop
        while not dones.all():
            if render:
                for i in range(test_episodes):
                    envs[i].render()

            if config.image_based:
                stack_obs = []
                for game_history in game_histories:
                    stack_obs.append(game_history.step_obs())
                stack_obs = prepare_observation_lst(stack_obs)
                stack_obs = torch.from_numpy(stack_obs).to(device).float() / 255.0
            else:
                stack_obs = [game_history.step_obs() for game_history in game_histories]
                stack_obs = torch.from_numpy(np.array(stack_obs)).to(device)
            
            #Set hooks
            temp_hooks = set()
            for l in hooked_layers:
                temp_hooks.add(SaveFeatures(l.layer, l.activations))

            #Call initial inference
            with autocast():
                network_output = model.initial_inference(stack_obs.float())

            #Manually call projection network
            proj = model.project(torch.from_numpy(network_output.hidden_state).to(device), with_grad=False)

            #Save chunks to disk if larger than desired chunk size (1 GB?) TODO: reset chunk size
            for l in hooked_layers:
                if sum(map(lambda x : torch.numel(x)*4, l.activations)) > 10**6:
                    save_chunk(l)

            #Remove hooks
            for h in temp_hooks:
                h.remove()

            hidden_state_roots = network_output.hidden_state
            reward_hidden_roots = network_output.reward_hidden
            value_prefix_pool = network_output.value_prefix
            policy_logits_pool = network_output.policy_logits.tolist()

            roots = cytree.Roots(test_episodes, config.action_space_size, config.num_simulations)
            roots.prepare_no_noise(value_prefix_pool, policy_logits_pool)
            # do MCTS for a policy (argmax in testing)
            MCTS(config).search(roots, model, hidden_state_roots, reward_hidden_roots)

            roots_distributions = roots.get_distributions()
            roots_values = roots.get_values()
            for i in range(test_episodes):
                if dones[i]:
                    continue

                distributions, value, env = roots_distributions[i], roots_values[i], envs[i]
                # select the argmax, not sampling
                action, _ = select_action(distributions, temperature=1, deterministic=True)

                obs, ori_reward, done, info = env.step(action)
                if config.clip_reward:
                    clip_reward = np.sign(ori_reward)
                else:
                    clip_reward = ori_reward

                game_histories[i].store_search_stats(distributions, value)
                game_histories[i].append(action, obs, clip_reward)

                dones[i] = done
                ep_ori_rewards[i] += ori_reward
                ep_clip_rewards[i] += clip_reward

            step += 1
            total_saved = 0
            for l in hooked_layers:
                total_saved += sum(map(lambda x : torch.numel(x)*4, l.activations))
            if use_pb:
                pb.set_description('{}, trial {}, mean score: {}, stored activations: {}, currently'
                                   ''.format(config.env_name, counter,
                                             ep_ori_rewards.mean(), total_saved))
                pb.update(1)

        for env in envs:
            env.close()

        #save the tail end of the data
        for l in hooked_layers:
            if sum(map(lambda x : torch.numel(x)*4, l.activations)) > 100:
                save_chunk(l)


    return ep_ori_rewards, step, save_path

