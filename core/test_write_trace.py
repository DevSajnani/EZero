import os
import time
import torch
import sys

import numpy as np
import core.ctree.cytree as cytree
import ffmpeg

from tqdm.auto import tqdm
from torch.cuda.amp import autocast as autocast
from core.mcts import MCTS
from core.game import GameHistory
from core.utils import select_action, prepare_observation_lst, str_to_arr


def as_uint8(arr: np.ndarray) -> np.ndarray:
    #a helper method used by Ben Bolte for writing video
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.uint8)
    if np.issubdtype(arr.dtype, np.floating):
        return (arr * 255).round().astype(np.uint8)
    raise NotImplementedError(f"Unsupported dtype: {arr.dtype}")


def test_write_trace(config, model, test_episodes, device, ae_names, ae_features, feature_nums, counter=0, render=False, save_video=False, final_test=True, use_pb=False):
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
    Outputs
    ----------
    trace: list of time series (lists of activation tensors) - one time series per test episode
    game histories: list of GameHistory objects, one per test episode
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

        #initialize trace for every feature
        trace = {}
        for a in ae_names:
            trace[a]={}
            for m in feature_nums:
                #a[0:2] is the layer code
                if m < model.features(a[0:2]):
                    trace[a][m]=[]
        
        step = 0
        ep_ori_rewards = np.zeros(test_episodes)
        ep_clip_rewards = np.zeros(test_episodes)
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

            #Call initial inference
            with autocast():
                network_output = model.initial_inference(stack_obs.float())
            hidden_state_roots = network_output.hidden_state
            reward_hidden_roots = network_output.reward_hidden
            value_prefix_pool = network_output.value_prefix
            policy_logits_pool = network_output.policy_logits.tolist()

            #for each feature of each autoencoder, get a (max) value and a heatmap
            os.makedirs(config.exp_path, exist_ok = True)
            for a in ae_names:
                for m in feature_nums:
                    if m < model.features(a[0:2]):
                        val, heatmap = model.heatmap(a[0:2],ae_features[a]['offset'],ae_features[a][m])
                        trace[a][m].append(val)

                        #store video frames to disk for all test episodes simultaneously
                        frames = assemble_frames(trace[a][m], game_histories, heatmap)
                        for i in range(test_episodes):
                            frame_path = os.path.join(config.exp_path, a+f'{m}{i}{step}'+'.f')
                            torch.save(frames[i], frame_path)
            

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

                #update game history
                game_histories[i].store_search_stats(distributions, value)
                game_histories[i].append(action, obs, clip_reward)

                dones[i] = done
                ep_ori_rewards[i] += ori_reward
                ep_clip_rewards[i] += clip_reward

            step += 1
            if use_pb:
                pb.set_description('{} In step {}, scores: {}(max: {}, min: {}) currently.'
                                   ''.format(config.env_name, counter,
                                             ep_ori_rewards.mean(), ep_ori_rewards.max(), ep_ori_rewards.min()))
                pb.update(1)
 
        for env in envs:
            env.close()

    #ffmpeg output params
    width = str_to_arr(game_histories[0].obs_history[-1]).shape[0]
    height = width + 16
    vidscale = 5

    parent_dir = os.path.join(config.exp_path, 'heatmaps')

    #Iterate through autencoders and convert the saved frames to video
    #TODO: also iterate through the traces and find the N highest activating exampls
    for a in ae_names:
        #make directory
        os.makedirs(os.path.join(parent_dir, a), exist_ok = True)

        for m in feature_nums:
            if m < model.features(a[0:2]):
                if use_pb:
                    pb.set_description(f'{config.env_name} Writing video for encoder ' + a + f', feature {m} ')
                    pb.update(1)

                for i in range(test_episodes):
                    #set up stream
                    out_file = os.path.join(parent_dir, a, f'{m}ep{i}.mp4')
                    stream = ffmpeg.input("pipe:", format="rawvideo", pix_fmt="rgb24", s=f"{width*vidscale}x{height*vidscale}", r=30)
                    stream = ffmpeg.output(stream, str(out_file), pix_fmt="yuv420p", vcodec="libx264", r=30, loglevel="error")
                    stream = ffmpeg.overwrite_output(stream)
                    stream = ffmpeg.run_async(stream, pipe_stdin=True)

                    #make it so that while there are more frames from the current episode, they get used
                    s=0
                    frame_path = os.path.join(config.exp_path, a+f'{m}{i}{s}'+'.f')
                    while(os.path.exists(frame_path)):
                        frame = torch.load(frame_path, map_location=torch.device('cpu'))
                        
                        #kron with ones scales up frame by vidscale
                        stream.stdin.write(as_uint8(np.kron(frame, np.ones((vidscale,vidscale,1)))).tobytes())

                        #delete used frame
                        os.remove(frame_path)
                        s+=1
                        frame_path = os.path.join(config.exp_path, a+f'{m}{i}{s}'+'.f')

                    #close stream
                    stream.stdin.close()
                    stream.wait()

    return trace, game_histories


def assemble_frames(trace, game_histories, heatmaps) -> np.ndarray:
    """Parameters:
    trace is a time series of lists (each len test_episodes) of activations of a feature
    game_histories is a list (len test_episodes) of game_history objects we'll get observations from - observations are of shape (3,96,96)
    heatmaps is a np array of heatmaps of size text_episodes x 6 x 6
    Output:
    Returns a list by 96 (size of observations) by 96+16 (16 for graph of trace) by 3 (colors) array of floats
    frames should look like the latest observation, overlaid with the respective heatmap"""
    test_episodes = len(game_histories)
    #bsize is the height of the trace grap
    bsize = 16

    #need to take the raw obs object and get it into a numpy array
    #should have format [H, W, C] height, width, channel
    gb_obs = [str_to_arr(g.obs_history[-1]) for g in game_histories]
    width = gb_obs[0].shape[0]
    #convert to float and divide by 255
    gb_obs = [o.astype(float)/255.0 for o in gb_obs]

    #smush the observation into just green and blue channels 
    gb_obs = [o[:,:,1:3] + 0.5*o[:,:,0:1] for o in gb_obs]
    gb_obs = [np.pad(o, ((0,bsize),(0,0),(1,0))) for o in gb_obs]

    frames = np.zeros((test_episodes, width+bsize, width, 3))

    #broadcast the observation to the frames
    frames = frames + gb_obs

    #normalize each heatmap individually to be between 0 and 1
    for m in heatmaps:
        if m.max() > m.min():
            m[...] = (m-m.min())/(m.max()-m.min())
        else:
            m[...] = np.zeros(m.shape)

    #scale up heatmaps by a factor of 96/6 = 16 and put them into red channel
    #np.kron takes a direct matrix product, which when done with ones looks like scaling up
    mapscale = 16
    scale_shape = (1,mapscale,mapscale)
    frames[:,:,:,0] = np.pad(np.kron(heatmaps, np.ones(scale_shape)) , ((0,0),(0,bsize),(0,0)))

    #make a graph of trace:
    #build a buffer of up to 96-1 points, each an average of frames
    buffers = np.zeros((test_episodes,width-1))
    avging = 6
    index = 0
    while index < width-1 and index + 1 < len(trace)/avging:
        chunk = trace[len(trace)-index*avging:len(trace)-(index-1)*avging]
        if len(chunk)>0:
            buffers[:,index] = np.mean(trace[len(trace)-index*avging:len(trace)-(index-1)*avging], axis=0)
        index+=1

    #rescale buffers and round to integers
    for b in buffers:
        if b.max() > b.min():
            b[...] = (b-b.min())*(bsize/(b.max()-b.min()))
        else:
            b[...] = np.zeros((width-1))
    buffers = buffers.astype(int)

    #draw a white line on the right side
    frames[:,width:width+bsize,-1]=np.ones((test_episodes,bsize,3))
    #draw in trace
    for n in range(test_episodes):
        for t in range(width-1):
            frames[n,-1-buffers[n,t],-2-t]=np.ones((3))

    return frames