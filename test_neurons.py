import torch

import argparse
import os
import random

from config.atari import game_config
from config.atari.heatmap_model import EfficientZeroNetH
from core.test_write_trace import test_write_trace
from core.game import GameHistory



def one_hot(n_neurons, n, device):
    temp = torch.zeros(n_neurons).to(device)
    temp[n]+=1
    return temp


#test input: python test_neurons.py --env MsPacmanNoFrameskip-v4 --model_path ./model/model.p --sae_paths ./sae/test_sae.p --random_features=5

if __name__ == '__main__':
    #gather arguments related to testing:
    parser = argparse.ArgumentParser(description='EfficientZero')
    parser.add_argument('--env', required=True, help='Name of the gym environment')
    parser.add_argument('--test_episodes', type=int, default=2, help='Evaluation episode count (default: %(default)s)')
    parser.add_argument('--model_path', type=str, default='./results/model.p', help='load model path')
    parser.add_argument('--sae_paths', nargs='+', type=str, default=['./results/sae.p'], help='load some autoencoder paths')
    parser.add_argument('--sae_layers', nargs='+', type=str, default=['p6'], help='layers for the loaded autoencoders. Currently supported: p6')
    parser.add_argument('--results_path', type=str, default='./results/test_neurons', help='save clips directory')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    parser.add_argument('--save_neurons', action='store_true', help='Flag: Attribute neurons if no autoencoder at file')
    parser.add_argument('--random_features', type=int, default=50, help='Number of random features to attribute (default: %(default)s)')
    parser.add_argument('--features', nargs='+', type=int, default=[], help='Specific features to attribute if possible')
    parser.add_argument('--feature_source', type=str, default='decoder', help='if untied weights, the feature should come from the decoder')
    #TODO: make sure they actually got passed and used

    args = parser.parse_args()
    assert os.path.exists(args.model_path), 'model not found at {}'.format(args.model_path)
    for p in args.sae_paths:
        if args.save_neurons:
            if not os.path.exists(p):
                print('model not found at {}, saving neurons'.format(p))
        else:
            assert os.path.exists(p), 'autoencoder not found at {}'.format(p)
    device = args.device

    #set configs
    #implied_args = type('', (object,),{"env":"BreakoutNoFrameskip-v4", "case":"atari","opr":"test","amp_type":"torch_amp","render":False,"seed":0,"use_priority":False,"use_max_priority":False,"debug":False,"device":'cpu',"cpu_actor":4,"gpu_actor":4,"p_mcts_num":4,"use_root_value":False,"use_augmentation":False,"revisit_policy_search_rate":0.99,"result_dir":"./","info":"none"})()
    implied_args = type('', (object,),{"env":args.env,
                                        "case":"atari",
                                        "opr":"test",
                                        "amp_type":"torch_amp",
                                        "render":False,
                                        "seed":0,
                                        "use_priority":False,
                                        "use_max_priority":False,
                                        "debug":False,
                                        "device":device,
                                        "cpu_actor":4,
                                        "gpu_actor":4,
                                        "p_mcts_num":4,
                                        "use_root_value":False,
                                        "use_augmentation":False,
                                        "revisit_policy_search_rate":0.99,
                                        "result_dir":args.results_path,
                                        "info":"none"
                                        })()
    exp_path = game_config.set_config(implied_args)

    #initialize model with a custom model that has internal variables and a heatmap method
    model = EfficientZeroNetH(
            game_config.obs_shape,
            game_config.action_space_size,
            game_config.blocks,
            game_config.channels,
            game_config.reduced_channels_reward,
            game_config.reduced_channels_value,
            game_config.reduced_channels_policy,
            game_config.resnet_fc_reward_layers,
            game_config.resnet_fc_value_layers,
            game_config.resnet_fc_policy_layers,
            game_config.reward_support.size,
            game_config.value_support.size,
            game_config.downsample,
            game_config.inverse_value_transform,
            game_config.inverse_reward_transform,
            game_config.lstm_hidden_size,
            bn_mt=game_config.bn_mt,
            proj_hid=game_config.proj_hid,
            proj_out=game_config.proj_out,
            pred_hid=game_config.pred_hid,
            pred_out=game_config.pred_out,
            init_zero=game_config.init_zero,
            state_norm=game_config.state_norm).to(device)
    
    #load model
    model.load_state_dict(torch.load(args.model_path, map_location=torch.device(device)), strict=False)

    #assemble set of feature numbers to test
    feature_nums = set()
    #TODO: pick a max number based on actual size of layers rather than a hardcoded constant?
    n_features = 1024
    feature_nums.update(random.sample(range(0,n_features),args.random_features))
    feature_nums.update(args.features)

    #encoders is the set of names of autoencoders - the layer (e.g. p6), followed by the file name
    autoencoders = set()
    #Load autoencoder features and make a dict
    ae_features = {}
    for n in range(len(args.sae_paths)):
        ae_name = args.sae_layers[n] + ''.join(filter(str.isalnum,args.sae_paths[n]))
        autoencoders.add(ae_name)

        #store an autoencoder's offset bias and features in a dict
        ae_features[ae_name]={}
        if os.path.exists(args.sae_paths[n]):
            #load state_dict
            sd = torch.load(p, map_location=torch.device(device))
            
            ae_features[ae_name]['offset'] = sd['_post_decoder_bias._bias_reference']
            #get the desired features
            #TODO: make compatible with convolutional layers
            for m in feature_nums:
                if m < model.features(args.sae_layers[n]):
                    ae_features[ae_name][m] = sd['_decoder._weight'][:,m]

        else:
            #construct one-hot features at the index given by the feature numbers
            #TODO: make compatible with convolutional layers
            ae_features[ae_name]['offset'] = torch.zeros(model.features(args.sae_layers[n])).to(device)
            for m in feature_nums:
                ae_features[ae_name][m] = one_hot(model.features(args.sae_layers[n]), m, device)



    #TODO: pass that dict of lists of vectors to test_write_trace, along with a list of keys

    #call test_write_trace to do the thing
    test_write_trace(game_config, model, args.test_episodes, device, autoencoders, ae_features, feature_nums, use_pb=True)