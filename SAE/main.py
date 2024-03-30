from classes import (ModifiedActivationStore, 
                     TiedCoder
)
from pathlib import Path 
import torch


if __name__== "__main__": 
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--activations_dir", 
    #     type=str, 
    #     help="Name of directory with stored activations"
    # )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'    
    run_dir = Path('../EZero/proj/6')
    store = ModifiedActivationStore()  

    # Create a Path object representing the directory
    # run_dir_path = Path(run_dir)

    # List all .pt files in the directory
    files = run_dir.glob("*.pt")
    
    # Iterate over the .pt files
    for case_file in files:
        temp_activations = torch.load(case_file)
        for activation in temp_activations:
            store.append(activation)
    store.shuffle()
    store.to_device(device)
    
    expansion_ratio = 8
    n_input_features = store.activation_size() # number of input activations / neurons 
    
    
    
#     train_autoencoder(
#     activation_store: ActivationStore,
#     autoencoder:SparseAutoencoder,
#     optimizer:AdamWithReset,
#     lx_coefficient,
#     sparsity_exponent,
#     bias_reg_coeff,
#     semiortho_coeff,
#     log_interval,
#     device,
#     train_batch_size: int,
#     metrics,
#     previous_steps,
#     max_steps,
#     use_ghost_grads,
#     window,
#     ghost_scale = 1e1,
# )