import os
import torch

# from sparse_autoencoder import (SparseAutoencoder, AdamWithReset)
from sparse_autoencoder.activation_store.base_store import ActivationStore
# from sparse_autoencoder.train.sweep_config import SweepParametersRuntime
# from sparse_autoencoder.metrics.metrics_container import MetricsContainer
# from sparse_autoencoder.loss.learned_activations_l1 import LearnedActivationsL1Loss
# from sparse_autoencoder.loss.decoded_activations_l2 import L2ReconstructionLoss
# from sparse_autoencoder.loss.reducer import LossReducer
# from sparse_autoencoder.metrics.train.l0_norm_metric import TrainBatchLearnedActivationsL0
# from sparse_autoencoder.metrics.train.abstract_train_metric import TrainMetricData
# from sparse_autoencoder.loss.abstract_loss import AbstractLoss, LossLogType, LossReductionType
# from sparse_autoencoder.tensor_types import (
#     InputOutputActivationBatch,
#     ItemTensor,
#     LearnedActivationBatch,
#     TrainBatchStatistic,
# )
# from sparse_autoencoder import Pipeline

# from torch.utils.data import DataLoader
# from typing import final

from classes import (
    ModifiedActivationStore, 
    LearnedActivationsLXLoss, 
    semiorthogonality_l2, 
    SoftCutoff, 
    TiedCoder
    )


def train_autoencoder(
    activation_store: ActivationStore,
    autoencoder:SparseAutoencoder,
    optimizer:AdamWithReset,
    lx_coefficient,
    sparsity_exponent,
    bias_reg_coeff,
    semiortho_coeff,
    log_interval,
    device,
    train_batch_size: int,
    metrics,
    previous_steps:int = 0,
    max_steps:int = 25000,
    use_ghost_grads=True,
    window=150,
    ghost_scale = 1e1,
):
    """Train the sparse autoencoder.

    Args:
        activation_store: Activation store from the generate section.
        train_batch_size: Train batch size.

    Returns:
        Number of times each neuron fired.
    """
    total_training_steps = previous_steps
    autoencoder_device = device

    activations_dataloader = DataLoader(
        activation_store,
        batch_size=train_batch_size,
    )

    loss = LossReducer(
        L2ReconstructionLoss(),
        LearnedActivationsLXLoss(lx_coefficient, sparsity_exponent),
    )

    learned_activations_fired_count = torch.zeros(
        autoencoder.n_learned_features, dtype=torch.int32, device=autoencoder_device
    )
    memory = 150

    last_m_fired_count = []

    #----------------------------------

    #initialize a tensor of same size as the hidden layer to track firing
    n_forward_pass_since_fired = torch.zeros(
        autoencoder._encoder._weight.size()[0] , device=autoencoder_device
    )

    #--------------------------------


    while total_training_steps < previous_steps + max_steps:
        # Zero the gradients
        optimizer.zero_grad()

        # Move the batch to the device (in place)
        store_batch = next(iter(activations_dataloader))
        batch = store_batch.detach().to(autoencoder_device)

        # Forward pass
        learned_activations, reconstructed_activations = autoencoder(batch)

        #compute loss
        total_loss, loss_metrics = loss.batch_scalar_loss_with_log(
            batch, learned_activations, reconstructed_activations
            )

# --------------------------------------------------------------------------------------------------------

        #track firing
        n_forward_pass_since_fired += train_batch_size

        non_zero_activations = (learned_activations != 0).float()
        neurons_fired = non_zero_activations.sum(dim=0)
        neurons_fired_indicator = (neurons_fired == 0)

        n_forward_pass_since_fired *= neurons_fired_indicator

        x_centred = batch - batch.mean(dim=0, keepdim=True)
        mse_loss = (
            torch.pow((reconstructed_activations - batch.float()), 2)
            / (x_centred**2).sum(dim=-1, keepdim=True).sqrt()
        )

        dead_neuron_mask = (n_forward_pass_since_fired > window).detach()

        if use_ghost_grads and dead_neuron_mask.sum() > 0:
          assert dead_neuron_mask is not None

          # ghost protocol

          # 1.
          residual = batch - reconstructed_activations
          residual_centred = residual - residual.mean(dim=0, keepdim=True)
          l2_norm_residual = torch.norm(residual, dim=-1)

          # 2.
          feature_acts_dead_neurons_only = torch.einsum('bi,oi->bo', batch, autoencoder._encoder._weight)*dead_neuron_mask
          feature_acts_dead_neurons_only = (feature_acts_dead_neurons_only+torch.sqrt(feature_acts_dead_neurons_only**2+scale**2))/2
          ghost_out = autoencoder._decoder.forward(feature_acts_dead_neurons_only)
          l2_norm_ghost_out = torch.norm(ghost_out, dim=-1)
          norm_scaling_factor = (l2_norm_residual / (1e-6 + l2_norm_ghost_out)).detach()
          ghost_out = ghost_out * norm_scaling_factor.unsqueeze(1)

          # 3.
          mse_loss_ghost_resid = (
              torch.pow((ghost_out - residual.detach().float()), 2)
              / (residual_centred.detach() ** 2).sum(dim=-1, keepdim=True).sqrt()
          )
          mse_rescaling_factor = ghost_scale*(mse_loss / (mse_loss_ghost_resid + 1e-6)).detach()
          mse_loss_ghost_resid = mse_rescaling_factor * mse_loss_ghost_resid

          mse_loss_ghost_resid = mse_loss_ghost_resid.mean()
          total_loss += mse_loss_ghost_resid

# -------------------------------------------------------------------------------------------------

        #add regularization
        total_loss += torch.sum(autoencoder._encoder._bias**2)*bias_reg_coeff

        #don't use semiorthogonality loss for untied autoencoder
        #semiortho_reg = semiorthogonality_l2(autoencoder._encoder._weight)*semiortho_coeff
        #total_loss += semiortho_reg

        # Store count of how many neurons have fired
        with torch.no_grad():
            fired = learned_activations > -autoencoder._encoder._bias
            learned_activations_fired_count.add_(fired.sum(dim=0))

            last_m_fired_count.append(fired.sum(dim=0))
            if len(last_m_fired_count)>memory:
                last_m_fired_count.pop(0)
            distinct_fired = last_m_fired_count[0]>0
            for c in last_m_fired_count:
                distinct_fired = (distinct_fired + c>0)>0


        # Backwards pass
        total_loss.backward()
        optimizer.step()

        # Log training metrics
        if wandb.run is not None and total_training_steps % log_interval == 0:
          wandb.log(
              data = {'l2 reconstruction loss': loss_metrics['l2_reconstruction_loss'],
                      f'learned activations l{sparsity_exponent} loss': loss_metrics['learned_activations_lx_loss_penalty'],
                      'active neurons': torch.sum(fired).item()/fired.size()[0],
                      #'regularization on semiorthogonality': semiortho_reg.item(),
                      f'active in last {memory}': torch.sum(distinct_fired).item()
                      }, step=total_training_steps, commit=True
          )
        total_training_steps += 1
    return total_training_steps, learned_activations_fired_count

    
