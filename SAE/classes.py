from typing import final
import torch 
from sparse_autoencoder.loss.abstract_loss import AbstractLoss, LossLogType, LossReductionType
from sparse_autoencoder.tensor_types import (
    InputOutputActivationBatch,
    ItemTensor,
    LearnedActivationBatch,
    TrainBatchStatistic,
)
from sparse_autoencoder.autoencoder.components import TiedBias, TiedBiasPosition
from torch.nn.parameter import Parameter
from sparse_autoencoder.activation_store.base_store import ActivationStore


@final
class LearnedActivationsLXLoss(AbstractLoss):
    """Learned activations L(X) (absolute error) loss.

    Loss that accepts an exponent, so it can be pointier than L1 loss.
    """


    def log_name(self) -> str:
        """Log name.

        Returns:
            Name of the loss module for logging.
        """
        return "learned_activations_lx_loss_penalty"

    def __init__(self, lx_coefficient: float, exponent=0.8) -> None:
        """Initialize the absolute error loss.

        Args:
            lx_coefficient: scale coefficient of this term in the loss. Expect it should be around 0.001
            exponent: the exponent of the norm
        """
        self.lx_coefficient = lx_coefficient
        super().__init__()
        self.exponent = exponent


    def _lx_loss(
        self,
        source_activations: InputOutputActivationBatch,  # noqa: ARG002
        learned_activations: LearnedActivationBatch,
        decoded_activations: InputOutputActivationBatch,  # noqa: ARG002
    ) -> tuple[TrainBatchStatistic, TrainBatchStatistic]:
        """Learned activations Lx (absolute error) loss.

        Args:
            source_activations: Source activations (input activations to the autoencoder from the
                source model).
            learned_activations: Learned activations (intermediate activations in the autoencoder).
            decoded_activations: Decoded activations.

        Returns:
            Tuple of itemwise absolute loss, and itemwise absolute loss multiplied by the lx
            coefficient.
        """
        absolute_loss = (torch.abs(learned_activations)**self.exponent).sum(dim=-1)
        absolute_loss_penalty = absolute_loss * self.lx_coefficient
        return absolute_loss, absolute_loss_penalty

    def forward(
        self,
        source_activations: InputOutputActivationBatch,
        learned_activations: LearnedActivationBatch,
        decoded_activations: InputOutputActivationBatch,
    ) -> TrainBatchStatistic:
        """Learned activations Lx (absolute error) loss.

        Args:
            source_activations: Source activations (input activations to the autoencoder from the
                source model).
            learned_activations: Learned activations (intermediate activations in the autoencoder).
            decoded_activations: Decoded activations.

        Returns:
            Loss per batch item.
        """
        return self._lx_loss(source_activations, learned_activations, decoded_activations)[1]

    # Override to add both the loss and the penalty to the log
    def batch_scalar_loss_with_log(
        self,
        source_activations: InputOutputActivationBatch,
        learned_activations: LearnedActivationBatch,
        decoded_activations: InputOutputActivationBatch,
        reduction: LossReductionType = LossReductionType.MEAN,
    ) -> tuple[ItemTensor, LossLogType]:
        """Learned activations L1 (absolute error) loss, with log.

        Args:
            source_activations: Source activations (input activations to the autoencoder from the
                source model).
            learned_activations: Learned activations (intermediate activations in the autoencoder).
            decoded_activations: Decoded activations.
            reduction: Loss reduction type. Typically you would choose LossReductionType.MEAN to
                make the loss independent of the batch size.

        Returns:
            Tuple of the L1 absolute error batch scalar loss and a dict of the properties to log
                (loss before and after the l1 coefficient).
        """
        absolute_loss, absolute_loss_penalty = self._lx_loss(
            source_activations, learned_activations, decoded_activations
        )

        match reduction:
            case LossReductionType.MEAN:
                batch_scalar_loss = absolute_loss.mean().squeeze()
                batch_scalar_loss_penalty = absolute_loss_penalty.mean().squeeze()
            case LossReductionType.SUM:
                batch_scalar_loss = absolute_loss.sum().squeeze()
                batch_scalar_loss_penalty = absolute_loss_penalty.sum().squeeze()

        metrics = {
            "learned_activations_lx_loss": batch_scalar_loss.item(),
            self.log_name(): batch_scalar_loss_penalty.item(),
        }

        return batch_scalar_loss_penalty, metrics

    def extra_repr(self) -> str:
        """Extra representation string."""
        return f"lx_coefficient={self.l1_coefficient}, exponent={self.exponent}"

# compute a regularization function for the coder that pushes weight to be semi-orthogonal
# do this by computing the (Frobenius) L2 norm of (W*W-I)

def semiorthogonality_l2(weight):
    #multiply the weight by its transpose
    diff = torch.einsum('oi,oj->ij', weight, weight)
    #subtract identity matrix
    for i in range(diff.size()[0]):
        diff[i,i] -= 1
    return torch.sum(diff**2)

#fancy nonlinearity for TiedCoder encoder - rather than shifting by bias, we're just cutting off everything below bias
#uses algebraic / numerically stable version of softplus and sigmoid
def SoftCutoff(x, scale, bias):
    #simultaneous softplus and cutoff
    y = (x+torch.sqrt(x**2+scale**2)) * (1+(x+bias)/torch.sqrt((x+bias)**2+scale**2/4)) / 4
    return y

#Define a encoder/decoder module with _bias and _weight tensors
from torch.nn.parameter import Parameter
class TiedCoder(torch.nn.Module):
    def __init__(self, bias, weight):
        super().__init__()
        self._bias = Parameter(bias)
        self._weight = Parameter(weight)
        self._scale = torch.std(precomputed_activations).item()/8


    def forward(self, x, position):
        if position == 'encoder':
            #Original version just returned torch.nn.Softplus(beta=2*int(1/self._scale))(torch.einsum('bi,oi->bo', x, self._weight) + self._bias)
            #But upgraded version applies a more complicated nonlinearity: don't subtract a bias, instead use the bias as a cutoff at which to multiply by a sigmoid
            return SoftCutoff(torch.einsum('bi,oi->bo', x, self._weight), self._scale, self._bias)
        elif position == 'decoder':
            return torch.einsum('oi,bo->bi',self._weight, x)
        else:
            raise ValueError('TiedCoder should be encoder or decoder')

#TODO: Define an autoencoder with tied weights that uses the same variable names in state dict
class TiedAutoencoder(torch.nn.Module):
    """Autoencoder with tied bias offset and tied input/output weights"""
    def __init__(self, input_features, learnt_features, geometric_median_dataset=None, n_components=None):
        super().__init__()

        self._learnt_features = learnt_features
        self.n_learned_features = learnt_features
        self._input_features = input_features
        self._n_components = n_components

        # Store the geometric median of the dataset (so that we can reset parameters). This is not a
        # parameter itself (the tied bias parameter is used for that), so gradients are disabled.
        if geometric_median_dataset is not None:
            self.geometric_median_dataset = geometric_median_dataset.clone()
            self.geometric_median_dataset.requires_grad = False
        else:
            self.geometric_median_dataset = torch.zeros(input_features)
            self.geometric_median_dataset.requires_grad = False

        # Initialize the tied bias
        self.tied_bias = Parameter(torch.empty(input_features))
        self.initialize_tied_parameters()

        # Initialize the tied encoder/decoder params
        self.enc_bias = torch.zeros(self._learnt_features)
        self.enc_weight = torch.zeros([self._learnt_features, self._input_features])
        torch.nn.init.kaiming_normal_(self.enc_weight, nonlinearity='relu')

        # Initialize the components
        self._pre_encoder_bias = TiedBias(self.tied_bias, TiedBiasPosition.PRE_ENCODER)
        self._encoder = TiedCoder(self.enc_bias, self.enc_weight)
        self._post_decoder_bias = TiedBias(self.tied_bias, TiedBiasPosition.POST_DECODER)


    def forward(self, x):
        x = self._pre_encoder_bias(x)
        learned_activations = self._encoder(x, 'encoder')
        x = self._encoder(learned_activations, 'decoder')
        decoded_activations = self._post_decoder_bias(x)

        return learned_activations, decoded_activations

    def update_dictionary_vectors(self, dictionary_vector_indices, updated_dictionary_weights, component_idx):
        if dictionary_vector_indices.numel() == 0:
            return

        with torch.no_grad():
            if component_idx is None:
                if self._n_components is not None:
                    error_message = "component_idx must be specified when n_components is not None"
                    raise ValueError(error_message)

                self.weight[dictionary_vector_indices] = updated_dictionary_weights
            else:
                self.weight[
                    component_idx, dictionary_vector_indices[component_idx]
                ] = updated_dictionary_weights[component_idx]

    def update_bias(self, update_parameter_indices, updated_bias_features, component_idx):
        if update_parameter_indices.numel() == 0:
            return

        with torch.no_grad():
            if component_idx is None:
                if self._n_components is not None:
                    error_message = "component_idx must be specified when n_components is not None"
                    raise ValueError(error_message)

                self.bias[update_parameter_indices] = updated_bias_features
            else:
                self.bias[component_idx, update_parameter_indices] = updated_bias_features

    def initialize_tied_parameters(self) -> None:
        """Initialize the tied parameters."""
        # The tied bias is initialised as the geometric median of the dataset
        self.tied_bias.data = self.geometric_median_dataset

    def reset_parameters(self) -> None:
        """Reset the parameters."""
        self.initialize_tied_parameters()
        for module in self.network:
            if "reset_parameters" in dir(module):
                module.reset_parameters()

    def pre_encoder_bias(self):
        return self._pre_encoder_bias
    def sae_dim(self):
        return self.n_learned_features
    def encoder(self):
        return self._encoder

    def post_decoder_bias(self):
        return self._post_decoder_bias
    def input_dim(self):
      return self._input_features


class ModifiedActivationStore(ActivationStore):
  def __init__(self):
    super().__init__()
    self._data = [] # we just store in a list

  def append(self, item) -> None:
      self._data.append(item)

  def extend(self, batch):
      self._data.extend(batch)

  def empty(self):
      self._data = []

  def __getitem__(self, index: int):
      return self._data[index]

  def __len__(self) -> int:
      return len(self._data)

  def shuffle(self) -> None:
    perm = torch.randperm(len(self._data))
    self._data = [self._data[i] for i in perm]

  def flatten(self) -> None:
    activations = self._data
    self._data = []
    for item in activations:
      self._data.extend(item)

  def to_device(self, device) -> None: 
    self._data = [activation.to(device) for activation in self._data]
  
  def activation_size(self) -> int: 
      return len(self._data[0])
