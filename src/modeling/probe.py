import copy
from dataclasses import dataclass

from torch import Tensor
from torch.nn import Linear, Module, ModuleDict, ModuleList
from torchmetrics import Metric
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError

from src.modeling.mixins import ModelPersistenceMixin


@dataclass
class ProbeConfig:
    n_layers: int
    """number of layers in the probed model"""
    modes: list[str]
    """ list of mode/phase names representing different operational modes"""
    block_model: type[Module] = Linear(in_features=768, out_features=1, bias=False)
    """block model type that will be used at each layer/mode combination"""
    out_features: int = 1
    """size of each output sample (1=Regression, 2+=Classification)"""
    in_features: int = 768
    """size of each input sample"""


class LayerBlock(Module):
    """
    A container for models associated with a single layer index.

    Each `LayerBlock` corresponds to one layer in the transformer model and contains a dictionary
    of models, one for each mode/phase.
    """

    def __init__(self, layer_idx: int, config: ProbeConfig):
        """
        Initializes a LayerBlock for a specific layer in the probe.

        Args:
            layer_idx (int): The index of the layer this block represents.
            config (ProbeConfig): The configuration specifying the number of modes/phases and model dimensions.
        """
        super().__init__()
        self.layer_idx = layer_idx  # stores the layer index for reference

        self.modes = ModuleDict(
            {mode: copy.deepcopy(config.block_model) for mode in config.modes}
        )

    def forward(self, x, mode):
        y = self.modes[mode](x)
        return y


LAYER_MODE_HIDDEN_STATES_DICT = dict[tuple[int, str], Tensor]


class Probe(Module, ModelPersistenceMixin):
    """
    Organizes a collection of LayerBlock modules across multiple layers.

    This allows training and evaluation of multiple linear models, with one collection per
    transformer hidden state layer, and each collection containing models for different modes.

    The probe can be indexed using `probe[layer,mode]` syntax.

    Trained probes can be loaded/saved using `save_pretrained` and `from_pretrained` methods
    provided by the `ModelPersistenceMixin`.
    """

    def __init__(self, config: ProbeConfig):
        super().__init__()
        self.config = config
        self.layers = ModuleList(
            [LayerBlock(layer_idx=i, config=config) for i in range(config.n_layers)]
        )

    def forward(self, x, layer: int, mode: str):
        y = self.layers[layer](x, mode)
        return y

    def full_forward(
        self, x: LAYER_MODE_HIDDEN_STATES_DICT
    ) -> LAYER_MODE_HIDDEN_STATES_DICT:
        """
        Performs a forward pass on all tensors in x. Keys in x must be a
        tuple (layer,mode) and must be in valid according to this probe."""
        results = {(L, M): self.forward(x[(L, M)], L, M) for (L, M) in x.keys()}
        return results

    def __getitem__(self, key):
        """Allows indexing into the probe using `probe[layer, phase]` syntax."""
        if isinstance(key, tuple) and len(key) == 2:
            layer, mode = key
            return self.layers[layer].modes[mode]
        else:
            raise KeyError("Index must be a tuple of (layer, mode)")

    def train(self, train: bool = True):
        self.requires_grad_(train)
        return super().train(train)

    def items(self):
        return iter(self)

    def values(self):
        for _, v in iter(self):
            yield v

    def keys(self):
        for k, _ in iter(self):
            yield k

    def __iter__(self):
        for layer in range(self.config.n_layers):
            for mode in self.config.modes:
                yield (layer, mode), self.layers[layer].modes[mode]

    @property
    def device(self):
        return next(self.parameters()).device


class LossTracker:

    regression_metrics: dict[tuple[int, str], list[tuple[str, Metric]]]
    classification_metrics: dict[tuple[int, str], list[tuple[str, Metric]]]
    counters: dict[tuple[int, str], int]
    losses: dict[tuple[int, str], float]
    total_count: int = 0
    total_loss: float = 0.0
    validation: bool = False
    _loss_per_token: dict = None

    def __init__(
        self,
        n_layers: int,
        modes: list[str],
        validation: bool = False,
        device: str = "cuda",
    ):
        self.n_layers = n_layers
        self.modes = modes
        self.validation = validation
        self.device = device
        self.zeroize()

    @classmethod
    def from_probe(cls, probe: Probe, validation: bool = False):
        return cls(probe.config.n_layers, probe.config.modes, validation, probe.device)

    def full_increment(
        self,
        loss_fn,
        outputs: LAYER_MODE_HIDDEN_STATES_DICT,
        labels: LAYER_MODE_HIDDEN_STATES_DICT,
        skip=0,
    ):
        for k in labels.keys():
            yhat = outputs[k][skip:]
            y = labels[k][skip:]
            loss = loss_fn(yhat, y)

            if not self.validation:
                loss.backward()
            L, M = k

            self.increment(L, M, y.shape[0], loss.item(), y, yhat)
        self._loss_per_token = None

    def increment(self, layer, mode, n_samples, loss_value, y, yhat):
        self.counters[(layer, mode)] += n_samples
        self.total_count += n_samples
        self.losses[(layer, mode)] += loss_value
        self.total_loss += loss_value
        self._loss_per_token = None
        for _, metric in self.classification_metrics[(layer, mode)]:
            metric(yhat, y)

        for _, metric in self.regression_metrics[(layer, mode)]:
            nbins = yhat.shape[-1]
            preds = yhat.argmax(-1) / nbins
            metric(preds, y / nbins)

    def zeroize(self):
        self.total_loss = 0.0
        self.total_count = 0
        self.losses = {(L, M): 0.0 for L in range(self.n_layers) for M in self.modes}
        self.counters = {(L, M): 0 for L in range(self.n_layers) for M in self.modes}
        self._loss_per_token = None

        self.classification_metrics = {
            (L, M): [
                ("acc", Accuracy("multiclass", num_classes=2).to(self.device)),
                ("F1", F1Score("multiclass", num_classes=2).to(self.device)),
                ("precision", Precision("multiclass", num_classes=2).to(self.device)),
                ("recall", Recall("multiclass", num_classes=2).to(self.device)),
            ]
            for L in range(self.n_layers)
            for M in self.modes
        }

        self.regression_metrics = {
            (L, M): [
                ("MSE", MeanSquaredError(squared=False).to(self.device)),
            ]
            for L in range(self.n_layers)
            for M in self.modes
        }

    def get_loss_per_token(self):
        """Returns the mean loss per token by-mode and by-layer, but only if the counter is >0.
        WARNING: the output may NOT contain all combinations of (layer, mode)."""

        if self._loss_per_token:
            return self._loss_per_token

        loss_per_token = {
            (layer, mode): (value / self.counters[(layer, mode)])
            for (layer, mode), value in self.losses.items()
        }
        self._loss_per_token = loss_per_token
        return loss_per_token

    def compute_eval_metrics(self):
        classification_metrics = {
            (layer, mode, desc): metric.compute().item()
            for (layer, mode), metric_list in self.classification_metrics.items()
            for (desc, metric) in metric_list
        }

        regression_metrics = {
            (layer, mode, desc): metric.compute().item()
            for (layer, mode), metric_list in self.regression_metrics.items()
            for (desc, metric) in metric_list
        }

        metrics = {}
        metrics.update(classification_metrics)
        metrics.update(regression_metrics)
        return metrics
