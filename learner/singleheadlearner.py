from .multiheadlearner import MultiHeadLearner
from networks.base_network import BaseNetwork


class SingleHeadLearner(MultiHeadLearner):
    def __init__(self, network: BaseNetwork, config, is_domain_shift):
        super().__init__(network, config, is_domain_shift)

    def on_init_start(self, trainer):
        """Called on trainer init before init of network

        Args:
            trainer (pl.Trainer): Pytorch lightning trainer
        """
        num_classes = self.config.dataset.dataset.num_classes
        # initialize networks' classifier as a single head with total classes
        self.network.classifier_head = self.network.get_classification_head(num_classes)
