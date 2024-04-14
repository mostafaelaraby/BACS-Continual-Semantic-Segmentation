import torch
import torch.nn as nn
import torch.nn.functional as F


class classification_head(nn.Module):
    def __init__(self, feat_dim, num_classes, stop_gradients=False) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.conv = nn.Conv2d(self.feat_dim, num_classes, 1)
        self.norm = nn.Sigmoid()
        self.stop_gradients = stop_gradients
        self.upsampling_layer = nn.Sequential(
            nn.Upsample(scale_factor=16, mode="bilinear", align_corners=True),
        )

    def get_distance(self, x, prototype):
        """Returns distance between representations and prototype saved

        Args:
            x (Tensor): input image representation
            prototype (Tensor): saved foreground representation

        Returns:
            Tensor: output distance between prototype and representation
        """
        # first apply sigmoid to the prototype and out input penultimate value
        if self.stop_gradients:
            x = x.detach()
            prototype = prototype.detach()
        x = self.norm(x)
        prototype = self.norm(prototype)
        return torch.abs(x - prototype)

    def predict(self, x, prototype):
        # x, prototype = batch
        prototype = prototype.view(1, self.feat_dim, 1, 1)
        distance = self.get_distance(x, prototype)
        output = self.conv(distance)
        return self.upsampling_layer(output)

    def forward(self, prototype, x):
        return self.predict(x, prototype)


class BgDetector(nn.Module):
    def __init__(self, in_channels: int) -> None:
        """Background Detector with siamese like architecture to detect bg shift

        Args:
            in_channels (int): number of input channels
        """
        super(BgDetector, self).__init__()
        self.stop_gradients = False
        self.inter_channels = in_channels // 4
        layers = [
            nn.Conv2d(in_channels, self.inter_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.inter_channels),
            nn.ReLU(),
            nn.Dropout(0.1),
        ]
        self.base_layers = nn.Sequential(*layers)
        self.seen_not_seen_clf = None

    def set_stop_gradients(self, stop_grads):
        if self.seen_not_seen_clf is None or stop_grads == self.stop_gradients:
            return
        self.stop_gradients = stop_grads
        if isinstance(self.seen_not_seen_clf, nn.ModuleList):
            for layer in self.seen_not_seen_clf:
                layer.stop_gradients = stop_grads
        else:
            self.seen_not_seen_clf.stop_gradients = stop_grads

    def get_classification_head(self, num_classes):
        """Returns classification head of the background head

        Args:
            num_classes (int):  number of classes

        Returns:
            nn.Module: Convolutional output layer for our bg detector
        """
        return classification_head(
            self.inter_channels, num_classes, stop_gradients=self.stop_gradients
        )

    def get_penultimate_output(self, x):
        """Override to return penultimate layre output for input image

        Args:
            x (tensor): input image
        """
        return self.base_layers(x)

    def get_penultimate_layer_dim(self):
        """Override to return dimensions of the used penultimate layer"""
        return self.inter_channels

    def get_seen_map_task(self, penultimate_output, prototype, task_num):
        """Returns seen map to specific task prototype

        Args:
            penultimate_output (Tensor): output representations
            prototype (Tensor): saved task prototype
            task_num (int): task number in case of multi-head BG detector

        Returns:
            Tensor: Output free logits
        """
        if isinstance(self.seen_not_seen_clf, nn.ModuleList):
            x = self.seen_not_seen_clf[task_num].predict(
                penultimate_output, prototype[task_num]
            )
        else:
            x = self.seen_not_seen_clf.predict(penultimate_output, prototype[task_num])
        return x

    def forward_seen_before(self, x, prototypes):
        """Forward seen before on all saved prototypes

        Args:
            x (Tensor): input representation of the image
            prototypes (Tensor): Saved prototypes of previous tasks

        Returns:
            Tensor: Concatenated map of per task seen
        """
        if isinstance(self.seen_not_seen_clf, nn.ModuleList):
            n_heads = prototypes.shape[0]
            task_out = []
            if n_heads == 1:
                return self.seen_not_seen_clf[0].predict(x, prototypes[0])
            for i in range(n_heads):
                task_out.append(self.seen_not_seen_clf[i].predict(x, prototypes[i]))
            x = torch.cat(task_out, dim=1)
        else:
            x = self.seen_not_seen_clf(x, prototypes[0])
        return x

    def get_seen_probs(self, x, prototypes, bg_detect=False):
        """Returns seen/unseen probabilities on all tasks

        Args:
            x (Tensor): input representations
            prototypes (Tensor): saved prototypes
            bg_detect (bool, optional): Bg detect enabled to detect shift in bg for prev tasks only (employed in our loss). Defaults to False.

        Returns:
            Tensor: Seen 1, unseen 0 in terms of sigmoid probabilities per pixel
        """
        if bg_detect:
            # put in eval mode
            revert_to_train = False
            if self.training:
                self.eval()
                revert_to_train = True
            # skip prototype of last task or not
            seen_before_logits = self.forward_seen_before(x, prototypes)
            if revert_to_train:
                self.train()
        else:
            seen_before_logits = self.forward_seen_before(x, prototypes)
        seen_prob = seen_before_logits.sigmoid()
        return seen_prob
