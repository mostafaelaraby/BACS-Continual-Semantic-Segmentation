"""Code adapted from https://github.com/arthurdouillard/CVPR2021_PLOP
"""
import torch
from torch.nn import functional as F
import math
from torch import nn


def bce(x, y):
    return -(y * torch.log(x + 1e-6) + (1 - y) * torch.log((1 - x) + 1e-6))


def _global_pod(x, spp_scales=[2, 4, 8], normalize=False):
    b = x.shape[0]
    w = x.shape[-1]

    emb = []
    for scale in spp_scales:
        tensor = F.avg_pool2d(x, kernel_size=w // scale)
        horizontal_pool = tensor.sum(dim=2).view(b, -1)
        vertical_pool = tensor.sum(dim=3).view(b, -1)

        if normalize:
            horizontal_pool = F.normalize(horizontal_pool, dim=1, p=2)
            vertical_pool = F.normalize(vertical_pool, dim=1, p=2)

        tensor_pool = torch.cat([horizontal_pool, vertical_pool], dim=-1)

        emb.append(tensor_pool)

    return torch.cat(emb, dim=1)


def _local_pod_masked(
    x, mask, spp_scales=[1, 2, 4], normalize=False, normalize_per_scale=False
):
    b = x.shape[0]
    c = x.shape[1]
    w = x.shape[-1]
    emb = []

    mask = mask[:, None].repeat(1, c, 1, 1)
    x[mask] = 0.0

    for scale in spp_scales:
        k = w // scale

        nb_regions = scale ** 2

        for i in range(scale):
            for j in range(scale):
                tensor = x[..., i * k : (i + 1) * k, j * k : (j + 1) * k]

                horizontal_pool = tensor.mean(dim=3).view(b, -1)
                vertical_pool = tensor.mean(dim=2).view(b, -1)

                if normalize_per_scale is True:
                    horizontal_pool = horizontal_pool / nb_regions
                    vertical_pool = vertical_pool / nb_regions
                if normalize:
                    horizontal_pool = F.normalize(horizontal_pool, dim=1, p=2)
                    vertical_pool = F.normalize(vertical_pool, dim=1, p=2)

                emb.append(horizontal_pool)
                emb.append(vertical_pool)

    return torch.cat(emb, dim=1)


def _local_pod(x, spp_scales=[1, 2, 4], normalize=False, normalize_per_scale=False):
    b = x.shape[0]
    w = x.shape[-1]
    emb = []

    for scale_index, scale in enumerate(spp_scales):
        k = w // scale

        nb_regions = scale ** 2

        for i in range(scale):
            for j in range(scale):
                tensor = x[..., i * k : (i + 1) * k, j * k : (j + 1) * k]

                horizontal_pool = tensor.mean(dim=3).view(b, -1)
                vertical_pool = tensor.mean(dim=2).view(b, -1)

                if normalize_per_scale is True:
                    horizontal_pool = horizontal_pool / nb_regions
                    vertical_pool = vertical_pool / nb_regions
                elif normalize_per_scale == "spm":
                    if scale_index == 0:
                        factor = 2 ** (len(spp_scales) - 1)
                    else:
                        factor = 2 ** (len(spp_scales) - scale_index)
                    horizontal_pool = horizontal_pool / factor
                    vertical_pool = vertical_pool / factor

                if normalize:
                    horizontal_pool = F.normalize(horizontal_pool, dim=1, p=2)
                    vertical_pool = F.normalize(vertical_pool, dim=1, p=2)

                emb.append(horizontal_pool)
                emb.append(vertical_pool)

    return torch.cat(emb, dim=1)


def features_distillation(
    list_attentions_a,
    list_attentions_b,
    collapse_channels="spatial",
    normalize=True,
    labels=None,
    index_new_class=None,
    pod_deeplab_mask=False,
    pod_deeplab_mask_factor=None,
    pod_factor=1.0,
    prepro="pow",
    deeplabmask_upscale=True,
    spp_scales=[1, 2, 4],
    pod_options=None,
    outputs_old=None,
    use_pod_schedule=True,
    nb_current_classes=-1,
    nb_new_classes=-1,
):
    """A mega-function comprising several features-based distillation.

    :param list_attentions_a: A list of attention maps, each of shape (b, n, w, h).
    :param list_attentions_b: A list of attention maps, each of shape (b, n, w, h).
    :param collapse_channels: How to pool the channels.
    :param memory_flags: Integer flags denoting exemplars.
    :param only_old: Only apply loss to exemplars.
    :return: A float scalar loss.
    """

    assert len(list_attentions_a) == len(list_attentions_b)

    if pod_deeplab_mask_factor is None:
        pod_deeplab_mask_factor = pod_factor

    # if collapse_channels in ("spatial_tuple", "spp", "spp_noNorm", "spatial_noNorm"):
    normalize = False

    apply_mask = "background"
    upscale_mask_topk = 1
    mask_position = "last"  # Others choices "all" "backbone"
    use_adaptative_factor = False
    mix_new_old = None

    loss = torch.tensor(0.0).type_as(list_attentions_a[0])
    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        adaptative_pod_factor = 1.0
        difference_function = "frobenius"
        pool = True
        use_adaptative_factor = False
        handle_extra_channels = "sum"
        normalize_per_scale = False

        if pod_options and pod_options.get("switch"):
            if i < len(list_attentions_a) - 1:
                if "before" in pod_options["switch"]:
                    collapse_channels = pod_options["switch"]["before"].get(
                        "type", collapse_channels
                    )
                    pod_factor = pod_options["switch"]["before"].get(
                        "factor", pod_factor
                    )
                    normalize = pod_options["switch"]["before"].get("norm", False)
                    prepro = pod_options["switch"]["before"].get("prepro", prepro)
                    use_adaptative_factor = pod_options["switch"]["before"].get(
                        "use_adaptative_factor", use_adaptative_factor
                    )
            else:
                if "after" in pod_options["switch"]:
                    collapse_channels = pod_options["switch"]["after"].get(
                        "type", collapse_channels
                    )
                    pod_factor = pod_options["switch"]["after"].get(
                        "factor", pod_factor
                    )
                    normalize = pod_options["switch"]["after"].get("norm", False)
                    prepro = pod_options["switch"]["after"].get("prepro", prepro)

                    apply_mask = pod_options["switch"]["after"].get(
                        "apply_mask", apply_mask
                    )
                    upscale_mask_topk = pod_options["switch"]["after"].get(
                        "upscale_mask_topk", upscale_mask_topk
                    )
                    use_adaptative_factor = pod_options["switch"]["after"].get(
                        "use_adaptative_factor", use_adaptative_factor
                    )
                    mix_new_old = pod_options["switch"]["after"].get(
                        "mix_new_old", mix_new_old
                    )

                    handle_extra_channels = pod_options["switch"]["after"].get(
                        "extra_channels", handle_extra_channels
                    )
                    spp_scales = pod_options["switch"]["after"].get(
                        "spp_scales", spp_scales
                    )
                    use_pod_schedule = pod_options["switch"]["after"].get(
                        "use_pod_schedule", use_pod_schedule
                    )

            mask_position = pod_options["switch"].get("mask_position", mask_position)
            normalize_per_scale = pod_options["switch"].get(
                "normalize_per_scale", normalize_per_scale
            )
            pool = pod_options.get("pool", pool)

        if a.shape[1] != b.shape[1]:
            assert i == len(list_attentions_a) - 1
            assert a.shape[0] == b.shape[0]
            assert a.shape[2] == b.shape[2]
            assert a.shape[3] == b.shape[3]

            assert handle_extra_channels in ("trim", "sum"), handle_extra_channels

            if handle_extra_channels == "sum":
                _b = torch.zeros_like(a).type_as(a)
                _b[:, 0] = b[:, 0] + b[:, index_new_class:].sum(dim=1)
                _b[:, 1:] = b[:, 1:index_new_class]
                b = _b
            elif handle_extra_channels == "trim":
                b = b[:, :index_new_class]

        # shape of (b, n, w, h)
        assert a.shape == b.shape, (a.shape, b.shape)

        if not pod_deeplab_mask and use_adaptative_factor:
            adaptative_pod_factor = (labels == 0).float().mean()

        if prepro == "pow":
            a = torch.pow(a, 2)
            b = torch.pow(b, 2)
        elif prepro == "none":
            pass
        elif prepro == "abs":
            a = torch.abs(a, 2)
            b = torch.abs(b, 2)
        elif prepro == "relu":
            a = torch.clamp(a, min=0.0)
            b = torch.clamp(b, min=0.0)

        if collapse_channels == "spatial":
            a_h = a.sum(dim=3).view(a.shape[0], -1)
            b_h = b.sum(dim=3).view(b.shape[0], -1)
            a_w = a.sum(dim=2).view(a.shape[0], -1)
            b_w = b.sum(dim=2).view(b.shape[0], -1)
            a = torch.cat([a_h, a_w], dim=-1)
            b = torch.cat([b_h, b_w], dim=-1)
        elif collapse_channels == "global":
            a = _global_pod(a, spp_scales, normalize=False)
            b = _global_pod(b, spp_scales, normalize=False)
        elif collapse_channels == "local":
            if pod_deeplab_mask and (
                (i == len(list_attentions_a) - 1 and mask_position == "last")
                or mask_position == "all"
            ):
                if pod_deeplab_mask_factor == 0.0:
                    continue

                pod_factor = pod_deeplab_mask_factor

                if apply_mask == "background":
                    mask = labels < index_new_class
                elif apply_mask == "old":
                    pseudo_labels = labels.clone()
                    mask_background = labels == 0
                    pseudo_labels[mask_background] = outputs_old.argmax(dim=1)[
                        mask_background
                    ]

                    mask = (labels < index_new_class) & (0 < pseudo_labels)
                else:
                    raise NotImplementedError(f"Unknown apply_mask={apply_mask}.")

                if deeplabmask_upscale:
                    a = F.interpolate(
                        torch.topk(a, k=upscale_mask_topk, dim=1)[0],
                        size=labels.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    b = F.interpolate(
                        torch.topk(b, k=upscale_mask_topk, dim=1)[0],
                        size=labels.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                else:
                    mask = F.interpolate(
                        mask[:, None].float(), size=a.shape[-2:]
                    ).bool()[:, 0]

                if use_adaptative_factor:
                    adaptative_pod_factor = mask.float().mean(dim=(1, 2))

                a = _local_pod_masked(
                    a,
                    mask,
                    spp_scales,
                    normalize=False,
                    normalize_per_scale=normalize_per_scale,
                )
                b = _local_pod_masked(
                    b,
                    mask,
                    spp_scales,
                    normalize=False,
                    normalize_per_scale=normalize_per_scale,
                )
            else:
                a = _local_pod(
                    a,
                    spp_scales,
                    normalize=False,
                    normalize_per_scale=normalize_per_scale,
                )
                b = _local_pod(
                    b,
                    spp_scales,
                    normalize=False,
                    normalize_per_scale=normalize_per_scale,
                )
        else:
            raise ValueError("Unknown method to collapse: {}".format(collapse_channels))

        if i == len(list_attentions_a) - 1 and pod_options is not None:
            if "difference_function" in pod_options:
                difference_function = pod_options["difference_function"]
        elif pod_options is not None:
            if "difference_function_all" in pod_options:
                difference_function = pod_options["difference_function_all"]

        if normalize:
            a = F.normalize(a, dim=1, p=2)
            b = F.normalize(b, dim=1, p=2)

        if difference_function == "frobenius":
            if isinstance(a, list):
                layer_loss = torch.tensor(
                    [torch.frobenius_norm(aa - bb, dim=-1) for aa, bb in zip(a, b)]
                ).type_as(a[0])
            else:
                layer_loss = torch.frobenius_norm(a - b, dim=-1)
        elif difference_function == "frobenius_mix":
            layer_loss_old = torch.frobenius_norm(a[0] - b[0], dim=-1)
            layer_loss_new = torch.frobenius_norm(a[1] - b[1], dim=-1)

            layer_loss = (
                mix_new_old * layer_loss_old + (1 - mix_new_old) * layer_loss_new
            )
        elif difference_function == "l1":
            if isinstance(a, list):
                layer_loss = torch.tensor(
                    [torch.norm(aa - bb, p=1, dim=-1) for aa, bb in zip(a, b)]
                ).type_as(a[0])
            else:
                layer_loss = torch.norm(a - b, p=1, dim=-1)
        elif difference_function == "kl":
            d1, d2, d3 = a.shape
            a = (a.view(d1 * d2, d3) + 1e-8).log()
            b = b.view(d1 * d2, d3) + 1e-8

            layer_loss = (
                F.kl_div(a, b, reduction="none").view(d1, d2, d3).sum(dim=(1, 2))
            )
        elif difference_function == "bce":
            d1, d2, d3 = a.shape
            layer_loss = (
                bce(a.view(d1 * d2, d3), b.view(d1 * d2, d3))
                .view(d1, d2, d3)
                .mean(dim=(1, 2))
            )
        else:
            raise NotImplementedError(
                f"Unknown difference_function={difference_function}"
            )

        assert torch.isfinite(layer_loss).all(), layer_loss
        assert (layer_loss >= 0.0).all(), layer_loss

        layer_loss = torch.mean(adaptative_pod_factor * layer_loss)
        if pod_factor <= 0.0:
            continue

        layer_loss = pod_factor * layer_loss
        if use_pod_schedule:
            layer_loss = layer_loss * math.sqrt(nb_current_classes / nb_new_classes)
        loss += layer_loss

    return loss / len(list_attentions_a)


def entropy(probabilities):
    """Computes the entropy per pixel.

    # References:
        * ESL: Entropy-guided Self-supervised Learning for Domain Adaptation in Semantic Segmentation
          Saporta et al.
          CVPR Workshop 2020

    :param probabilities: Tensor of shape (b, c, w, h).
    :return: One entropy per pixel, shape (b, w, h)
    """
    factor = 1 / math.log(probabilities.shape[1] + 1e-8)
    return -factor * torch.mean(probabilities * torch.log(probabilities + 1e-8), dim=1)


def soft_crossentropy(
    logits,
    labels,
    logits_old,
    mask_valid_pseudo,
    mask_background,
    pseudo_soft,
    pseudo_soft_factor=1.0,
):
    if pseudo_soft not in ("soft_certain", "soft_uncertain"):
        raise ValueError(f"Invalid pseudo_soft={pseudo_soft}")
    nb_old_classes = logits_old.shape[1]
    bs, nb_new_classes, w, h = logits.shape

    loss_certain = F.cross_entropy(logits, labels, reduction="none", ignore_index=255)
    loss_uncertain = (
        torch.log_softmax(logits_old, dim=1)
        * torch.softmax(logits[:, :nb_old_classes], dim=1)
    ).sum(dim=1)

    if pseudo_soft == "soft_certain":
        mask_certain = ~mask_background
        # mask_uncertain = mask_valid_pseudo & mask_background
    elif pseudo_soft == "soft_uncertain":
        mask_certain = (mask_valid_pseudo & mask_background) | (~mask_background)
        # mask_uncertain = ~mask_valid_pseudo & mask_background

    loss_certain = mask_certain.float() * loss_certain
    loss_uncertain = (~mask_certain).float() * loss_uncertain

    return loss_certain + pseudo_soft_factor * loss_uncertain


class UnbiasedKnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction="mean", alpha=1.0):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, mask=None):

        new_cl = inputs.shape[1] - targets.shape[1]

        targets = targets * self.alpha

        new_bkg_idx = (
            torch.tensor([0] + [x for x in range(targets.shape[1], inputs.shape[1])])
            .type_as(inputs)
            .long()
        )

        den = torch.logsumexp(inputs, dim=1)  # B, H, W
        outputs_no_bgk = inputs[:, 1:-new_cl] - den.unsqueeze(dim=1)  # B, OLD_CL, H, W
        outputs_bkg = (
            torch.logsumexp(torch.index_select(inputs, index=new_bkg_idx, dim=1), dim=1)
            - den
        )  # B, H, W

        labels = torch.softmax(targets, dim=1)  # B, BKG + OLD_CL, H, W

        # make the average on the classes 1/n_cl \sum{c=1..n_cl} L_c
        loss = (
            labels[:, 0] * outputs_bkg + (labels[:, 1:] * outputs_no_bgk).sum(dim=1)
        ) / targets.shape[1]

        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == "mean":
            outputs = -torch.mean(loss)
        elif self.reduction == "sum":
            outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs


class UnbiasedCrossEntropy(nn.Module):
    def __init__(self, old_cl=None, reduction="mean", ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.old_cl = old_cl

    def forward(self, inputs, targets):

        old_cl = self.old_cl
        outputs = torch.zeros_like(inputs)  # B, C (1+V+N), H, W
        den = torch.logsumexp(inputs, dim=1)  # B, H, W       den of softmax
        outputs[:, 0] = (
            torch.logsumexp(inputs[:, 0:old_cl], dim=1) - den
        )  # B, H, W       p(O)
        outputs[:, old_cl:] = inputs[:, old_cl:] - den.unsqueeze(
            dim=1
        )  # B, N, H, W    p(N_i)

        labels = targets.clone()  # B, H, W
        labels[
            targets < old_cl
        ] = 0  # just to be sure that all labels old belongs to zero

        loss = F.nll_loss(
            outputs, labels, ignore_index=self.ignore_index, reduction=self.reduction
        )

        return loss


class WeightedCrossEntropy(nn.Module):
    def __init__(self, gamma=2, old_cl=None, threshold=0.5, ignore_index=255, ukd=True):
        super().__init__()
        self.ignore_index = ignore_index
        self.old_cl = old_cl
        self.eps = 1e-4
        self.gamma = gamma
        self.base_loss = None
        self.threshold=threshold
        self.ukd = ukd

    def flatten_prob(self, prob, N, C):
        if prob.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            prob = prob.view(N, C, -1)
            prob = prob.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            prob = prob.view(-1, prob.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]
        return prob

    def _custom_unbiased(
        self, inputs, targets, seen_not_seen_probs, task_num
    ):
        max_seen_not_seen = seen_not_seen_probs.max(1)[0].detach()
        weights = torch.zeros_like(inputs).type_as(max_seen_not_seen)
        max_seen_not_seen[max_seen_not_seen > self.threshold] = 1.0
        weights[:, 0] = max_seen_not_seen
        N, C = inputs.shape[:2]
        weights = self.flatten_prob(weights, N, C)
        masked_targets = targets * (targets != self.ignore_index)
        masked_targets = masked_targets.view(-1, 1)
        weights = weights.gather(1, masked_targets).view(-1)
        # increasing the gamma would increase the focus of our model on true background
        focal_modulation = (1.0 - weights) ** self.gamma
        old_cl = self.old_cl
        outputs = torch.zeros_like(inputs)
        den = torch.logsumexp(
            inputs, dim=1
        )
        outputs[:, 0] = inputs[:, 0] - den
        outputs[:, 1] = torch.logsumexp(inputs[:, 1:], dim=1) - den
        labels_bg_fg = targets.clone()
        labels_bg_fg[(targets != 0) & (targets != self.ignore_index)] = 1
        loss_bg_fg = focal_modulation * F.nll_loss(
            outputs, labels_bg_fg, ignore_index=self.ignore_index, reduction="none"
        ).view(-1)
        labels_new_vs_rest = targets.clone()
        labels_new_vs_rest[(targets < old_cl)] = 0
        outputs = torch.zeros_like(inputs)
        if self.ukd:
            outputs[:, 0] = (
                torch.logsumexp(inputs[:, 0:old_cl], dim=1) - den
            )  # B, H, W       p(O)
        outputs[:, old_cl:] = inputs[:, old_cl:] - den.unsqueeze(
            dim=1
        )  # B, N, H, W    p(N_i)
        loss_new_vs_rest = F.nll_loss(
            outputs,
            labels_new_vs_rest,
            ignore_index=self.ignore_index,
            reduction="none",
        ).view(-1)
        loss = (loss_bg_fg + loss_new_vs_rest).mean() 
        return loss

    def forward(self, inputs, targets, seen_not_seen_probs, task_num):
        return self._custom_unbiased(inputs, targets, seen_not_seen_probs, task_num)


class IcarlCriterion(nn.Module):
    def __init__(self, reduction="mean", ignore_index=255, bkg=False):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.bkg = bkg

    def forward(self, inputs, targets, output_old):
        # inputs of size B x C x H x W
        n_cl = torch.tensor(inputs.shape[1]).type_as(targets)
        labels_new = torch.where(targets != self.ignore_index, targets, n_cl)
        # replace ignore with numclasses + 1 (to enable one hot and then remove it)
        targets = F.one_hot(labels_new, inputs.shape[1] + 1).float().permute(0, 3, 1, 2)
        targets = targets[:, : inputs.shape[1], :, :]  # remove 255 from 1hot
        if self.bkg:
            targets[:, 1 : output_old.shape[1], :, :] = output_old[:, 1:, :, :]
        else:
            targets[:, : output_old.shape[1], :, :] = output_old

        # targets is B x C x H x W so shape[1] is C
        loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        # loss has shape B x C x H x W
        loss = loss.sum(dim=1)  # sum the contributions of the classes
        if self.reduction == "mean":
            # if targets have only zeros, we skip them
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

def normalize(x):
    x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
    x_normalized = x.div(x_norm + 0.00001)
    return x_normalized