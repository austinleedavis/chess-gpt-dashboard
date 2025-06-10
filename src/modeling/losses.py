import torch
import torch.nn.functional as F


class OrdinalCrossEntropyLoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()

        self.reduction = reduction

        if reduction == "mean":
            self.reduction_fun = torch.mean
        elif reduction == "sum":
            self.reduction_fun = torch.sum
        elif reduction == "none":
            self.reduction_fun = lambda x: x
        else:
            raise ValueError(f'reduction must be one of {["mean","sum","none"]}')

    def forward(self, input, target):
        # input: (N, C), target: (N,)
        # target is a list of integers representing the bin index
        # input is the raw output of the model
        # R, C = input.shape
        # bins = torch.arange(C).to(input.device)
        # target

        # weights = (
        #     abs(bins.unsqueeze(0) - target.unsqueeze(1)).float() + 2
        # ).sqrt()  # viable, but not smooth

        # s = math.log(C)
        # import math

        # mu = 0
        # dnorm = lambda x, s, mu: (1 / s) * torch.exp(-((x - mu) ** 2) / (2 * s * s))
        # weights = 1 - dnorm(bins.unsqueeze(0), s, target.unsqueeze(1))
        # px.line(weights.T)
        # norm_input = input - input.min(dim=1).values.unsqueeze(1)

        # norm_input = input * weights

        # losses = (norm_input.softmax(dim=1))[torch.arange(R), target]

        # average over the batch
        # return self.reduction_fun(losses)
        return my_loss_fn(input, target)


def my_loss_fn(logits, targets, reduction="mean"):
    _, num_classes = logits.shape

    Px = logits

    targets_one_hot = F.one_hot(targets, num_classes).float()

    class_indices = torch.arange(num_classes, device=Px.device).unsqueeze(0)
    distances = torch.abs(class_indices - targets.unsqueeze(1))

    loss = (1 + distances) * (targets_one_hot - Px) ** 2

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:  # assume reduction == "none"
        return loss


# # if __name__ == "__main__":
# #     import plotly_express as px

# #     n_pos = 5
# #     n_bins = 200
# #     target = torch.linspace(0, n_bins - 1, n_pos).long()
# #     input = torch.zeros((n_pos, n_bins))
# #     input[torch.arange(n_pos), target] = 1

# #     # ordloss = OrdinalCrossEntropyLoss("sum")
# #     celoss = torch.nn.CrossEntropyLoss(reduction="none")
# #     px.bar(celoss(input, target))
# #     # ordloss(input,target)

# #     R, C = input.shape
# #     bins = torch.arange(C).to(input.device)
# #     target

# #     weights = (
# #         abs(bins.unsqueeze(0) - target.unsqueeze(1)).float() + 2
# #     ).sqrt()  # viable, but not smooth

# #     spread_affinity = 0
# #     height = 1
# #     penalty_shift = 1 / height  # 2
# #     penalty_scale = height  # 0.5

# #     std = torch.std(bins.float())
# #     s2 = 2 * std ** (spread_affinity)
# #     gaussian_penalty = lambda x, mu: torch.exp(
# #         penalty_scale * (penalty_shift - torch.exp(-((x - mu) ** 2) / s2))
# #     )
# #     weights = gaussian_penalty(bins.unsqueeze(0), target.unsqueeze(1))
# #     px.line(
# #         weights.T, title="Weight value by Bin index", range_y=(0, weights.max().item())
# #     )

# #     prob_x = weights * (input).softmax(0)
# #     celoss(prob_x, target)
# #     px.bar(the_loss_value)
# #     px.line(
# #         (input * weights).softmax(0).T,
# #         title="Weight value by Bin index",
# #         range_y=(0, 2),
# #     )

# #     ############7

# #     def get_weight(logits, height=1.0, spread_affinity=1.0):
# #         shift = 1 / height
# #         scale = height

# #         R, C = logits.shape
# #         bins = torch.arange(C).to(logits.device)
# #         std = torch.std(bins.float())

# #         s2 = 2 * std ** (spread_affinity)
# #         gaussian_penalty = lambda x, mu: scale * (
# #             torch.exp(-((x - mu) ** 2) / s2) + shift
# #         )
# #         weights = gaussian_penalty(bins.unsqueeze(0), logits.unsqueeze(1))
# #         return weights

# #     ########
# #     import torch.nn.functional as F

# #     alpha = 1

# #     # logits[torch.arange(n_pos),targets] = 1
# # import torch


# def my_loss_fn(logits, targets, reduction="mean"):
#     _, num_classes = logits.shape

#     Px = logits

#     targets_one_hot = F.one_hot(targets, num_classes).float()

#     class_indices = torch.arange(num_classes, device=Px.device).unsqueeze(0)
#     distances = torch.abs(class_indices - targets.unsqueeze(1))

#     loss = (1 + distances) * (targets_one_hot - Px) ** 2

#     if reduction == "mean":
#         return loss.mean()
#     elif reduction == "sum":
#         return loss.sum()
#     else:  # assume reduction == "none"
#         return loss


# losses_by_mass = []
# for mass in torch.linspace(0, 10, 20).tolist():
#     losses_by_index = []
#     for i in range(50):
#         losses = []
#         for run in range(30):
#             n_pos = 50
#             num_classes = 128
#             targets = torch.linspace(0, num_classes - 1, n_pos).long()
#             logits = torch.rand((n_pos, num_classes))
#             logits[torch.arange(n_pos), (targets + i) % num_classes] += mass

#             # logits[torch.arange(n_pos), (targets+10)%n_pos] += 10

#             loss = my_loss_fn(logits, targets, "mean")
#             losses.append(loss)
#         losses_by_index.append(losses)
#         # outputs: ~
#     losses_by_mass.append(losses_by_index)

# losses = torch.tensor(losses_by_mass)
# losses.shape
