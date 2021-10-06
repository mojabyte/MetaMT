import torch
import torch.nn.functional as F
import numpy as np
import time


def compute_prototypes(
    support_features: torch.Tensor, support_labels: torch.Tensor
) -> torch.Tensor:
    """
    Compute class prototypes from support features and labels
    Args:
        support_features: for each instance in the support set, its feature vector
        support_labels: for each instance in the support set, its label
    Returns:
        for each label of the support set, the average feature vector of instances with this label
    """
    seen_labels = torch.unique(support_labels)

    # Prototype i is the mean of all instances of features corresponding to labels == i
    return torch.cat(
        [
          support_features[(support_labels == l).nonzero(as_tuple=True)[0]].mean(0).reshape(1, -1)
          for l in seen_labels
        ]
    )


def pt_learner(model,
                support_images,
                support_labels,
                query_images,
                query_labels,
                criterion,
                optimizer,
                args):
  model.train()  
  optimizer.zero_grad()

  # with torch.no_grad():
  _, support_features = model.forward(support_images)
  
  prototypes = compute_prototypes(support_features, support_labels)
  # prototypes = prototypes.detach()
  outputs, query_features = model.forward(query_images)

  #   dists = torch.cdist(z_query, prototypes)
  #   classification_scores = -dists
  #   loss = criterion(classification_scores, query_labels)
  # input = torch.cat((support_features, query_features))
  # target = torch.cat((support_labels, query_labels))

  # loss, prototypes = prototypical_loss(input, target, args.ways)
  loss = criterion(query_features, outputs, query_labels, prototypes)

  loss.backward()

  torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
  optimizer.step()

  return loss, prototypes


## For Pt.
def pt_evaluate(model, dataloader, prototypes, criterion, device):
  
  ce = torch.nn.CrossEntropyLoss()

  with torch.no_grad():
    total_loss = 0.0
    model.eval()
    for i, batch in enumerate(dataloader):

      sample, labels = batch
      sample, labels = sample.to(device), labels.to(device)
      
      logits, features = model.forward(sample)
      # loss = criterion(features, logits, labels, prototypes)
      loss = ce(logits, labels)
      # loss, acc = criterion(features, target=labels)
      loss = loss.mean()
      total_loss += loss.item()

    total_loss /= len(dataloader)
    return total_loss