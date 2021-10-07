import torch
import torch_xla
import torch_xla.core.xla_model as xm


## According to the article, this learner does not use query set
def reptile_learner(model, queue, optimizer, args):
    # Support_set: [shot_num*ways, C, H, W], [shot_num*ways]
    # Query_set:   [q_num*ways,    C, H, W], [q_num*ways]
    model.train()

    old_vars = []
    running_vars = []
    for param in model.parameters():
        old_vars.append(param.data.clone())

    n = len(queue)
    losses = 0

    for i in range(n):
        for _ in range(args.update_step):
            optimizer.zero_grad()
            # logits, _ = model.forward(support_images)
            output = model.forward(queue[i]["task"], queue[i]["batch"][0])
            loss = output[0].mean()
            # loss_cls = criterion(logits, support_labels)
            # loss_cls = loss_cls.mean()
            # loss = loss_cls
            loss.backward()
            losses += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            if args.tpu:
                # Optimizer for TPU
                xm.optimizer_step(optimizer, barrier=True)
            else:
                # Optimizer for GPU
                optimizer.step()



        if running_vars == []:
            for _, param in enumerate(model.parameters()):
                running_vars.append(param.data.clone())
        else:
            for idx, param in enumerate(model.parameters()):
                running_vars[idx].data += param.data.clone()

        for idx, param in enumerate(model.parameters()):
            param.data = old_vars[idx].data.clone()

    for param in running_vars:
        param /= n

    for idx, param in enumerate(model.parameters()):
        param.data = old_vars[idx].data + args.beta * (
            running_vars[idx].data - old_vars[idx].data
        )

    return losses / (n * args.update_step)


def reptile_evaluate(model, dataloader, criterion, device):
    with torch.no_grad():
        total_loss = 0.0
        model.eval()
        for i, data in enumerate(dataloader):

            sample, labels = data
            sample, labels = sample.to(device), labels.to(device)

            logits, _ = model.forward(sample)

            loss = criterion(logits, labels)
            loss = loss.mean()
            total_loss += loss.item()

        total_loss /= len(dataloader)
        return total_loss
