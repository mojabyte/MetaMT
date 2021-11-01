import torch
import time


def reptile_learner(model, queue, optimizer, iteration, args):
    model.train()

    old_vars = [param.data.clone() for param in model.parameters()]

    queue_length = len(queue)
    losses = 0

    for k in range(args.update_step):
        for i in range(queue_length):
            main_time = time.time()
            optimizer.zero_grad()

            data = queue[i]["batch"][k]
            task = queue[i]["task"]

            output = model.forward(task, data)

            print("forward:", time.time() - main_time)
            main_time = time.time()

            loss = output[0].mean()
            print("mean loss:", time.time() - main_time)
            main_time = time.time()

            loss.backward()
            print("backward:", time.time() - main_time)
            main_time = time.time()

            losses += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            print("finalize:", time.time() - main_time)

    beta = args.beta * (1 - iteration / args.meta_iteration)
    for idx, param in enumerate(model.parameters()):
        param.data = (1 - beta) * old_vars[idx].data + beta * param.data

    return losses / (queue_length * args.update_step)


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
