import argparse, gc, time, torch, os, logging, warnings, sys
import random

import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from data import CorpusQA, CorpusSC
from model import BertMetaLearning
from datapath import loc, get_loc

# import torch_xla
# import torch_xla.core.xla_model as xm

from sampler import TaskSampler
from learners.reptile_learner import reptile_learner
from utils.logger import Logger

from transformers import AdamW

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument("--meta_lr", type=float, default=2e-5, help="meta learning rate")
parser.add_argument("--dropout", type=float, default=0.1, help="")
parser.add_argument("--hidden_dims", type=int, default=768, help="")  # 768

# bert-base-multilingual-cased
# xlm-roberta-base
parser.add_argument(
    "--model_name",
    type=str,
    default="xlm-roberta-base",
    help="name of the pretrained model",
)
parser.add_argument(
    "--local_model", action="store_true", help="use local pretrained model"
)

parser.add_argument("--sc_labels", type=int, default=3, help="")
parser.add_argument("--qa_labels", type=int, default=2, help="")

parser.add_argument("--qa_batch_size", type=int, default=8, help="batch size")
parser.add_argument("--sc_batch_size", type=int, default=32, help="batch size")

parser.add_argument("--task_per_queue", type=int, default=8, help="")
parser.add_argument(
    "--update_step", type=int, default=3, help="number of Reptile update steps"
)
parser.add_argument("--temp", type=float, default=1.0)
parser.add_argument("--beta", type=float, default=1.0, help="")

# ---------------
parser.add_argument("--epochs", type=int, default=5, help="iterations")  # 5
parser.add_argument(
    "--start_epoch", type=int, default=0, help="start iterations from"
)  # 0
parser.add_argument("--ways", type=int, default=2, help="number of ways")  # 2
parser.add_argument(
    "--query_ways", type=int, default=2, help="number of ways for query"
)
parser.add_argument("--shot", type=int, default=4, help="number of shots")  # 4
parser.add_argument("--query_num", type=int, default=0, help="number of queries")  # 0
parser.add_argument("--meta_iteration", type=int, default=3000, help="")
# ---------------

parser.add_argument("--seed", type=int, default=42, help="seed for numpy and pytorch")
parser.add_argument(
    "--log_interval",
    type=int,
    default=200,
    help="Print after every log_interval batches",
)
parser.add_argument("--data_dir", type=str, default="data/", help="directory of data")
parser.add_argument("--cuda", action="store_true", help="use CUDA")
parser.add_argument("--tpu", action="store_true", help="use TPU")
parser.add_argument("--save", type=str, default="saved/", help="")
parser.add_argument("--load", type=str, default="", help="")
parser.add_argument("--log_file", type=str, default="main_output.txt", help="")
parser.add_argument("--grad_clip", type=float, default=5.0)
parser.add_argument("--meta_tasks", type=str, default="sc")
parser.add_argument("--queue_length", default=8, type=int)

parser.add_argument("--num_workers", type=int, default=0, help="")
parser.add_argument("--pin_memory", action="store_true", help="")
parser.add_argument(
    "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
)
parser.add_argument("--scheduler", action="store_true", help="use scheduler")
parser.add_argument("--step_size", default=3000, type=int)
parser.add_argument("--last_step", default=0, type=int)
parser.add_argument("--gamma", default=0.1, type=float)
parser.add_argument("--warmup", default=0, type=int)
parser.add_argument(
    "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
)

args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

sys.stdout = Logger(os.path.join(args.save, args.log_file))
print(args)

task_types = args.meta_tasks.split(",")
list_of_tasks = []

for tt in loc["train"].keys():
    if tt[:2] in task_types:
        list_of_tasks.append(tt)

for tt in task_types:
    if "_" in tt:
        list_of_tasks.append(tt)

list_of_tasks = list(set(list_of_tasks))
print(list_of_tasks)


def evaluate(model, task, data):
    with torch.no_grad():
        total_loss = 0.0
        for batch in data:
            output = model.forward(task, batch)
            loss = output[0].mean()
            total_loss += loss.item()
        total_loss /= len(data)
        return total_loss


def evaluateMeta(model, dev_loaders):
    loss_dict = {}
    total_loss = 0
    model.eval()
    for task in list_of_tasks:
        loss = evaluate(model, task, dev_loaders[task])
        loss_dict[task] = loss
        total_loss += loss
    return loss_dict, total_loss


def get_batch(dataloader_iter, dataloader):
    try:
        batch = next(dataloader_iter)
    except StopIteration:
        dataloader_iter = iter(dataloader)
        batch = next(dataloader_iter)
    return batch


class Sampler:
    def __init__(self, p, dataloaders):
        # Sampling Weights
        self.init_p = p

        self.task_per_queue = args.queue_length
        self.list_of_tasks = list_of_tasks
        self.dataloaders = dataloaders
        self.list_of_iters = {k: iter(dataloaders[k]) for k in self.list_of_tasks}

    def __iter__(self):
        return self

    def __next__(self):
        curr_p = self.init_p

        tasks = np.random.choice(self.list_of_tasks, self.task_per_queue, p=curr_p)
        queue = [
            {
                "task": tasks[i],
                "batch": get_batch(
                    self.list_of_iters[tasks[i]], self.dataloaders[tasks[i]]
                ),
            }
            for i in range(self.task_per_queue)
        ]
        return queue


def UniformBatchSampler(dataloaders, corpus_len):
    p = np.array(
        [
            corpus_len[y] * 1.0 / sum([corpus_len[x] for x in list_of_tasks])
            for y in list_of_tasks
        ]
    )
    p_temp = np.power(p, 1.0 / args.temp)
    p_temp = p_temp / np.sum(p_temp)
    print(p_temp)
    sampler = iter(Sampler(p_temp, dataloaders))
    return sampler


def main():
    if torch.cuda.is_available():
        if not args.cuda:
            args.cuda = True

        torch.cuda.manual_seed_all(args.seed)

    # DEVICE = (
    #     xm.xla_device() if args.tpu else torch.device("cuda" if args.cuda else "cpu")
    # )

    DEVICE = torch.device("cuda" if args.cuda else "cpu")

    # loader
    train_loaders = {}
    dev_loaders = {}
    corpus_len = {}

    for k in list_of_tasks:
        train_corpus = None
        dev_corpus = None
        batch_size = 32

        if "qa" in k:
            train_corpus = CorpusQA(
                *get_loc("train", k, args.data_dir),
                model_name=args.model_name,
                local_files_only=args.local_model,
            )
            dev_corpus = CorpusQA(
                *get_loc("dev", k, args.data_dir),
                model_name=args.model_name,
                local_files_only=args.local_model,
            )
            batch_size = args.qa_batch_size
        elif "sc" in k:
            train_corpus = CorpusSC(
                *get_loc("train", k, args.data_dir),
                model_name=args.model_name,
                local_files_only=args.local_model,
            )
            dev_corpus = CorpusSC(
                *get_loc("dev", k, args.data_dir),
                model_name=args.model_name,
                local_files_only=args.local_model,
            )
            batch_size = args.sc_batch_size
        else:
            continue

        train_sampler = TaskSampler(
            train_corpus,
            n_way=args.ways,
            # n_query_way=args.query_ways,
            n_shot=args.shot,
            n_query=args.query_num,
            n_tasks=args.meta_iteration,
            reptile_step=args.update_step,
        )
        train_loader = DataLoader(
            train_corpus,
            batch_sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            collate_fn=train_sampler.episodic_collate_fn,
        )
        train_loaders[k] = train_loader
        corpus_len[k] = len(train_corpus)

        dev_loader = DataLoader(
            dev_corpus, batch_size=batch_size, pin_memory=args.pin_memory
        )
        dev_loaders[k] = dev_loader

        gc.collect()

    if args.load != "":
        print(f"loading model {args.load}...")
        model = torch.load(args.load)
    else:
        model = BertMetaLearning(args).to(DEVICE)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
            "lr": args.meta_lr,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
            "lr": args.meta_lr,
        },
    ]

    optim = AdamW(optimizer_grouped_parameters, lr=args.meta_lr, eps=args.adam_epsilon)

    scheduler = StepLR(
        optim,
        step_size=args.step_size,
        gamma=args.gamma,
        last_epoch=args.last_step - 1,
    )

    ## == 2) Learn model
    global_time = time.time()

    min_task_losses = {
        "qa": float("inf"),
        "sc": float("inf"),
    }

    sampler = UniformBatchSampler(train_loaders, corpus_len)

    try:
        for epoch_item in range(args.start_epoch, args.epochs):
            print(f"======================= Epoch {epoch_item} =======================")
            train_loss = 0.0

            # train_loader_iterations = [
            #     iter(train_loader) for train_loader in train_loaders
            # ]

            for miteration_item, metabatch in enumerate(sampler):
                # == Data preparation ===========
                # queue = [
                #     {"batch": next(train_loader_iterations[i]), "task": task}
                #     for i, task in enumerate(list_of_tasks)
                # ]

                # if args.queue_length < len(train_loader_iterations):
                #     queue = random.sample(queue, args.queue_length)

                ## == train ===================
                loss = reptile_learner(model, metabatch, optim, miteration_item, args)
                train_loss += loss

                ## == validation ==============
                if (miteration_item + 1) % args.log_interval == 0:

                    total_loss = train_loss / args.log_interval
                    train_loss = 0.0

                    # evalute on val_dataset
                    val_loss_dict, val_loss_total = evaluateMeta(model, dev_loaders)

                    loss_per_task = {}
                    for task in val_loss_dict.keys():
                        if task[:2] in loss_per_task.keys():
                            loss_per_task[task[:2]] = (
                                loss_per_task[task[:2]] + val_loss_dict[task]
                            )
                        else:
                            loss_per_task[task[:2]] = val_loss_dict[task]

                    for task in loss_per_task.keys():
                        if loss_per_task[task] < min_task_losses[task]:
                            print("Saving " + task + "  Model")
                            torch.save(
                                model, os.path.join(args.save, "model_" + task + ".pt"),
                            )
                            min_task_losses[task] = loss_per_task[task]

                    print(
                        "Time: %f, Step: %d, Train Loss: %f, Val Loss: %f"
                        % (
                            time.time() - global_time,
                            miteration_item + 1,
                            total_loss,
                            val_loss_total,
                        )
                    )
                    global_time = time.time()

                    total_loss = 0

                if args.scheduler:
                    scheduler.step()

    except KeyboardInterrupt:
        print("skipping training")

    print("Saving new last model...")
    torch.save(model, os.path.join(args.save, "model_last.pt"))


if __name__ == "__main__":
    main()

