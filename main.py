import argparse, time, torch, os, logging, warnings, sys

import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from data import CorpusQA, CorpusSC, CorpusTC, CorpusPO, CorpusPA
from model import BertMetaLearning
from datapath import loc, get_loc

# import torch_xla
# import torch_xla.core.xla_model as xm

from sampler import TaskSampler
from learners.reptile_learner import reptile_learner
from utils.logger import Logger

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)

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
parser.add_argument("--tc_labels", type=int, default=10, help="")
parser.add_argument("--po_labels", type=int, default=18, help="")
parser.add_argument("--pa_labels", type=int, default=2, help="")

parser.add_argument("--qa_batch_size", type=int, default=8, help="batch size")
parser.add_argument("--sc_batch_size", type=int, default=32, help="batch size")  # 32
parser.add_argument("--tc_batch_size", type=int, default=32, help="batch size")
parser.add_argument("--po_batch_size", type=int, default=32, help="batch_size")
parser.add_argument("--pa_batch_size", type=int, default=8, help="batch size")

parser.add_argument("--task_per_queue", type=int, default=8, help="")
parser.add_argument(
    "--update_step", type=int, default=3, help="number of REPTILE update steps"
)
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
parser.add_argument("--meta_tasks", type=str, default="sc,pa,qa,tc,po")

parser.add_argument(
    "--sampler", type=str, default="uniform_batch", choices=["uniform_batch"]
)
parser.add_argument("--temp", type=float, default=1.0)

parser.add_argument("--num_workers", type=int, default=0, help="")
parser.add_argument("--n_best_size", default=20, type=int)  # 20
parser.add_argument("--max_answer_length", default=30, type=int)  # 30
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
    for i, task in enumerate(list_of_tasks):
        loss = evaluate(model, task, dev_loaders[i])
        loss_dict[task] = loss
        total_loss += loss
    return loss_dict, total_loss


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
    train_loaders = []
    dev_loaders = []

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
        elif "tc" in k:
            train_corpus = CorpusTC(
                get_loc("train", k, args.data_dir)[0],
                model_name=args.model_name,
                local_files_only=args.local_model,
            )
            dev_corpus = CorpusTC(
                get_loc("dev", k, args.data_dir)[0],
                model_name=args.model_name,
                local_files_only=args.local_model,
            )
            batch_size = args.tc_batch_size
        elif "po" in k:
            train_corpus = CorpusPO(
                get_loc("train", k, args.data_dir)[0],
                model_name=args.model_name,
                local_files_only=args.local_model,
            )
            dev_corpus = CorpusPO(
                get_loc("dev", k, args.data_dir)[0],
                model_name=args.model_name,
                local_files_only=args.local_model,
            )
            batch_size = args.po_batch_size
        elif "pa" in k:
            train_corpus = CorpusPA(
                get_loc("train", k, args.data_dir)[0],
                model_name=args.model_name,
                local_files_only=args.local_model,
            )
            dev_corpus = CorpusPA(
                get_loc("dev", k, args.data_dir)[0],
                model_name=args.model_name,
                local_files_only=args.local_model,
            )
            batch_size = args.pa_batch_size
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
            pin_memory=True,
            collate_fn=train_sampler.episodic_collate_fn,
        )
        train_loaders.append(train_loader)

        dev_loader = DataLoader(dev_corpus, batch_size=batch_size, pin_memory=True)
        dev_loaders.append(dev_loader)

    model = BertMetaLearning(args).to(DEVICE)

    if args.load != "":
        print(f"loading model {args.load}...")
        model = torch.load(args.load)

    # steps = args.epochs * args.meta_iteration

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
    # scheduler = get_linear_schedule_with_warmup(
    #     optim,
    #     num_warmup_steps=args.warmup,
    #     num_training_steps=steps,
    #     last_epoch=args.start_epoch - 1,
    # )
    scheduler = StepLR(
        optim,
        step_size=args.step_size,
        gamma=args.gamma,
        last_epoch=args.last_step - 1,
    )

    logger = {}
    logger["total_val_loss"] = []
    logger["val_loss"] = {k: [] for k in list_of_tasks}
    logger["train_loss"] = []
    logger["args"] = args

    ## = Model Update config.
    # criterion  = nn.CrossEntropyLoss()
    # criterion_mt = losses.NTXentLoss(temperature=0.07)
    #   criterion = CPELoss(args)
    # criterion = PrototypicalLoss(n_support=args.shot)
    # optim = SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    # optim = Adam(model.parameters(),
    #               lr=args.lr,
    #               weight_decay=args.wd)

    # scheduler = StepLR(optim, step_size=2, gamma=args.gamma)

    ## == 2) Learn model
    global_time = time.time()

    min_task_losses = {
        "qa": float("inf"),
        "sc": float("inf"),
        "po": float("inf"),
        "tc": float("inf"),
        "pa": float("inf"),
    }

    try:
        for epoch_item in range(args.start_epoch, args.epochs):
            print(
                "===================================== Epoch %d ====================================="
                % epoch_item
            )
            train_loss = 0.0

            train_loader_iterations = [
                iter(train_loader) for train_loader in train_loaders
            ]
            for miteration_item in range(args.meta_iteration):

                # == Data preparation ===========
                queue = [
                    {"batch": next(train_loader_iterations[i]), "task": task}
                    for i, task in enumerate(list_of_tasks)
                ]

                ## == train ===================
                loss = reptile_learner(model, queue, optim, miteration_item, args)
                # loss, prototypes = pt_learner(
                #     model,
                #     support_images,
                #     support_labels,
                #     query_images,
                #     query_labels,
                #     criterion,
                #     optim,
                #     args,
                # )
                train_loss += loss

                ## == validation ==============
                if (miteration_item + 1) % args.log_interval == 0:

                    total_loss = train_loss / args.log_interval
                    train_loss = 0.0

                    # evalute on val_dataset
                    val_loss_dict, val_loss_total = evaluateMeta(model, dev_loaders)
                    # val_loss_total = reptile_evaluate(model, dev_loader, criterion, DEVICE) # For Reptile
                    # val_loss_total = pt_evaluate(
                    #     model, val_dataloader, prototypes, criterion, device
                    # )  # For Pt.

                    loss_per_task = {}
                    for task in val_loss_dict.keys():
                        if task[:2] in loss_per_task.keys():
                            loss_per_task[task[:2]] = (
                                loss_per_task[task[:2]] + val_loss_dict[task]
                            )
                        else:
                            loss_per_task[task[:2]] = val_loss_dict[task]

                    print(
                        "Time: %f, Step: %d, Train Loss: %f, Val Loss: %f"
                        % (
                            time.time() - global_time,
                            miteration_item + 1,
                            total_loss,
                            val_loss_total,
                        )
                    )
                    print("===============================================")
                    global_time = time.time()

                    for task in loss_per_task.keys():
                        if loss_per_task[task] < min_task_losses[task]:
                            torch.save(
                                model, os.path.join(args.save, "model_" + task + ".pt"),
                            )
                            min_task_losses[task] = loss_per_task[task]
                            print("Saving " + task + "  Model")
                    total_loss = 0

                if args.scheduler:
                    scheduler.step()

    except KeyboardInterrupt:
        print("skipping training")

    # save last model
    torch.save(model, os.path.join(args.save, "model_last.pt"))
    print("Saving new last model")


if __name__ == "__main__":
    main()

