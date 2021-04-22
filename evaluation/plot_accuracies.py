import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

BATCH_SIZE = 200


def plot_train_losses(args):
    train_logs = json.load(open(os.path.join(args.data_dir, "train_logs.json")))
    eval_logs = json.load(open(os.path.join(args.data_dir, "eval_logs.json")))

    train_logs = pd.DataFrame(train_logs).T
    eval_logs = pd.DataFrame(eval_logs).T

    train_logs["samples"] = train_logs.index.map(lambda x: (int(x) * BATCH_SIZE) - (args.start_iteration * BATCH_SIZE))
    eval_logs["samples"] = eval_logs.index.map(lambda x: (int(x) * BATCH_SIZE) - (args.start_iteration * BATCH_SIZE))

    train_logs.set_index("samples", inplace=True)
    eval_logs.set_index("samples", inplace=True)

    eval_logs["exact_match"] = eval_logs.exact_match.map(lambda x: x * 100)

    sns.lineplot(data=eval_logs[["perplexity", "accuracy", "exact_match"]])
    if args.x_max:
        plt.xlim(0, args.x_max)
    plt.show()

    # train_logs["ppl"] = train_logs.lm_loss.map(lambda x: np.exp(x))
    sns.lineplot(data=train_logs[["loss", "actions_loss", "lm_loss", "ppl"]])

    if args.x_max:
        plt.xlim(0, args.x_max)
    plt.show()


def plot_eval_accuracies(args):
    accs = []
    baseline = None
    pathlist = Path(args.data_dir).rglob('*_accuracies.json')
    for path in pathlist:
        with open(path) as f:
            data = json.load(f)

            if path.stem.split("_")[0] == "pretrain":
                baseline = data
            else:
                # add x-axis (iteration) information
                iteration = int(path.stem.split("_")[2])
                data["iteration"] = iteration

                accs.append(data)

    accs = pd.DataFrame(accs)
    accs["samples"] = accs["iteration"].map(lambda x: (x * BATCH_SIZE) - (args.start_iteration * BATCH_SIZE))

    del accs["iteration"]
    accs.set_index("samples", inplace=True)
    sns.lineplot(data=accs, dashes=False)

    if baseline:
        for i, (key, value) in enumerate(baseline.items()):
            plt.axhline(value, color=sns.color_palette()[i], linestyle="--")

    plt.ylim(0, 1)
    if args.x_max:
        plt.xlim(0, args.x_max)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Dir with accuracy JSON files")
    parser.add_argument("--start-iteration", type=int, default=0)
    parser.add_argument("--x-max", type=int, default=None)

    args = parser.parse_args()

    plot_train_losses(args)
    plot_eval_accuracies(args)
