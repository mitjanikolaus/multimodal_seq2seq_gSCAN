import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BATCH_SIZE = 200
START_ITERATION = 192053


def plot_train_losses(args):
    train_logs = json.load(open(os.path.join(args.data_dir, "train_logs.json")))
    eval_logs = json.load(open(os.path.join(args.data_dir, "eval_logs.json")))

    train_logs = pd.DataFrame(train_logs).T
    eval_logs = pd.DataFrame(eval_logs).T

    train_logs["samples"] = train_logs.index.map(lambda x: (int(x) * BATCH_SIZE) - (START_ITERATION * BATCH_SIZE))
    eval_logs["samples"] = eval_logs.index.map(lambda x: (int(x) * BATCH_SIZE) - (START_ITERATION * BATCH_SIZE))

    train_logs.set_index("samples", inplace=True)
    eval_logs.set_index("samples", inplace=True)

    # sns.lineplot(data=eval_logs[["exact_match"]])
    # plt.show()

    sns.lineplot(data=train_logs[["loss", "actions_loss", "lm_loss"]])
    plt.show()


def plot_eval_accuracies(args):
    accs = []
    baseline = None
    pathlist = Path(args.data_dir).rglob('*_accuracies.json')
    for path in pathlist:
        path_in_str = str(path)
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
    accs["samples"] = accs["iteration"].map(lambda x: (x * BATCH_SIZE) - (START_ITERATION * BATCH_SIZE))

    del accs["iteration"]
    accs.set_index("samples", inplace=True)
    sns.lineplot(data=accs, dashes=False)

    if baseline:
        for i, (key, value) in enumerate(baseline.items()):
            plt.axhline(value, color=sns.color_palette()[i], linestyle="--")

    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Dir with accuracy JSON files")
    args = parser.parse_args()

    plot_train_losses(args)
    plot_eval_accuracies(args)
