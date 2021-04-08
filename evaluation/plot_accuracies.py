import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DEFAULT_DIR = os.path.expanduser("~/lis-cluster/multimodal_seq2seq_gSCAN/out_learner_t_l_reset_optimizer")

BATCH_SIZE = 200
START_ITERATION = 192053


def plot(args):
    accs = []
    baseline = None
    pathlist = Path(args.data_dir).rglob('*.json')
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
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DIR, help="Dir with accuracy JSON files")
    args = parser.parse_args()

    plot(args)
