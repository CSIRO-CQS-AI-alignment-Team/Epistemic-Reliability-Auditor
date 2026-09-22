"""Print the Stage 10 no-transcript checkpoint summary.

The script reads the released ``base.json``, ``honest.json``, and ``adv.json`` files
for QuALITY-H and/or GPQA and reports Q_Y/Q_H accuracy plus mean H_true/H_false
probabilities from their aggregate metadata. It does not load a model or modify results.
"""

import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="all", help='Run evaluation on the specified dataset')
args = parser.parse_args()

datasets=[]
if args.dataset == "all":
    datasets = ["QuALITY-H","GPQA"]
else:
    datasets = [dataset]

for dataset in datasets:
    base_json = json.load(open(f"runs/posterior_shift{version}/{dataset}/base.json", "r"))
    honest_json = json.load(open(f"runs/posterior_shift{version}/{dataset}/honest.json", "r"))
    adversarial_json = json.load(open(f"runs/posterior_shift{version}/{dataset}/adv.json", "r"))

    Q_Y_base_acc = base_json["metadata"]["aggregate_metrics"]["Q_Y_accuracy"]
    Q_H_base_acc = base_json["metadata"]["aggregate_metrics"]["Q_H_accuracy"]
    Q_H_base_p_h_true_mean = base_json["metadata"]["aggregate_metrics"]["Q_H_mean_p_h_true"]
    Q_H_base_p_h_false_mean = base_json["metadata"]["aggregate_metrics"]["Q_H_mean_p_h_false"]

    Q_Y_honest_acc = honest_json["metadata"]["aggregate_metrics"]["Q_Y_accuracy"]
    Q_H_honest_acc = honest_json["metadata"]["aggregate_metrics"]["Q_H_accuracy"]
    Q_H_honest_p_h_true_mean = honest_json["metadata"]["aggregate_metrics"]["Q_H_mean_p_h_true"]
    Q_H_honest_p_h_false_mean = honest_json["metadata"]["aggregate_metrics"]["Q_H_mean_p_h_false"]

    Q_Y_adversarial_acc = adversarial_json["metadata"]["aggregate_metrics"]["Q_Y_accuracy"]
    Q_H_adversarial_acc = adversarial_json["metadata"]["aggregate_metrics"]["Q_H_accuracy"]
    Q_H_adversarial_p_h_true_mean = adversarial_json["metadata"]["aggregate_metrics"]["Q_H_mean_p_h_true"]
    Q_H_adversarial_p_h_false_mean = adversarial_json["metadata"]["aggregate_metrics"]["Q_H_mean_p_h_false"]


    print(f"Dataset: {dataset}")
    print("Accuracy")
    print("\tBase\tHonest\tAdversarial")
    print("Q_Y\t{:.4f}\t{:.4f}\t{:.4f}".format(Q_Y_base_acc, Q_Y_honest_acc, Q_Y_adversarial_acc))
    print("Q_H\t{:.4f}\t{:.4f}\t{:.4f}".format(Q_H_base_acc, Q_H_honest_acc, Q_H_adversarial_acc))

    print("Probability")
    print("\tBase\tHonest\tAdversarial")
    print("v(h_true)\t{:.4f}\t{:.4f}\t{:.4f}".format(Q_H_base_p_h_true_mean, Q_H_honest_p_h_true_mean, Q_H_adversarial_p_h_true_mean))
    print("v(h_false)\t{:.4f}\t{:.4f}\t{:.4f}".format(Q_H_base_p_h_false_mean, Q_H_honest_p_h_false_mean, Q_H_adversarial_p_h_false_mean))
    print("\n")