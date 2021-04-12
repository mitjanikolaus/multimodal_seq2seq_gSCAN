from GroundedScan.vocabulary import Vocabulary
from seq2seq.predict import predict
from seq2seq.helpers import sequence_accuracy

import torch.nn as nn
from typing import Iterator
from typing import Tuple
import numpy as np


def evaluate(data_iterator: Iterator, model: nn.Module, lm_vocab:Vocabulary, max_decoding_steps: int, pad_idx: int, sos_idx: int,
             eos_idx: int, max_examples_to_evaluate=None, dataset=None, test_batch_size=1) -> Tuple[float, float, float, float]:

    accuracy, exact_match, auxiliary_accuracy_target, lm_perplexity = predict(
        data_iterator=data_iterator, model=model, lm_vocab=lm_vocab, max_decoding_steps=max_decoding_steps,
        pad_idx=pad_idx,
        sos_idx=sos_idx, eos_idx=eos_idx, max_examples_to_evaluate=max_examples_to_evaluate, dataset=dataset,
        test_batch_size=test_batch_size)
    #
    # for input_sequence, _, _, output_sequence, target_sequence, _, _, aux_acc_target, lm_perplexity in predict(
    #         data_iterator=data_iterator, model=model, lm_vocab=lm_vocab, max_decoding_steps=max_decoding_steps, pad_idx=pad_idx,
    #         sos_idx=sos_idx, eos_idx=eos_idx, max_examples_to_evaluate=max_examples_to_evaluate, dataset=dataset,
    #         test_batch_size=test_batch_size):
    #     accuracy = sequence_accuracy(output_sequence, target_sequence[0].tolist()[1:-1])
    #     if accuracy == 100:
    #         exact_match += 1
    #     accuracies.append(accuracy)
    #     target_accuracies.append(aux_acc_target)
    #     lm_perplexities.append(lm_perplexity.item())
    return accuracy, exact_match, auxiliary_accuracy_target, lm_perplexity
    # return (float(np.mean(np.array(accuracies))), (exact_match / len(accuracies)) * 100,
    #         float(np.mean(np.array(target_accuracies))), float(np.mean(lm_perplexities)))


