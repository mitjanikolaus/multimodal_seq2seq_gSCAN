import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Iterator
import time
import json
import matplotlib.pyplot as plt



import numpy as np

from seq2seq.helpers import sequence_accuracy
from seq2seq.gSCAN_dataset import GroundedScanDataset, Vocabulary
from GroundedScan.world import Situation

import pdb
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


def predict_and_save(dataset: GroundedScanDataset, model: nn.Module, output_file_path: str, max_decoding_steps: int,
                     lm_vocab:Vocabulary, max_testing_examples=None, save_predictions=True, **kwargs):
    """
    Predict all data in dataset with a model and write the predictions to output_file_path.
    :param dataset: a dataset with test examples
    :param model: a trained model from model.py
    :param output_file_path: a path where a .json file with predictions will be saved.
    :param max_decoding_steps: after how many steps to force quit decoding
    :param max_testing_examples: after how many examples to stop predicting, if None all examples will be evaluated
    """
    if save_predictions:
        raise NotImplementedError("Saving predictions currently not implemented.")

    cfg = locals().copy()

    output = []
    accuracies = []
    with torch.no_grad():
        i = 0

        for (input_batch, input_lengths, _, situation_batch, _, target_batch,
             target_lengths, agent_positions, target_positions, _) in dataset.get_data_iterator(
                batch_size=kwargs["test_batch_size"]
        ):
            i += 1
            if max_testing_examples:
                if i * kwargs["test_batch_size"] >= max_testing_examples:
                    break

            scores, predicted_action_sequences, action_sequence_lengths, _ = model.predict_actions_batch(
                input_batch,
                input_lengths,
                situation_batch,
                dataset.target_vocabulary.sos_idx,
                dataset.target_vocabulary.eos_idx,
                max_decoding_steps)

            for j in range(kwargs["test_batch_size"]):
                accuracy = sequence_accuracy(predicted_action_sequences[j][:int(target_lengths[j])][1:-1].tolist(),
                              target_batch[j][:int(target_lengths[j])][1:-1].tolist())
                accuracies.append(accuracy)
            # accuracy = sequence_accuracy(output_sequence, target_sequence[0].tolist()[1:-1])
            # input_str_sequence = dataset.array_to_sentence(input_sequence[0].tolist(), vocabulary="input")
            # input_str_sequence = input_str_sequence[1:-1]  # Get rid of <SOS> and <EOS>
            # target_str_sequence = dataset.array_to_sentence(target_sequence[0].tolist(), vocabulary="target")
            # target_str_sequence = target_str_sequence[1:-1]  # Get rid of <SOS> and <EOS>
            # output_str_sequence = dataset.array_to_sentence(output_sequence, vocabulary="target")
            # output.append({"input": input_str_sequence, "prediction": output_str_sequence,
            #                "derivation": derivation_spec,
            #                "target": target_str_sequence, "situation": situation_spec,
            #                "attention_weights_input": attention_weights_commands,
            #                "attention_weights_situation": attention_weights_situations,
            #                "accuracy": accuracy,
            #                "exact_match": True if accuracy == 100 else False,
            #                "position_accuracy": position_accuracy})
    logger.info("Made predictions for {} examples.".format(i*kwargs["test_batch_size"]))

    logger.info(f"Accuracy: {np.mean(accuracies):.3f}")
    exact_match_acc = np.mean([acc == 100 for acc in accuracies])
    logger.info(f"Exact Match Accuracy: {exact_match_acc:.3f}")

    # if save_predictions:
        # with open(output_file_path, mode='w') as outfile:
        #     json.dump(output, outfile, indent=4)

    return output_file_path, exact_match_acc


def predict(data_iterator: Iterator, model: nn.Module, lm_vocab: Vocabulary, max_decoding_steps: int, pad_idx: int, sos_idx: int,
            eos_idx: int, max_examples_to_evaluate=None, test_batch_size=1, dataset=None) -> (float, float, float):
    """
    Loop over all data in data_iterator and predict until <EOS> token is reached.
    :param data_iterator: iterator containing the data to predict
    :param model: a trained model from model.py
    :param lm_vocab: lm vocabulary
    :param max_decoding_steps: after how many steps to abort decoding
    :param pad_idx: the padding idx of the target vocabulary
    :param sos_idx: the start-of-sequence idx of the target vocabulary
    :param eos_idx: the end-of-sequence idx of the target vocabulary
    :param: max_examples_to_evaluate: after how many examples to break prediction, if none all are predicted
    """
    # Disable dropout and other regularization.
    model.eval()
    start_time = time.time()

    # Loop over the data.
    i = 0

    accuracies = []
    lm_losses = []

    no_shape_correct = 0
    no_color_correct = 0
    no_samples = 0
    for (input_sequence_instruct, input_sequence_action, input_lengths_instruct, input_lengths_action,
         derivation_spec, situation, situation_spec, target_sequence, target_lengths, agent_positions,
         target_positions, situation_image, unique_objects) in data_iterator:
        i += 1
        if max_examples_to_evaluate:
            if i*test_batch_size >= max_examples_to_evaluate:
                break

        if test_batch_size > 1:
            raise NotImplementedError("Eval with batch size > 1 not implemented")

        # sampled_target_objects = []
        # sampled_target_positions = []
        # sampled_target_position_symbols = []
        # sampled_target_position_tensors = []
        # for i, unique_objects_sample in enumerate(unique_objects):
        #     sampled_target_object = random.sample(unique_objects_sample, 1)[0]
        #     sampled_target_objects.append(sampled_target_object)
        #
        #     sampled_target_position = int(sampled_target_object["row"]) * int(situation_spec[i]['grid_size']) + \
        #                           int(sampled_target_object["column"])
        #     sampled_target_positions.append(sampled_target_position)
        #
        #     sampled_target_position_symbol = [
        #         dataset.item_to_array('<' + str(sampled_target_position) + '>', vocabulary='input'),
        #         dataset.item_to_array('<' + str(sampled_target_position) + '/>', vocabulary='input')]
        #     sampled_target_position_symbols.append(sampled_target_position_symbol)
        #
        #     sampled_target_position_tensor = torch.tensor([sampled_target_position],
        #                                               dtype=torch.long, device=device)  # .unsqueeze(dim=0)
        #     sampled_target_position_tensors.append(sampled_target_position_tensor)
        #
        # sampled_target_position_tensors = torch.stack(sampled_target_position_tensors)

        # scores, predicted_action_sequences, action_sequence_lengths, lm_loss = model.predict_actions_batch(
        #     input_batch,
        #     input_lengths,
        #     situation_batch,
        #     dataset.target_vocabulary.sos_idx,
        #     dataset.target_vocabulary.eos_idx,
        #     max_decoding_steps)

        # lm_losses.append(lm_loss)
        # for j in range(test_batch_size):
        #     accuracy = sequence_accuracy(predicted_action_sequences[j][:int(target_lengths[j])][1:-1].tolist(),
        #                                  target_batch[j][:int(target_lengths[j])][1:-1].tolist())
        #     accuracies.append(accuracy)
        sampled_target_object = random.sample(unique_objects[0], 1)
        sampled_target_position = int(sampled_target_object[0]["row"]) * int(situation_spec[0]['grid_size']) + \
                                  int(sampled_target_object[0]["column"])

        sampled_target_position_symbols = [
            dataset.item_to_array('<' + str(sampled_target_position) + '>', vocabulary='input'),
            dataset.item_to_array('<' + str(sampled_target_position) + '/>',
                                         vocabulary='input')]
        sampled_target_position_tensor = torch.tensor([sampled_target_position],
                                                      dtype=torch.long, device=device)  # .unsqueeze(dim=0)

        # Encode the input sequence for intruction gen.
        encoded_input = model.encode_input(commands_input=input_sequence_instruct,
                                           commands_lengths=input_lengths_instruct,
                                           situations_input=situation,
                                           target_positions=sampled_target_position_tensor,
                                           mode='instruct'
                                           )

        # get encoder lm perplexity
        logits = encoded_input['instruction_lm_logits']
        # _, _, vocabulary_size = logits.size()
        lm_loss = model.get_lm_loss(logits, input_sequence_instruct)

        ## sample sentences and eval targeted generations
        # potentially limit eval to scenes with n objects
        # if len(situation_spec[0]['placed_objects']) == 1:
        no_samples += 1
        encoded_situation = encoded_input['encoded_situations']
        # unsqueze for batch size
        sampled_sentence = model.sample(lm_vocab, encoded_situation, sos_idx, eos_idx,
                                        sampled_target_position_symbols, sampled_target_position_tensor)
        # get original sentence
        original_sent = [lm_vocab.idx_to_word(wid) for wid in input_sequence_instruct[0, 1:-1].tolist()]
        # get sampled obj specs
        color, shape, size = sampled_target_object[0]['color'], sampled_target_object[0]['shape'], \
                             sampled_target_object[0]['size']

        # eval targeted gen
        if color in sampled_sentence:
            no_color_correct += 1
        if shape in sampled_sentence:
            no_shape_correct += 1

        count = 0
        if i % 10 == 0 and count < 6:
            count += 1
            relevant_situation = Situation.from_representation(situation_spec[0])
            dataset.dataset.initialize_world(relevant_situation)
            rendered_image = dataset.dataset._world.render().getArray()

            logger.info("original sent # %s" % (i))
            logger.info(' '.join(original_sent))
            logger.info("Sampled sent # %s" % (i))
            logger.info(' '.join(sampled_sentence))
            logger.info("Color: %s Shape: %s Size: %s" % (color, shape, size))
            logger.info("\n")

            plt.figure()
            plt.imshow(rendered_image)
            plt.show()

        # Encode the input sequence for action seq prediction
        encoded_input = model.encode_input(commands_input=input_sequence_action,
                                           commands_lengths=input_lengths_action,
                                           situations_input=situation,
                                           target_positions=sampled_target_position_tensor,
                                           mode='action'
                                           )

        # For efficiency
        projected_keys_visual = model.visual_attention.key_layer(
            encoded_input["encoded_situations"])  # [bsz, situation_length, dec_hidden_dim]
        projected_keys_textual = model.textual_attention.key_layer(
            encoded_input["encoded_commands"]["encoder_outputs"])  # [max_input_length, bsz, dec_hidden_dim]

        # Iteratively decode the output.
        output_sequence = []
        contexts_situation = []
        hidden = model.attention_decoder.initialize_hidden(
            model.tanh(model.enc_hidden_to_dec_hidden(encoded_input["hidden_states"])))
        token = torch.tensor([sos_idx], dtype=torch.long, device=device)
        decoding_iteration = 0
        attention_weights_commands = []
        attention_weights_situations = []
        while token != eos_idx and decoding_iteration <= max_decoding_steps:
            (output, hidden, context_situation, attention_weights_command,
             attention_weights_situation) = model.decode_input(
                target_token=token, hidden=hidden, encoder_outputs=projected_keys_textual,
                input_lengths=input_lengths_action, encoded_situations=projected_keys_visual)
            output = F.log_softmax(output, dim=-1)
            token = output.max(dim=-1)[1]
            output_sequence.append(token.data[0].item())
            attention_weights_commands.append(attention_weights_command.tolist())
            attention_weights_situations.append(attention_weights_situation.tolist())
            contexts_situation.append(context_situation.unsqueeze(1))
            decoding_iteration += 1

        if output_sequence[-1] == eos_idx:
            output_sequence.pop()
            attention_weights_commands.pop()
            attention_weights_situations.pop()
        if model.auxiliary_task:
            target_position_scores = model.auxiliary_task_forward(torch.cat(contexts_situation, dim=1).sum(dim=1))
            auxiliary_accuracy_target = model.get_auxiliary_accuracy(target_position_scores, target_positions)
        else:
            auxiliary_accuracy_agent, auxiliary_accuracy_target = 0, 0
        yield (input_sequence_action, derivation_spec, situation_spec, output_sequence, target_sequence,
               attention_weights_commands, attention_weights_situations, auxiliary_accuracy_target, lm_loss)

    elapsed_time = time.time() - start_time
    logging.info("Predicted for {} examples.".format(i * test_batch_size))
    logging.info("Done predicting in {} seconds.".format(elapsed_time))

    if model.auxiliary_task:
        raise NotImplementedError("Auxiliary target currently not implemented.")
    auxiliary_accuracy_target = 0.0

    exact_match = [acc == 100 for acc in accuracies] * 100

    perplexity = np.exp(np.mean(lm_losses))
    return np.mean(accuracies), np.mean(exact_match), auxiliary_accuracy_target, perplexity
