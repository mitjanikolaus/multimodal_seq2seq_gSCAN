import json
import logging
import torch
import os

from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import LambdaLR

from seq2seq.model import Model
from seq2seq.gSCAN_dataset import GroundedScanDataset
from seq2seq.helpers import log_parameters
from seq2seq.evaluate import evaluate



logger = logging.getLogger(__name__)
use_cuda = True if torch.cuda.is_available() else False


def forward_pass_teacher_to_learner_curious(situation_batch, model_teacher, model_learner, training_set, max_decoding_steps, weight_lm_loss):
    # Generate instruction and target action sequence using teacher
    encoded_situations = model_teacher.encode_situations(situation_batch)

    _, sampled_sentences, sentence_lengths = model_teacher.sample_instructions(encoded_situations,
                                                                               training_set.target_vocabulary.sos_idx,
                                                                               training_set.target_vocabulary.eos_idx,
                                                                               max_decoding_steps)

    # Generate target action sequences using teacher
    _, teacher_action_sequences, teacher_action_sequence_lengths, _ = model_teacher.predict_actions_batch(
        sampled_sentences,
        sentence_lengths,
        situation_batch,
        training_set.target_vocabulary.sos_idx,
        training_set.target_vocabulary.eos_idx,
        max_decoding_steps)

    # logger.info(f"Teacher-generated instructions")
    # for sentence, action_sequence in zip(sampled_sentences, teacher_action_sequences):
    #     print(" ".join([instruction_vocab.idx_to_word(wid) for wid in sentence if
    #                 wid != training_set.target_vocabulary.pad_idx]))
    #     print(" ".join([action_vocab.idx_to_word(a) for a in action_sequence if a != action_vocab.pad_idx]))

    # Forward pass though learner using teacher-generated instruction and action targets.
    with torch.no_grad():
        target_scores_cur, _, _ = model_learner(commands_input=sampled_sentences,
                                             commands_lengths=sentence_lengths,
                                             situations_input=situation_batch,
                                             target_batch=teacher_action_sequences.clone(),
                                             target_lengths=teacher_action_sequence_lengths)

        curiosities = model_learner.get_curiousity_scores(target_scores_cur, teacher_action_sequences.clone())

    # Select batch_size/2 elements with top curiosity scores
    _, indices = curiosities.topk(int(curiosities.shape[0]/2))

    # Do second forward pass with gradient computation on filtered batch elements
    target_scores, _, instruction_lm_scores = model_learner(commands_input=sampled_sentences[indices],
                                                             commands_lengths=sentence_lengths[indices],
                                                             situations_input=situation_batch[indices],
                                                             target_batch=teacher_action_sequences[indices],
                                                             target_lengths=teacher_action_sequence_lengths[indices])

    actions_loss = model_learner.get_loss(target_scores, teacher_action_sequences[indices])

    # trim target because produced instructions might be shorter in filtered batch
    sampled_sentences = sampled_sentences[indices, :sentence_lengths[indices].max()]

    lm_loss = model_learner.get_lm_loss(instruction_lm_scores, sampled_sentences)
    loss = actions_loss + (weight_lm_loss * lm_loss)

    return loss, actions_loss, lm_loss, target_scores, teacher_action_sequences[indices]


def forward_pass_teacher_to_learner(situation_batch, model_teacher, model_learner, training_set, max_decoding_steps, weight_lm_loss):
    # Generate instruction and target action sequence using teacher
    encoded_situations = model_teacher.encode_situations(situation_batch)

    _, sampled_sentences, sentence_lengths = model_teacher.sample_instructions(encoded_situations,
                                                                               training_set.target_vocabulary.sos_idx,
                                                                               training_set.target_vocabulary.eos_idx,
                                                                               max_decoding_steps)

    # Generate target action sequences using teacher
    _, teacher_action_sequences, teacher_action_sequence_lengths, _ = model_teacher.predict_actions_batch(
        sampled_sentences,
        sentence_lengths,
        situation_batch,
        training_set.target_vocabulary.sos_idx,
        training_set.target_vocabulary.eos_idx,
        max_decoding_steps)

    # logger.info(f"Teacher-generated instructions")
    # for sentence, action_sequence in zip(sampled_sentences, teacher_action_sequences):
    #     print(" ".join([instruction_vocab.idx_to_word(wid) for wid in sentence if
    #                 wid != training_set.target_vocabulary.pad_idx]))
    #     print(" ".join([action_vocab.idx_to_word(a) for a in action_sequence if a != action_vocab.pad_idx]))

    # Forward pass though learner using teacher-generated instruction and action targets.
    target_scores, target_position_scores, instruction_lm_scores = model_learner(commands_input=sampled_sentences,
                                                                                 commands_lengths=sentence_lengths,
                                                                                 situations_input=situation_batch,
                                                                                 target_batch=teacher_action_sequences,
                                                                                 target_lengths=teacher_action_sequence_lengths)

    actions_loss = model_learner.get_loss(target_scores, teacher_action_sequences)

    lm_loss = model_learner.get_lm_loss(instruction_lm_scores, sampled_sentences)
    loss = actions_loss + (weight_lm_loss * lm_loss)

    return loss, actions_loss, lm_loss, target_scores, teacher_action_sequences

def forward_pass_learner_to_teacher_to_learner(situation_batch, model_teacher, model_learner, training_set, max_decoding_steps, weight_lm_loss):
    # Generate instruction and target action sequence using learner
    encoded_situations = model_learner.encode_situations(situation_batch)
    _, sampled_sentences, sentence_lengths = model_learner.sample_instructions(encoded_situations,
                                                                               training_set.target_vocabulary.sos_idx,
                                                                               training_set.target_vocabulary.eos_idx,
                                                                               max_decoding_steps)

    # Generate target action sequences using teacher
    _, teacher_action_sequences, teacher_action_sequence_lengths, _ = model_teacher.predict_actions_batch(
        sampled_sentences,
        sentence_lengths,
        situation_batch,
        training_set.target_vocabulary.sos_idx,
        training_set.target_vocabulary.eos_idx,
        max_decoding_steps)

    # logger.info(f"Learner-generated instructions with teacher action sequences:")
    # for sentence, action_sequence in zip(sampled_sentences, teacher_action_sequences):
    #     print(" ".join([instruction_vocab.idx_to_word(wid) for wid in sentence if
    #                 wid != training_set.target_vocabulary.pad_idx]))
    #     print(" ".join([action_vocab.idx_to_word(a) for a in action_sequence if a != action_vocab.pad_idx]))

    # Forward pass though learner using teacher-generated action targets.
    target_scores, target_position_scores, instruction_lm_scores = model_learner(
        commands_input=sampled_sentences, commands_lengths=sentence_lengths,
        situations_input=situation_batch, target_batch=teacher_action_sequences,
        target_lengths=teacher_action_sequence_lengths)

    actions_loss = model_learner.get_loss(target_scores, teacher_action_sequences)

    # it does not make sense to calculate LM loss (as the situation has been sampled by the learner)
    lm_loss = 0
    loss = actions_loss + (weight_lm_loss * lm_loss)

    return loss, actions_loss, lm_loss, target_scores, teacher_action_sequences


def train(data_path: str, data_directory: str, generate_vocabularies: bool, input_vocab_path: str,
          target_vocab_path: str, embedding_dimension: int, num_encoder_layers: int, encoder_dropout_p: float,
          encoder_bidirectional: bool, training_batch_size: int, test_batch_size: int, max_decoding_steps: int,
          num_decoder_layers: int, decoder_dropout_p: float, cnn_kernel_size: int, cnn_dropout_p: float,
          cnn_hidden_num_channels: int, simple_situation_representation: bool, decoder_hidden_size: int,
          encoder_hidden_size: int, learning_rate: float, adam_beta_1: float, adam_beta_2: float, lr_decay: float,
          lr_decay_steps: int, reset_optimizer: bool, resume_from_file_teacher: str, resume_from_file_learner: str,
          objective: str,max_training_iterations: int, output_directory: str,
          print_every: int, evaluate_every: int, conditional_attention: bool, auxiliary_task: bool,
          weight_target_loss: float, weight_lm_loss: float, attention_type: str, k: int, max_training_examples=None, seed=42, **kwargs):
    device = torch.device(type='cuda') if use_cuda else torch.device(type='cpu')
    cfg = locals().copy()

    torch.manual_seed(seed)

    logger.info("Loading Training set...")
    training_set = GroundedScanDataset(data_path, data_directory, split="train",
                                       input_vocabulary_file=input_vocab_path,
                                       target_vocabulary_file=target_vocab_path,
                                       generate_vocabulary=generate_vocabularies, k=k)
    training_set.read_dataset(max_examples=max_training_examples,
                              simple_situation_representation=simple_situation_representation)
    logger.info("Done Loading Training set.")
    logger.info("  Loaded {} training examples.".format(training_set.num_examples))
    logger.info("  Input vocabulary size training set: {}".format(training_set.input_vocabulary_size))
    logger.info("  Most common input words: {}".format(training_set.input_vocabulary.most_common(5)))
    logger.info("  Output vocabulary size training set: {}".format(training_set.target_vocabulary_size))
    logger.info("  Most common target words: {}".format(training_set.target_vocabulary.most_common(5)))

    if generate_vocabularies:
        training_set.save_vocabularies(input_vocab_path, target_vocab_path)
        logger.info("Saved vocabularies to {} for input and {} for target.".format(input_vocab_path, target_vocab_path))

    logger.info("Loading Dev. set...")
    test_set = GroundedScanDataset(data_path, data_directory, split="dev",
                                   input_vocabulary_file=input_vocab_path,
                                   target_vocabulary_file=target_vocab_path, generate_vocabulary=False, k=0)
    test_set.read_dataset(max_examples=None,
                          simple_situation_representation=simple_situation_representation)

    # Shuffle the test set to make sure that if we only evaluate max_testing_examples we get a random part of the set.
    test_set.shuffle_data()
    logger.info("Done Loading Dev. set.")

    model_teacher = Model(input_vocabulary_size=training_set.input_vocabulary_size,
                  target_vocabulary_size=training_set.target_vocabulary_size,
                  num_cnn_channels=training_set.image_channels,
                  input_padding_idx=training_set.input_vocabulary.pad_idx,
                  target_pad_idx=training_set.target_vocabulary.pad_idx,
                  target_eos_idx=training_set.target_vocabulary.eos_idx,
                  **cfg)
    model_teacher = model_teacher.cuda() if use_cuda else model_teacher

    model_learner = Model(input_vocabulary_size=training_set.input_vocabulary_size,
                  target_vocabulary_size=training_set.target_vocabulary_size,
                  num_cnn_channels=training_set.image_channels,
                  input_padding_idx=training_set.input_vocabulary.pad_idx,
                  target_pad_idx=training_set.target_vocabulary.pad_idx,
                  target_eos_idx=training_set.target_vocabulary.eos_idx,
                  **cfg)
    model_learner = model_learner.cuda() if use_cuda else model_learner

    # Load model and vocabularies if resuming.
    assert os.path.isfile(resume_from_file_teacher), "No checkpoint found at {}".format(resume_from_file_teacher)

    logger.info("Loading checkpoint from file at '{}'".format(resume_from_file_teacher))
    model_teacher.load_model(resume_from_file_teacher)

    log_parameters(model_learner)
    trainable_parameters = [parameter for parameter in model_learner.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=learning_rate, betas=(adam_beta_1, adam_beta_2))

    # Load model and vocabularies if resuming.
    start_iteration = 1
    scheduler_last_epoch = -1
    best_iteration = 1
    best_accuracy = 0
    best_exact_match = 0
    best_loss = float('inf')
    is_best = False
    if resume_from_file_learner:
        assert os.path.isfile(resume_from_file_learner), "No checkpoint found at {}".format(resume_from_file_learner)
        logger.info("Loading checkpoint from file at '{}'".format(resume_from_file_learner))
        optimizer_state_dict = model_learner.load_model(resume_from_file_learner)

        if not reset_optimizer:
            logger.info("Loading optimizer state dict from checkpoint.")
            optimizer.load_state_dict(optimizer_state_dict)
            scheduler_last_epoch = model_learner.trained_iterations
        start_iteration = model_learner.trained_iterations
        logger.info("Loaded checkpoint '{}' (iter {})".format(resume_from_file_learner, start_iteration))

    scheduler = LambdaLR(optimizer, lr_lambda=lambda t: lr_decay ** (t / lr_decay_steps), last_epoch=scheduler_last_epoch)

    instruction_vocab = training_set.get_vocabulary('input')
    action_vocab = training_set.get_vocabulary('target')

    training_iteration = start_iteration

    logger.info("Initial evaluation")
    with torch.no_grad():
        model_learner.eval()
        instruction_vocab = test_set.get_vocabulary('input')
        logger.info("Evaluating..")

        accuracy, exact_match, target_accuracy, perplexity = evaluate(
            test_set.get_data_iterator(batch_size=test_batch_size), model=model_learner, lm_vocab=instruction_vocab,
            max_decoding_steps=max_decoding_steps, pad_idx=test_set.target_vocabulary.pad_idx,
            sos_idx=test_set.target_vocabulary.sos_idx,
            eos_idx=test_set.target_vocabulary.eos_idx,
            max_examples_to_evaluate=kwargs["max_testing_examples"], dataset=test_set,
            test_batch_size=test_batch_size)
        logger.info("  Evaluation Accuracy: %5.2f Exact Match: %5.2f "
                    " Target Accuracy: %5.2f Perplexity: %5.2f"
                    % (accuracy, exact_match, target_accuracy, perplexity))
        if exact_match > best_exact_match:
            is_best = True
            best_accuracy = accuracy
            best_exact_match = exact_match
            model_learner.update_state(accuracy=accuracy, exact_match=exact_match, is_best=is_best)
        file_name = f"checkpoint_iter_{str(training_iteration)}.pth.tar"
        model_learner.save_checkpoint(file_name=file_name, is_best=False,
                                      optimizer_state_dict=optimizer.state_dict())
        if is_best:
            model_learner.save_checkpoint(file_name=file_name, is_best=is_best,
                                          optimizer_state_dict=optimizer.state_dict())

    logger.info("Training starts..")
    train_logs = {}
    train_logs_file = os.path.join(output_directory, "train_logs.json")
    eval_logs = {}
    eval_logs_file = os.path.join(output_directory, "eval_logs.json")
    while training_iteration < max_training_iterations:
        # Shuffle the dataset and loop over it.
        training_set.shuffle_data()
        for _, _, _, situation_batch, _, _, _, _, _, _ in training_set.get_data_iterator(
                batch_size=training_batch_size
        ):

            is_best = False
            model_learner.train()
            model_teacher.eval()

            if auxiliary_task:
                raise NotImplementedError()

            if objective == OBJECTIVE_TEACHER_TO_LEARNER:
                loss, actions_loss, lm_loss, target_scores, target_action_sequences = forward_pass_teacher_to_learner(
                    situation_batch, model_teacher, model_learner, training_set, max_decoding_steps, weight_lm_loss)

            if objective == OBJECTIVE_TEACHER_TO_LEARNER_CURIOUS:
                loss, actions_loss, lm_loss, target_scores, target_action_sequences = forward_pass_teacher_to_learner_curious(
                    situation_batch, model_teacher, model_learner, training_set, max_decoding_steps, weight_lm_loss)

            elif objective == OBJECTIVE_LEARNER_TO_TEACHER_TO_LEARNER:
                loss, actions_loss, lm_loss, target_scores, target_action_sequences = \
                    forward_pass_learner_to_teacher_to_learner(situation_batch, model_teacher, model_learner,
                                                               training_set, max_decoding_steps, weight_lm_loss)
            elif objective == OBJECTIVE_TL_LTL:
                _, actions_loss_tl, lm_loss, _, _ = forward_pass_teacher_to_learner(
                    situation_batch, model_teacher, model_learner, training_set, max_decoding_steps, weight_lm_loss)

                _, actions_loss_ltl, _, target_scores, target_action_sequences = \
                    forward_pass_learner_to_teacher_to_learner(situation_batch, model_teacher, model_learner,
                                                               training_set, max_decoding_steps, weight_lm_loss)
                # TODO average or add?
                actions_loss = (actions_loss_tl + actions_loss_ltl) / 2

                loss = actions_loss + (weight_lm_loss * lm_loss)

            else:
                raise NotImplementedError("Unknown objective: ", objective)


            # Backward pass and update model parameters.
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            model_learner.update_state(is_best=is_best)

            # Print current metrics.
            if training_iteration % print_every == 0:
                accuracy, exact_match = model_learner.get_metrics(target_scores, target_action_sequences)
                if auxiliary_task:
                    raise NotImplementedError()
                else:
                    auxiliary_accuracy_target = 0.
                learning_rate = scheduler.get_last_lr()[0]
                logger.info("Iteration %08d, loss %8.4f, actions_loss %8.4f, lm_loss %8.4f, accuracy %5.2f, exact match %5.2f, learning_rate %.5f,"
                            " aux. accuracy target pos %5.2f" % (training_iteration, loss, actions_loss, lm_loss, accuracy, exact_match,
                                                                 learning_rate, auxiliary_accuracy_target))

                logs = {"loss": loss.item(), "actions_loss": actions_loss.item(), "lm_loss": lm_loss.item(),
                        "accuracy": accuracy, "exact_match": exact_match, "learning_rate": learning_rate}
                if objective == OBJECTIVE_TL_LTL:
                    logs["actions_loss_tl"] = actions_loss_tl.item()
                    logs["actions_loss_ltl"] = actions_loss_ltl.item()
                train_logs[training_iteration] = logs
                json.dump(train_logs, open(train_logs_file, mode='w'))

            # Evaluate on test set.
            if training_iteration % evaluate_every == 0:
                with torch.no_grad():
                    model_learner.eval()
                    instruction_vocab = test_set.get_vocabulary('input')
                    logger.info("Evaluating..")

                    accuracy, exact_match, target_accuracy, perplexity = evaluate(
                        test_set.get_data_iterator(batch_size=test_batch_size), model=model_learner, lm_vocab=instruction_vocab,
                        max_decoding_steps=max_decoding_steps, pad_idx=test_set.target_vocabulary.pad_idx,
                        sos_idx=test_set.target_vocabulary.sos_idx,
                        eos_idx=test_set.target_vocabulary.eos_idx,
                        max_examples_to_evaluate=kwargs["max_testing_examples"], dataset=test_set,
                        test_batch_size=test_batch_size)
                    logger.info("  Evaluation Accuracy: %5.2f Exact Match: %5.2f "
                                " Target Accuracy: %5.2f Perplexity: %5.2f"
                                % (accuracy, exact_match, target_accuracy, perplexity))
                    if exact_match > best_exact_match:
                        is_best = True
                        best_accuracy = accuracy
                        best_exact_match = exact_match
                        model_learner.update_state(accuracy=accuracy, exact_match=exact_match, is_best=is_best)
                    file_name = f"checkpoint_iter_{str(training_iteration)}.pth.tar"
                    model_learner.save_checkpoint(file_name=file_name, is_best=False,
                                                  optimizer_state_dict=optimizer.state_dict())
                    if is_best:
                        model_learner.save_checkpoint(file_name=file_name, is_best=is_best,
                                              optimizer_state_dict=optimizer.state_dict())

                    logs = {"accuracy": accuracy, "exact_match": exact_match, "perplexity": perplexity}
                    eval_logs[training_iteration] = logs
                    json.dump(eval_logs, open(eval_logs_file, mode='w'))

            training_iteration += 1
            if training_iteration > max_training_iterations:
                break
    logger.info("Finished training.")


import argparse
import logging
import os
import torch

from seq2seq.gSCAN_dataset import GroundedScanDataset
from seq2seq.model import Model
from seq2seq.predict import predict_and_save

OBJECTIVE_TEACHER_TO_LEARNER = "teacher-to-learner"
OBJECTIVE_TEACHER_TO_LEARNER_CURIOUS = "teacher-to-learner-curious"
OBJECTIVE_LEARNER_TO_TEACHER_TO_LEARNER = "learner-to-teacher-to-learner"
OBJECTIVE_TL_LTL = "both"

FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT, level=logging.DEBUG,
                    datefmt="%Y-%m-%d %H:%M")
logger = logging.getLogger(__name__)
use_cuda = True if torch.cuda.is_available() else False

if use_cuda:
    logger.info("Using CUDA.")
    logger.info("Cuda version: {}".format(torch.version.cuda))

parser = argparse.ArgumentParser(description="Sequence to sequence models for Grounded SCAN")

# General arguments
parser.add_argument("--mode", type=str, default="run_tests", help="train, test or predict", required=True)
parser.add_argument("--output_directory", type=str, default="output", help="In this directory the models will be "
                                                                           "saved. Will be created if doesn't exist.")
parser.add_argument("--resume_from_file_teacher", type=str, default="", help="Full path to previously saved teacher "
                                                                             "model to load.")
parser.add_argument("--resume_from_file_learner", type=str, default="", help="Full path to previously saved learner "
                                                                             "model to load.")
parser.add_argument("--objective", type=str, default=OBJECTIVE_TEACHER_TO_LEARNER,
                    choices=[OBJECTIVE_TEACHER_TO_LEARNER, OBJECTIVE_LEARNER_TO_TEACHER_TO_LEARNER, OBJECTIVE_TL_LTL,
                             OBJECTIVE_TEACHER_TO_LEARNER_CURIOUS],
                    help="Which objective to use")


# Data arguments
parser.add_argument("--split", type=str, default="test", help="Which split to get from Grounded Scan.")
parser.add_argument("--data_directory", type=str, default="data/uniform_dataset", help="Path to folder with data.")
parser.add_argument("--input_vocab_path", type=str, default="training_input_vocab.txt",
                    help="Path to file with input vocabulary as saved by Vocabulary class in gSCAN_dataset.py")
parser.add_argument("--target_vocab_path", type=str, default="training_target_vocab.txt",
                    help="Path to file with target vocabulary as saved by Vocabulary class in gSCAN_dataset.py")
parser.add_argument("--generate_vocabularies", dest="generate_vocabularies", default=False, action="store_true",
                    help="Whether to generate vocabularies based on the data.")
parser.add_argument("--load_vocabularies", dest="generate_vocabularies", default=True, action="store_false",
                    help="Whether to use previously saved vocabularies.")

# Training and learning arguments
parser.add_argument("--training_batch_size", type=int, default=200)
parser.add_argument("--k", type=int, default=0, help="How many examples from the adverb_1 split to move to train.")
parser.add_argument("--test_batch_size", type=int, default=200, help="Test batch size.")
parser.add_argument("--max_training_examples", type=int, default=None, help="If None all are used.")
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument('--lr_decay', type=float, default=0.9)
parser.add_argument('--lr_decay_steps', type=float, default=20000)
parser.add_argument("--adam_beta_1", type=float, default=0.9)
parser.add_argument("--adam_beta_2", type=float, default=0.999)
parser.add_argument("--print_every", type=int, default=100)
parser.add_argument("--evaluate_every", type=int, default=1000, help="How often to evaluate the model by decoding the "
                                                                     "test set (without teacher forcing).")
parser.add_argument("--max_training_iterations", type=int, default=100000)
parser.add_argument("--weight_target_loss", type=float, default=0.3, help="Only used if --auxiliary_task set.")
parser.add_argument("--weight_lm_loss", type=float, default=1, help="Weight for the LM target loss.")

parser.add_argument("--reset_optimizer",
                    dest="reset_optimizer",
                    default=False,
                    action="store_true",
                    help="Reset the optimizer when continuing training (instead of using optimizer config from "
                         "checkpoint).")

# Testing and predicting arguments
parser.add_argument("--max_testing_examples", type=int, default=None)
parser.add_argument("--splits", type=str, default="test", help="comma-separated list of splits to predict for.")
parser.add_argument("--max_decoding_steps", type=int, default=120, help="After 120 decoding steps, the decoding process "
                                                                       "is stopped regardless of whether an EOS token "
                                                                       "was generated.")
parser.add_argument("--output_file_name", type=str, default="predict.json")

# Situation Encoder arguments
parser.add_argument("--simple_situation_representation", dest="simple_situation_representation", default=True,
                    action="store_true", help="Represent the situation with 1 vector per grid cell. "
                                              "For more information, read grounded SCAN documentation.")
parser.add_argument("--image_situation_representation", dest="simple_situation_representation", default=False,
                    action="store_false", help="Represent the situation with the full gridworld RGB image. "
                                               "For more information, read grounded SCAN documentation.")
parser.add_argument("--cnn_hidden_num_channels", type=int, default=50)
parser.add_argument("--cnn_kernel_size", type=int, default=7, help="Size of the largest filter in the world state "
                                                                   "model.")
parser.add_argument("--cnn_dropout_p", type=float, default=0.1, help="Dropout applied to the output features of the "
                                                                     "world state model.")
parser.add_argument("--auxiliary_task", dest="auxiliary_task", default=False, action="store_true",
                    help="If set to true, the model predicts the target location from the joint attention over the "
                         "input instruction and world state.")
parser.add_argument("--no_auxiliary_task", dest="auxiliary_task", default=True, action="store_false")

parser.add_argument("--test_dir", type=str, default="", help="Directory of saved models to load for testing.")
parser.add_argument("--test_checkpoints", type=int, nargs="+", default=[], help="Checkpoints to evaluate.")


# Command Encoder arguments
parser.add_argument("--embedding_dimension", type=int, default=25)
parser.add_argument("--num_encoder_layers", type=int, default=1)
parser.add_argument("--encoder_hidden_size", type=int, default=100)
parser.add_argument("--encoder_dropout_p", type=float, default=0.3, help="Dropout on instruction embeddings and LSTM.")
parser.add_argument("--encoder_bidirectional", dest="encoder_bidirectional", default=False, action="store_true")
parser.add_argument("--encoder_unidirectional", dest="encoder_bidirectional", default=True, action="store_false")

# Decoder arguments
parser.add_argument("--num_decoder_layers", type=int, default=1)
parser.add_argument("--attention_type", type=str, default='bahdanau', choices=['bahdanau', 'luong'],
                    help="Luong not properly implemented.")
parser.add_argument("--decoder_dropout_p", type=float, default=0.3, help="Dropout on decoder embedding and LSTM.")
parser.add_argument("--decoder_hidden_size", type=int, default=100)
parser.add_argument("--conditional_attention", dest="conditional_attention", default=True, action="store_true",
                    help="If set to true joint attention over the world state conditioned on the input instruction is"
                         " used.")
parser.add_argument("--no_conditional_attention", dest="conditional_attention", default=False, action="store_false")

# Other arguments
parser.add_argument("--seed", type=int, default=42)


def main(flags):
    for argument, value in flags.items():
        logger.info("{}: {}".format(argument, value))

    if not os.path.exists(flags["output_directory"]):
        os.mkdir(os.path.join(os.getcwd(), flags["output_directory"]))

    if not flags["simple_situation_representation"]:
        raise NotImplementedError("Full RGB input image not implemented. Implement or set "
                                  "--simple_situation_representation")
    # Some checks on the flags
    if flags["generate_vocabularies"]:
        assert flags["input_vocab_path"] and flags["target_vocab_path"], "Please specify paths to vocabularies to save."

    data_path = os.path.join(flags["data_directory"], "dataset.txt")
    if flags["mode"] == "train":
        train(data_path=data_path, **flags)
    elif flags["mode"] == "test":
        assert os.path.exists(os.path.join(flags["data_directory"], flags["input_vocab_path"])) and os.path.exists(
            os.path.join(flags["data_directory"], flags["target_vocab_path"])), \
            "No vocabs found at {} and {}".format(flags["input_vocab_path"], flags["target_vocab_path"])
        splits = flags["splits"].split(",")

        for checkpoint in flags["test_checkpoints"]:
            exact_match_accs = {}
            for split in splits:
                logger.info("Loading {} dataset split...".format(split))
                test_set = GroundedScanDataset(data_path, flags["data_directory"], split=split,
                                               input_vocabulary_file=flags["input_vocab_path"],
                                               target_vocabulary_file=flags["target_vocab_path"], generate_vocabulary=False,
                                               k=flags["k"], log_stats=False)
                test_set.read_dataset(max_examples=flags["max_testing_examples"],
                                      simple_situation_representation=flags["simple_situation_representation"])
                logger.info("Done Loading {} dataset split.".format(flags["split"]))
                logger.info("  Loaded {} examples.".format(test_set.num_examples))
                # logger.info("  Input vocabulary size: {}".format(test_set.input_vocabulary_size))
                # logger.info("  Output vocabulary size: {}".format(test_set.target_vocabulary_size))

                model = Model(input_vocabulary_size=test_set.input_vocabulary_size,
                              target_vocabulary_size=test_set.target_vocabulary_size,
                              num_cnn_channels=test_set.image_channels,
                              input_padding_idx=test_set.input_vocabulary.pad_idx,
                              target_pad_idx=test_set.target_vocabulary.pad_idx,
                              target_eos_idx=test_set.target_vocabulary.eos_idx,
                              **flags)
                model = model.cuda() if use_cuda else model

                # Load model and vocabularies if resuming.
                test_checkpoint_path = os.path.join(flags["test_dir"], f"checkpoint_iter_{checkpoint}.pth.tar")
                assert os.path.isfile(test_checkpoint_path), "No checkpoint found at {}".format(test_checkpoint_path)
                logger.info("Loading checkpoint from file at '{}'".format(test_checkpoint_path))
                model.load_model(test_checkpoint_path)
                start_iteration = model.trained_iterations
                logger.info("Loaded checkpoint '{}' (iter {})".format(test_checkpoint_path, start_iteration))
                output_file_name = "_".join([split, flags["output_file_name"]])
                output_file_path = os.path.join(flags["test_dir"], output_file_name)
                instruction_vocab = test_set.get_vocabulary('input')
                _, exact_match_acc = predict_and_save(dataset=test_set, model=model,
                                                                output_file_path=output_file_path,
                                                                lm_vocab=instruction_vocab, save_predictions=False,
                                                                **flags)
                exact_match_accs[split] = exact_match_acc
            logger.info("\n\n\nAccuracies overview:")
            for split, acc in exact_match_accs.items():
                logger.info(f"{split}: {acc:.2f}")

            accuracies_file = test_checkpoint_path.replace(".pth.tar", "_accuracies.json")
            json.dump(exact_match_accs, open(accuracies_file, mode='w'))

    elif flags["mode"] == "predict":
        raise NotImplementedError()
    else:
        raise ValueError("Wrong value for parameters --mode ({}).".format(flags["mode"]))


if __name__ == "__main__":
    input_flags = vars(parser.parse_args())
    main(flags=input_flags)

