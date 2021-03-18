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


def train(data_path: str, data_directory: str, generate_vocabularies: bool, input_vocab_path: str,
          target_vocab_path: str, embedding_dimension: int, num_encoder_layers: int, encoder_dropout_p: float,
          encoder_bidirectional: bool, training_batch_size: int, test_batch_size: int, max_decoding_steps: int,
          num_decoder_layers: int, decoder_dropout_p: float, cnn_kernel_size: int, cnn_dropout_p: float,
          cnn_hidden_num_channels: int, simple_situation_representation: bool, decoder_hidden_size: int,
          encoder_hidden_size: int, learning_rate: float, adam_beta_1: float, adam_beta_2: float, lr_decay: float,
          lr_decay_steps: int, resume_from_file_teacher: str, resume_from_file_learner: str, max_training_iterations: int, output_directory: str,
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
    scheduler = LambdaLR(optimizer,
                         lr_lambda=lambda t: lr_decay ** (t / lr_decay_steps))

    # Load model and vocabularies if resuming.
    start_iteration = 1
    best_iteration = 1
    best_accuracy = 0
    best_exact_match = 0
    best_loss = float('inf')
    if resume_from_file_learner:
        assert os.path.isfile(resume_from_file_learner), "No checkpoint found at {}".format(resume_from_file_learner)
        logger.info("Loading checkpoint from file at '{}'".format(resume_from_file_learner))
        optimizer_state_dict = model_learner.load_model(resume_from_file_learner)
        optimizer.load_state_dict(optimizer_state_dict)
        start_iteration = model_learner.trained_iterations
        logger.info("Loaded checkpoint '{}' (iter {})".format(resume_from_file_learner, start_iteration))

    logger.info("Training starts..")
    training_iteration = start_iteration
    while training_iteration < max_training_iterations:

        # Shuffle the dataset and loop over it.
        training_set.shuffle_data()
        for (input_batch, input_lengths, _, situation_batch, _, _, _, _, _, _) in training_set.get_data_iterator(
                batch_size=training_batch_size):
            is_best = False
            model_learner.train()
            model_teacher.eval()

            # Generate instruction and target action sequence using teacher
            encoded_situations = model_teacher.encode_situations(situation_batch)

            instruction_vocab = training_set.get_vocabulary('input')
            action_vocab = training_set.get_vocabulary('target')
            _, sampled_sentences, sentence_lengths = model_teacher.sample_instructions(encoded_situations,
                                                                                       training_set.target_vocabulary.sos_idx,
                                                                                       training_set.target_vocabulary.eos_idx,
                                                                                       max_decoding_steps)

            # Generate target action sequences using teacher
            _, teacher_action_sequences, teacher_action_sequence_lengths = model_teacher.predict_actions_batch(sampled_sentences,
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
            target_scores, target_position_scores, instruction_lm_scores = model_learner(commands_input=sampled_sentences, commands_lengths=sentence_lengths,
                                                          situations_input=situation_batch, target_batch=teacher_action_sequences,
                                                          target_lengths=teacher_action_sequence_lengths)

            actions_loss = model_learner.get_loss(target_scores, teacher_action_sequences)

            lm_loss = model_learner.get_lm_loss(instruction_lm_scores, sampled_sentences)
            loss = actions_loss + (weight_lm_loss * lm_loss)

            if auxiliary_task:
                raise NotImplementedError()
            else:
                target_loss = 0
            loss += weight_target_loss * target_loss

            # Backward pass and update model parameters.
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            model_learner.update_state(is_best=is_best)

            # Print current metrics.
            if training_iteration % print_every == 0:
                accuracy, exact_match = model_learner.get_metrics(target_scores, teacher_action_sequences)
                if auxiliary_task:
                    raise NotImplementedError()
                else:
                    auxiliary_accuracy_target = 0.
                learning_rate = scheduler.get_lr()[0]
                logger.info("Iteration %08d, loss %8.4f, actions_loss %8.4f, lm_loss %8.4f, accuracy %5.2f, exact match %5.2f, learning_rate %.5f,"
                            " aux. accuracy target pos %5.2f" % (training_iteration, loss, actions_loss, lm_loss, accuracy, exact_match,
                                                                 learning_rate, auxiliary_accuracy_target))

            # Evaluate on test set.
            if training_iteration % evaluate_every == 0:
                with torch.no_grad():
                    model_learner.eval()
                    instruction_vocab = test_set.get_vocabulary('input')
                    logger.info("Evaluating..")

                    accuracy, exact_match, target_accuracy, perplexity = evaluate(
                        test_set.get_data_iterator(batch_size=1), model=model_learner, lm_vocab=instruction_vocab,
                        max_decoding_steps=max_decoding_steps, pad_idx=test_set.target_vocabulary.pad_idx,
                        sos_idx=test_set.target_vocabulary.sos_idx,
                        eos_idx=test_set.target_vocabulary.eos_idx,
                        max_examples_to_evaluate=kwargs["max_testing_examples"], dataset=test_set.dataset)
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
parser.add_argument("--training_batch_size", type=int, default=50)
parser.add_argument("--k", type=int, default=0, help="How many examples from the adverb_1 split to move to train.")
parser.add_argument("--test_batch_size", type=int, default=1, help="Currently only 1 supported due to decoder.")
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


# Testing and predicting arguments
parser.add_argument("--max_testing_examples", type=int, default=None)
parser.add_argument("--splits", type=str, default="test", help="comma-separated list of splits to predict for.")
parser.add_argument("--max_decoding_steps", type=int, default=30, help="After 30 decoding steps, the decoding process "
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

    if flags["test_batch_size"] > 1:
        raise NotImplementedError("Test batch size larger than 1 not implemented.")

    data_path = os.path.join(flags["data_directory"], "dataset.txt")
    if flags["mode"] == "train":
        train(data_path=data_path, **flags)
    elif flags["mode"] == "test":
        raise NotImplementedError()
    elif flags["mode"] == "predict":
        raise NotImplementedError()
    else:
        raise ValueError("Wrong value for parameters --mode ({}).".format(flags["mode"]))


if __name__ == "__main__":
    input_flags = vars(parser.parse_args())
    main(flags=input_flags)

