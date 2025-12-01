#!/usr/bin/env python3
"""
Command-line interface for training and evaluating HMM and CRF taggers.
"""
import argparse
import logging
from pathlib import Path
import os, os.path, datetime
from typing import Callable, Tuple, Union

import torch, torch.backends.mps
from corpus import TaggedCorpus                
from eval import model_cross_entropy, viterbi_error_rate, write_tagging, log as eval_log
from hmm import HiddenMarkovModel
from crf import ConditionalRandomField

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    #####
    filegroup = parser.add_argument_group("Model and data files")
    #####

    filegroup.add_argument("input", type=str, help="input sentences for evaluation (labeled dev set or test set)")

    filegroup.add_argument(
        "-m",
        "--model",
        type=str,
        help="load from here if not --train, save to here if --train (this is a .pkl file)"
    )

    filegroup.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        help="continue training from this pretrained model or checkpoint, rather than from a random initialization"
    )

    filegroup.add_argument(
        "-t",
        "--train",
        type=str,
        nargs="+",
        help="training data files to train the model further"
    )

    filegroup.add_argument(
        "-o",
        "--output_file",
        type=str,
        default=None,
        help="where to save the prediction outputs (defaults to model.output if --model is specified)"
    )

    filegroup.add_argument(
        "-e",
        "--eval_file",
        type=str,
        default=None,
        help="where to log intermediate and final evaluation, which may also be logged to stderr (defaults to model.eval if --model is specified)"
    )


    #####
    traingroup = parser.add_argument_group("Training procedure")
    #####

    traingroup.add_argument(
        "--loss",
        type=str,
        default="cross_entropy",
        choices=['cross_entropy','viterbi_error'],
        help="loss function to evaluate on during training and final evaluation"
    )

    traingroup.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="tolerance for detecting convergence of loss function during training"
    )

    traingroup.add_argument(
        "--max_steps",
        type=int,
        default=50000,
        help="maximum number of training steps (measured in sentences, not epochs or minibatches)"
    )

    #####
    modelgroup = parser.add_argument_group("Tagging model structure")
    #####

    modelgroup.add_argument(
        "-u",
        "--unigram",
        action="store_true",
        default=False,
        help="model should be only a unigram HMM or CRF (baseline)"
    )
    
    modelgroup.add_argument(
        "--crf",
        action="store_true",
        default=False,
        help="model should be a CRF rather than an HMM"
    )

    modelgroup.add_argument(
        "-l",
        "--lexicon",
        type=str,
        default=None,
        help="neuralized model (HW7) with word embeddings drawn from this lexicon" 
    )

    modelgroup.add_argument(
        "-L",
        "--problex",
        action="store_true",
        default=False,
        help="neuralized model (HW7) with word embeddings based on probabilities in training data (may be combined with --lexicon)" 
    )

    modelgroup.add_argument(
        "-a",
        "--awesome",
        action="store_true",
        default=False,
        help="model should use extra improvements"
    )

    #####
    # top-level options
    #####
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet",   dest="logging_level", action="store_const", const=logging.WARNING
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=['cpu','cuda','mps'],
        help="device to use for PyTorch (cpu or cuda, or mps if you are on a mac)"
    )

    #####
    hmmgroup = parser.add_argument_group("HMM options (ignored for CRF)")
    #####

    hmmgroup.add_argument(
        "-λ",
        "--lambda",
        dest="λ",    # can't write "args.lambda" in code since it's a Python keyword
        type=float,
        default=0,
        help="lambda for add-lambda smoothing in the HMM M-step"
    )

    #####
    crfgroup = parser.add_argument_group("CRF or gradient-training options (ignored for HMM)")
    #####

    crfgroup.add_argument(
        "-r",
        "--rnn_dim",
        type=int,
        default=None,
        help="model should encode context using recurrent neural nets with this hidden-state dimensionality (>= 0)"
    )

    crfgroup.add_argument(
        "--reg",
        type=float,
        default=0.0,
        help="l2 regularization coefficient during training (default 0)"
    )

    crfgroup.add_argument(
        "--lr",
        type=float,
        default=0.05,
        help="learning rate during CRF training (default 0.05)"
    )

    crfgroup.add_argument(
        "--batch_size",
        type=int,
        default=30,
        help="mini-batch size: number of training sentences per gradient update (default 30)"
    )

    crfgroup.add_argument(
        "--eval_interval",
        type=int,
        default=2000,
        help="how often to evaluate the model (after training on this many sentences) (default 2000)"
    )

    args = parser.parse_args()

    ### Any arg manipulation and checking goes here

    args.load_path = None    # new attribute
    if args.model:
        if args.train:
            # --model says where to save
            if os.path.exists(args.model):
                log.warning(f"Warning: {args.model} already exists; will be overwritten at end of training")
                if not args.checkpoint:
                    log.warning(f"Warning: won't load {args.model}: use --checkpoint for that")
        else:
            # --model says where to load from
            args.load_path = args.model
    elif args.train:
        log.warning(f"Warning: model won't be saved at end of training (use --model for that)")

    if args.checkpoint:
        if args.train:
            # --checkpoint says where to load from
            args.load_path = args.checkpoint
        else:
            parser.error("--checkpoint can only be used with --train")

    if args.load_path and not os.path.exists(args.load_path):
            parser.error(f"file {args.load_path} doesn't exist")
      
    # default paths for saving the tagging outputs and eval
    if args.model:
        resultname = os.path.splitext(args.model)[0]
        resultname += "_" + os.path.splitext(os.path.basename(args.input))[0]
        if not args.output_file:
            args.output_file = resultname + ".output"
        if not args.eval_file:
            args.eval_file = resultname + ".eval"
    if not args.output_file:
        log.warning(f"Warning: won't save tagging output (use --output_file for that)")
    if not args.eval_file:
        log.warning(f"Warning: won't save evaluation messages (use --eval_file for that)")      

    # What kind of model should we create?  Store it in args.new_model_class.
    if args.load_path:      # don't need to create a new model
        args.new_model_class = None 
    elif not args.crf:      # create an HMM
        if args.rnn_dim or args.lexicon or args.problex:
            raise NotImplementedError("No neural HMM implemented (possible, but not required for these homeworks)")
        else:
            args.new_model_class = HiddenMarkovModel
    else:                   # create some sort of CRF
        if args.rnn_dim or args.lexicon or args.problex:
            from crf_neural import ConditionalRandomFieldNeural  # module provided with hw-rnn homework
            args.new_model_class = ConditionalRandomFieldNeural
        else: 
            args.new_model_class = ConditionalRandomField          

    return args

def main() -> None:
    args = parse_args()
    
    # Which messages to log (and how to log them).  
    # The logging module works a bit counterintuitively here, but
    # here's how to set it up: don't let loggers filter by level,
    # but instead have their handlers do so.
    logging.basicConfig(level=logging.DEBUG)  # log everything, and install default handler (print to stderr)
    for h in logging.getLogger().handlers:
        h.setLevel(args.logging_level)  # make default handler quieter
    if args.eval_file:
        # handler for eval logger. It appends to eval_file, since
        # we might be resuming training from a checkpoint, or evaluating
        # a model on multiple files.
        h = logging.FileHandler(args.eval_file, mode='a')
        h.setLevel(logging.INFO)  # make this handler noisy enough
        eval_log.addHandler(h)

    # Specify hardware device where all tensors should be computed and
    # stored.  This will give errors unless you have such a device.
    # E.g., 'gpu' will work in a Kaggle Notebook where you have
    # turned on GPU acceleration.
    if args.device == 'mps':
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                logging.critical("MPS not available because the current PyTorch install was not "
                    "built with MPS enabled.")
            else:
                logging.critical("MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine.")
            exit(1)
    torch.set_default_device(args.device)
        
    # Load or create the model, and load the training corpus.
    train_paths = [Path(t) for t in args.train] if args.train else []
    new_model_class = args.new_model_class
    known_vocab = None   # we may need this
    if args.load_path:
        # load an existing model and use its vocab/tagset/lexicon
        model = HiddenMarkovModel.load(args.load_path, device=args.device)  # HMM is ancestor of all classes
        for option in 'crf', 'unigram', 'rnn_dim', 'lexicon', 'problex', 'awesome': 
           if getattr(args, option):
               log.warning(f"Ignoring --{option} in favor of loaded model")       
        # integerize the training corpus using the vocab/tagset of the model
        train_corpus = TaggedCorpus(*train_paths, tagset=model.tagset, vocab=model.vocab)
    else:
        # Build a new model of the required class from scratch, building vocab/tagset from training corpus.
        assert new_model_class is not None
        train_corpus = TaggedCorpus(*train_paths)
        if not getattr(new_model_class, 'neural', False):
            # simple non-neural model
            model = new_model_class(train_corpus.tagset, train_corpus.vocab, 
                                    unigram=args.unigram)
        else:
            # For a neural model, we have to call the constructor with extra arguments.
            # We start by making a lexicon of word embeddings.
            from lexicon import build_lexicon  # module provided with hw-rnn homework
            if args.lexicon:
                # The user gave us a file of pretrained lexical embeddings.
                known_vocab = train_corpus.vocab   # save training vocab, since it may be replaced
                lexicon = build_lexicon(train_corpus, 
                                        embeddings_file=Path(args.lexicon) if args.lexicon else None, 
                                        newvocab=TaggedCorpus(Path(args.input)).vocab,  # add only eval words from file
                                        problex=args.problex)
            else:
                # No lexicon was specified, so default to simpler embeddings of the training words.
                if args.problex:
                    lexicon = build_lexicon(train_corpus, problex=args.problex)
                else:
                    # Simple one-hot embeddings are our final fallback if nothing else was specified.
                    lexicon = build_lexicon(train_corpus, one_hot=True)

            # Now create the model.
            model = new_model_class(train_corpus.tagset, train_corpus.vocab, 
                                    rnn_dim=(args.rnn_dim or 0), lexicon=lexicon,   # neural model args
                                    unigram=args.unigram)
    
    # Load the input data (eval corpus), using the same vocab and tagset.
    eval_corpus = TaggedCorpus(Path(args.input), tagset=model.tagset, vocab=model.vocab)
    
    # Construct the primary loss function on the eval corpus.
    # This will be monitored throughout training and used for early stopping.
    loss = lambda x: model_cross_entropy(x, eval_corpus)
    other_loss = lambda x: viterbi_error_rate(x, eval_corpus, show_cross_entropy=False,
                                              known_vocab=known_vocab or train_corpus.vocab)  # training words only
    if args.loss == 'cross_entropy': 
        pass
    elif args.loss == 'viterbi_error':   # only makes sense if eval_corpus is supervised
        loss, other_loss = other_loss, loss   # swap

    # Train on the training corpus, if given.    
    if train_corpus:
        if isinstance(model, ConditionalRandomField):
            for option in 'λ':
                if getattr(args, option):
                    log.warning(f"Ignoring --{option} since we're training by SGD and don't have an M step ")      
            model.train(corpus=train_corpus,
                        loss=loss,
                        minibatch_size=args.batch_size,
                        eval_interval=args.eval_interval,
                        lr=args.lr,
                        reg=args.reg,
                        tolerance=args.tolerance,
                        max_steps=args.max_steps,
                        save_path=args.model)
        elif isinstance(model, HiddenMarkovModel): 
            for option in 'reg', 'lr', 'batch_size', 'eval_interval':
                if getattr(args, option):
                    log.warning(f"Ignoring --{option} since we're training by batch EM, not SGD")      
            model.train(corpus=train_corpus,
                        loss=loss,
                        λ=args.λ,    # type: ignore
                        tolerance=args.tolerance,
                        max_steps=args.max_steps,
                        save_path=args.model)
        else:
            raise NotImplementedError()
        
        if hasattr(model, "total_training_time"):
            eval_log.info(f"Total training time: {datetime.timedelta(seconds=int(model.total_training_time))}\n---")  # type: ignore
    else:
        # Any training-related options are irrelevant, but let's not complain if
        # they were provided: we happen to be training on 0 files this time,
        # that's all!
        pass  
                     
    with torch.inference_mode():   # turn off all gradient tracking
        # Run the model on the input data (eval corpus).
        if args.output_file:
            write_tagging(model, eval_corpus, Path(args.output_file))
            eval_log.info(f"Wrote tagging to {args.output_file}")

        # Show how well we did on the input data.  
        loss(model)         # show the main loss function (using the logger) -- redundant if we trained
        other_loss(model)   # show the other loss function (using the logger)
        eval_log.info("===")
    
if __name__ == "__main__":
    main()
