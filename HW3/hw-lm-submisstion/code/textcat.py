#!/usr/bin/env python3
"""
Computes the total log probability of the sequences of tokens in each file,
according to a given smoothed trigram model.  
"""
import argparse
import logging
import math
from pathlib import Path
import torch

from probs import Wordtype, LanguageModel, num_tokens, read_trigrams

log = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_1",
        type=Path,
        help="path to trained model 1",
    )
    parser.add_argument(
        "model_2",
        type=Path,
        help="path to trained model 2"
    )
    parser.add_argument(
        "prior",
        type=float,
        help='the prior probability for model 1'
    )
    parser.add_argument(
        "test_files",
        type=Path,
        nargs="+",
        help="one or more text files to classify"
    )


    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=['cpu','cuda','mps'],
        help="device to use for PyTorch (cpu or cuda, or mps if you are on a mac)"
    )
    # for verbosity of logging
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet",   dest="logging_level", action="store_const", const=logging.WARNING
    )

    return parser.parse_args()


def file_log_prob(file: Path, lm: LanguageModel) -> float:
    """The file contains one sentence per line. Return the total
    log-probability of all these sentences, under the given language model.
    (This is a natural log, as for all our internal computations.)
    """
    log_prob = 0.0
    x: Wordtype; y: Wordtype; z: Wordtype    # type annotation for loop variables below
    for (x, y, z) in read_trigrams(file, lm.vocab):
        log_prob += lm.log_prob(x, y, z)  # log p(z | xy)
        if log_prob == -math.inf: break 
    
    return log_prob

def main():
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    # Specify hardware device where all tensors should be computed and stored.  This will give errors unless you have such a device (e.g., 'gpu' will work in a Kaggle Notebook where you have turned on GPU acceleration).
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
        

    lm1 = LanguageModel.load(args.model_1, device=args.device)
    lm2 = LanguageModel.load(args.model_2, device=args.device)

    log_prior_1 = math.log(args.prior)
    log_prior_2 = math.log(1 - args.prior)
    
    model_1_counts = 0
    model_2_counts = 0

    for file in args.test_files:
        score_1 = file_log_prob(file, lm1) + log_prior_1
        score_2 = file_log_prob(file, lm2) + log_prior_2
        if score_1 > score_2:
            model_1_counts += 1
            print(f"{args.model_1.name} {file.name}")
        else:
            model_2_counts += 1
            print(f"{args.model_2.name} {file.name}")
    
    total_files = len(args.test_files)
    print(f"{model_1_counts} files were more probably from {args.model_1.name} ({100*model_1_counts/total_files:.2f}%)")
    print(f"{model_2_counts} files were more probably from {args.model_2.name} ({100*model_2_counts/total_files:.2f}%)")


if __name__ == "__main__":
    main()

