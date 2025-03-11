import argparse
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
import random
from verifier_utils import VerifierDataset, sentence_split

def parse_args():
    parser = argparse.ArgumentParser()

    # Experiment settings

    parser.add_argument('--exp_name', type=str, default="dedoop_strat")
    
    # Data input settings
    parser.add_argument('--train_data_dir', type=str, default='data/incidentals_train_sents_sb_marked.json',
                        help='the path to the directory containing the training data.')
    parser.add_argument('--val_data_dir', type=str, default='data/incidentals_val_sents_sb_marked.json',
                        help='the path to the directory containing the validation data.')
    parser.add_argument('--test_data_dir', type=str, default='data/incidentals_test_sents_sb_marked.json',
                        help='the path to the directory containing the validation data.')
    parser.add_argument('--ext_test_data_dir', type=str, default='data/incidentals_rf_sents_sb_marked.json',
                        help='the path to the directory containing the validation data.')
    
    # Data output settings
    parser.add_argument('--output_dir', type=str, default="processed_data")
    parser.add_argument('--stratify_by_report', type=bool, default=True)
    parser.add_argument('--de_duplicate_sentences', type=bool, default=True)
    parser.add_argument('--negative_samples', type=int, default=2426)
    parser.add_argument('--num_neg_examples_per_report', type=int, default=2)
    
    # Other settings
    parser.add_argument('--seed', type=int, default=42,
                        help='Initial random seed.')
    
    
    args = parser.parse_args()
    return args


def main():

    # Parse arguments
    args = parse_args()

    # Set seed
    random.seed(args.seed)

    # Load data into huggingface dataset
    ds = load_dataset("json", 
                  data_files={"train": args.train_data_dir, 
                              "validation": args.val_data_dir,
                              "test": args.test_data_dir,
                              "ext_test": args.ext_test_data_dir
                             }
                 )
    
    train = VerifierDataset(ds['train'].to_pandas(), args)
    val = VerifierDataset(ds['validation'].to_pandas(), args)

    train_df = train.create_verifier_df()
    val_df = val.create_verifier_df()

    train_df.to_csv(f"{args.output_dir}/verifier_train_dataset_{args.exp_name}.csv")
    val_df.to_csv(f"{args.output_dir}/verifier_val_dataset_{args.exp_name}.csv")


if __name__ == '__main__':
    main()
    
