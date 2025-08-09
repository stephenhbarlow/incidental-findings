import argparse
from datasets import load_dataset
from sklearn.metrics import classification_report
from collections import Counter
import math

from model_utils import init_model_tokenizer_inference, init_hf_model_tokenizer_inference
from evaluation_utils import evaluate_generator_model, evaluate_verifier_model, create_df_from_generations
from prompt_templates import *

def parse_args():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--dataset', type=str, default="ext_test", help="Choose 'train', 'validation', 'test' or 'ext_test'")
    parser.add_argument('--train_data_dir', type=str, default='data/incidentals_train_sents_sb_marked.json',
                        help='the path to the directory containing the training data.')
    parser.add_argument('--val_data_dir', type=str, default='data/incidentals_val_sents_sb_marked.json',
                        help='the path to the directory containing the validation data.')
    parser.add_argument('--test_data_dir', type=str, default='data/incidentals_test_sents_sb_marked.json',
                        help='the path to the directory containing the validation data.')
    parser.add_argument('--ext_test_data_dir', type=str, default='data/incidentals_rf_sents_sb_marked.json',
                        help='the path to the directory containing the validation data.')
    parser.add_argument('--verifier_val_dir', type=str, default='processed_data/verifier_val_dataset_dedoop_strat.csv')
    
    # Model settings
    parser.add_argument('--model_name', type=str, default="trained_models/generators/phi-4-14b_3_epoch_unsloth16-64")
    parser.add_argument('--tokenizer', type=str, default="microsoft/phi-4")
    parser.add_argument('--max_seq_length', type=int, default=4096)
    parser.add_argument('--quantization', type=bool, default=True)
    parser.add_argument('--backend', type=str, default="unsloth", help="'hf' or 'unsloth'")
    parser.add_argument('--unsloth_chat_template', type=str, default="phi-4")


    # Generation settings
    parser.add_argument('--prompt_template_name', type=str, default="CoT")
    parser.add_argument('--generation_strategy', type=str, default="temperature")
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--top_p', type=float, default=0.5)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--early_stopping', type=bool, default=True)
    parser.add_argument('--do_sample', type=bool, default=True)
    parser.add_argument('--max_time', type=float, default=60.0)

    # other settings
    parser.add_argument('--num_votes', type=int, default=11)
    parser.add_argument('--seed', type=int, default=42,
                        help='Initial random seed.')
    
    args = parser.parse_args()
    return args


def majority_vote(*prediction_dicts):
    majority_labels = []
    majority_sentences = []
    num_dicts = len(prediction_dicts)
    majority_threshold = math.floor(num_dicts / 2) + 1  # strict majority
    print(f"Majority Threshold: {majority_threshold}")
    
    # Assume all dicts have the same true labels/incidentals
    true_labels = prediction_dicts[0]['true_labels']
    true_incidentals = prediction_dicts[0]['true_incidentals']
    report_texts = prediction_dicts[0]['report_text']
    
    # ----- Majority voting for labels (strict majority) -----
    for preds in zip(*(d['predicted_labels'] for d in prediction_dicts)):
        counter = Counter(preds)
        label, count = counter.most_common(1)[0]
        if count >= majority_threshold:
            majority_labels.append(label)
        else:
            majority_labels.append(0)  # or None if you want "no decision"
    
    # ----- Majority voting for incidentals (strict majority) -----
    for preds in zip(*(d['predicted_incidentals'] for d in prediction_dicts)):
        # Flatten all normalized incidentals
        normalized_all = [
            inc.lower().strip().lstrip('0123456789,;:.)(- ')
            for p in preds for inc in p
        ]
        
        # Count votes for each incidental
        votes = Counter(normalized_all)
        
        # Keep only those with majority support
        consensus = [
            inc for inc, count in votes.items() if count >= majority_threshold
        ]
        
        # Retrieve original-cased incidentals preserving order
        final_list = []
        seen_norm = set()
        for p in preds:
            for inc in p:
                norm_inc = inc.lower().strip().lstrip('0123456789,;:.)(- ')
                if norm_inc in consensus and norm_inc not in seen_norm:
                    final_list.append(inc)
                    seen_norm.add(norm_inc)
        
        majority_sentences.append(final_list)
    
    # ----- Metrics -----
    tp, fp, fn = 0, 0, 0
    for (p, t) in zip(majority_sentences, true_incidentals):
        np = [x.lower().strip().lstrip('0123456789,;:.)(- ') for x in p]
        nt = [x.lower().strip().lstrip('0123456789,;:.)(- ') for x in t]
        for sentence in np:
            if sentence in nt:
                tp += 1
                nt.remove(sentence)
            else:
                fp += 1
        fn += len(nt)
    
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_labels": majority_labels,
        "true_labels": true_labels,
        "predicted_incidentals": majority_sentences,
        "true_incidentals": true_incidentals,
        "report_text": report_texts
    }


def main():

    # parse arguments
    args = parse_args()

    if args.generation_strategy == "beam" and args.backend == "unsloth":
        print("*** WARNING BEAM SEARCH FOR LLAMA NOT SUPPORTED BY UNSLOTH - SETTING BACKEND TO HUGGINGFACE ***")
        args.backend = "hf"

    output_dir = args.model_name


    # Set prompt template - this works but is not a good way to do it.
    if args.backend == "unsloth":
        if args.prompt_template_name == "CoT":
            args.prompt_template = cot_prompt_template
        elif args.prompt_template_name == "CoT-long":
            args.prompt_template = cot_prompt_template_long
        elif args.prompt_template_name == "basic":
            args.prompt_template = basic_prompt_template
        elif args.prompt_template_name == "basic-standard":
            args.prompt_template = basic_standard_prompt_template
        else:
            args.prompt_template= standard_prompt_template
    else:
        if args.prompt_template_name == "CoT":
            args.prompt_template = cot_prompt_template_hf
        elif args.prompt_template_name == "CoT-long":
            args.prompt_template = cot_prompt_template_long_hf
        elif args.prompt_template_name == "basic":
            args.prompt_template = basic_prompt_template_hf
        else:
            args.prompt_template= standard_prompt_template_hf

    # Load data into huggingface dataset
    ds = load_dataset("json", 
                data_files={"train": args.train_data_dir, 
                            "validation": args.val_data_dir,
                            "test": args.test_data_dir,
                            "ext_test": args.ext_test_data_dir
                            }
                )
    
    if args.backend == "unsloth":
        model, tokenizer = model, tokenizer = init_model_tokenizer_inference(args.model_name, args.tokenizer, args)

    else:
        model, tokenizer = init_hf_model_tokenizer_inference(args.model_name, args.tokenizer, args)

    ds = ds.map(args.prompt_template, fn_kwargs={"tokenizer": tokenizer}, load_from_cache_file=False)
    ds.set_format("pt", columns=["prompt"], output_all_columns=True)

    dict_list = []
    for _ in range(args.num_votes):
        dict_list.append(evaluate_generator_model(model, tokenizer, ds[args.dataset], args))

    eval_dict = majority_vote(*dict_list)

    generation_df = create_df_from_generations(eval_dict)
    generation_df.to_csv(f"{output_dir}/majority_inference_generations_on_{args.dataset}-{args.generation_strategy}-{args.backend}-{args.num_beams}beams-sample{args.do_sample}-num_votes{args.num_votes}.csv")

    exp_name = f"Majority inference ({args.generation_strategy})\n\nModel: {args.model_name}\n\nDataset: {args.dataset}\n\nNumber of Votes: {args.num_votes}"

    incidental_results_string = f"{exp_name}-incidental stats\n\nPrecision: {eval_dict['precision']}\n\nRecall: {eval_dict['recall']}\n\nF1: {eval_dict['f1']}"
    print(incidental_results_string)

    report = classification_report(eval_dict['true_labels'], eval_dict['predicted_labels'], digits=3)
    binary_results_string = f"Experiment: {exp_name}-binary stats\n\n{report}"
    print(binary_results_string)

    with open(f"{output_dir}/majority_inference_results_on_{args.dataset}-{args.generation_strategy}-{args.backend}-{args.num_beams}beams-sample{args.do_sample}-num_votes{args.num_votes}.txt", "w") as text_file:
        text_file.write(f"{incidental_results_string}\n\n{binary_results_string}")


if __name__ == '__main__':
    main()
    
