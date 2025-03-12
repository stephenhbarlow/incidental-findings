import argparse
from datasets import load_dataset
from sklearn.metrics import classification_report

from model_utils import init_model_tokenizer_inference, init_hf_model_tokenizer_inference
from evaluation_utils import evaluate_generator_model, evaluate_verifier_model, create_df_from_generations
from prompt_templates import *

def parse_args():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--dataset', type=str, default="test", help="Choose 'train', 'validation', 'test' or 'ext_test'")
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
    parser.add_argument('--model_name', type=str, default="trained_models/generators/llama-3.1-8b_generator-model_3_epoch")
    parser.add_argument('--model_type', type=str, default="generator", help="'generator or 'verifier'")
    parser.add_argument('--tokenizer', type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument('--max_seq_length', type=int, default=4096)
    parser.add_argument('--quantization', type=bool, default=True)
    parser.add_argument('--backend', type=str, default="hf", help="'hf' or 'unsloth'")

    # Generation settings
    parser.add_argument('--prompt_template_name', type=str, default="CoT")
    parser.add_argument('--generation_strategy', type=str, default="beam")
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--early_stopping', type=bool, default=False)

    # other settings
    parser.add_argument('--seed', type=int, default=42,
                        help='Initial random seed.')
    
    args = parser.parse_args()
    return args


def main():

    # parse arguments
    args = parse_args()

    if args.generation_strategy == "beam" and args.backend == "unsloth":
        print("*** WARNING BEAM SEARCH FOR LLAMA NOT SUPPORTED BY UNSLOTH - SETTING BACKEND TO HUGGINGFACE ***")
        args.backend = "hf"

    output_dir = args.model_name

    if args.model_type == "generator":

        # Set prompt template - this works but is not a good way to do it.
        if args.backend == "unsloth":
            if args.prompt_template_name == "CoT":
                args.prompt_template = cot_prompt_template
            elif args.prompt_template_name == "CoT-long":
                args.prompt_template = cot_prompt_template_long
            elif args.prompt_template_name == "basic":
                args.prompt_template = basic_prompt_template
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

        eval_dict = evaluate_generator_model(model, tokenizer, ds["validation"], args)

        generation_df = create_df_from_generations(eval_dict)
        generation_df.to_csv(f"{output_dir}/standard_inference_generations_on_{args.dataset}-{args.generation_strategy}-{args.backend}.csv")

        exp_name = f"Standard inference ({args.generation_strategy})\n\nModel: {args.model_name}\n\nDataset: {args.dataset}"

        incidental_results_string = f"{exp_name}-incidental stats\n\nPrecision: {eval_dict['precision']}\n\nRecall: {eval_dict['recall']}\n\nF1: {eval_dict['f1']}"
        print(incidental_results_string)

        report = classification_report(eval_dict['true_labels'], eval_dict['predicted_labels'], digits=3)
        binary_results_string = f"Experiment: {exp_name}-binary stats\n\n{report}"
        print(binary_results_string)

        with open(f"{output_dir}/standard_inference_results_on_{args.dataset}-{args.generation_strategy}.txt", "w") as text_file:
            text_file.write(f"{incidental_results_string}\n\n{binary_results_string}")

    else:
        # Load data into huggingface dataset
        ds = load_dataset("json", 
                    data_files={
                                "validation": args.val_data_dir,
                                }
                    )

        if args.backend == "unsloth":
            model, tokenizer = model, tokenizer = init_model_tokenizer_inference(args.generator_model_name, args.generator_tokenizer, args)

        else:
            model, tokenizer = init_hf_model_tokenizer_inference(args.model_name, args.tokenizer, args)

        ds = ds.map(reward_prompt_template, fn_kwargs={"tokenizer": tokenizer}, load_from_cache_file=False)
        ds.set_format("pt", columns=["prompt"], output_all_columns=True)

        eval_dict = evaluate_verifier_model(model, tokenizer, ds["validation"], args)

        report = classification_report(eval_dict['true_labels'], eval_dict['predicted_labels'], digits=3)
        binary_results_string = f"Experiment: {args.exp_name}-binary stats\n\n{report}"
        print(binary_results_string)

        with open(f"{output_dir}/verifier_results-{args.generation_strategy}-{args.backend}.txt", "w") as text_file:
            text_file.write(f"{binary_results_string}")



if __name__ == '__main__':
    main()
    
