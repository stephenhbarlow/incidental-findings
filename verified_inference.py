import argparse
from datasets import load_dataset
from sklearn.metrics import classification_report

from verifier_utils import VerifiedInferencePipeline
from model_utils import init_model_tokenizer_inference
from evaluation_utils import create_df_from_generations
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
    
    # Model settings
    parser.add_argument('--generator_model_name', type=str, default="trained_models/generators/llama-31-8b_generator-model_3_epoch_rank16")
    parser.add_argument('--verifier_model_name', type=str, default="trained_models/verifiers/llama-3.1-8b_verification-model_dedooped_1_epochs-4upsample-1ratio")
    parser.add_argument('--generator_tokenizer', type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument('--verifier_tokenizer', type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument('--max_seq_length', type=int, default=4096)
    parser.add_argument('--quantization', type=bool, default=True)
    parser.add_argument('--unsloth_chat_template', type=str, default="llama")


    # Generation settings
    parser.add_argument('--prompt_template_name', type=str, default="CoT")
    parser.add_argument('--generation_strategy', type=str, default="temperature")
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--top_p', type=float, default=0.5)
    parser.add_argument('--v_temperature', type=float, default=0.1)
    parser.add_argument('--v_top_p', type=float, default=0.1)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--early_stopping', type=bool, default=True)
    parser.add_argument('--do_sample', type=bool, default=True)
    parser.add_argument('--max_time', type=float, default=120.0)

    # other settings
    parser.add_argument('--num_generations', type=int, default=5, help="Number of generations for each document.")
    parser.add_argument('--seed', type=int, default=42,
                        help='Initial random seed.')
    
    args = parser.parse_args()
    return args


def main():

    # parse arguments
    args = parse_args()

    output_dir = args.generator_model_name

    # Set prompt template - this works but is not a good way to do it.
    if args.prompt_template_name == "CoT":
        args.prompt_template = cot_prompt_template
    elif args.prompt_template_name == "CoT-long":
        args.prompt_template = cot_prompt_template_long
    elif args.prompt_template_name == "basic":
        args.prompt_template = basic_prompt_template
    else:
        args.prompt_template= standard_prompt_template

    # Load data into huggingface dataset
    ds = load_dataset("json", 
                  data_files={"train": args.train_data_dir, 
                              "validation": args.val_data_dir,
                              "test": args.test_data_dir,
                              "ext_test": args.ext_test_data_dir
                             }
                 )
    
    generator_model, generator_tokenizer = init_model_tokenizer_inference(args.generator_model_name, args.generator_tokenizer, args)
    verifier_model, verifier_tokenizer = init_model_tokenizer_inference(args.verifier_model_name, args.verifier_tokenizer, args)

    ds = ds.map(args.prompt_template, fn_kwargs={"tokenizer": generator_tokenizer}, load_from_cache_file=False)
    ds.set_format("pt", columns=["prompt"], output_all_columns=True)


    pipeline = VerifiedInferencePipeline(generator_model, 
                                         generator_tokenizer,
                                         verifier_model,
                                         verifier_tokenizer,
                                         ds[args.dataset],
                                         args)
    
    eval_dict = pipeline.evaluate()

    generation_df = create_df_from_generations(eval_dict)
    generation_df.to_csv(f"{output_dir}/verified_inference_generations_on_{args.dataset}-{args.num_generations}gens-temp{args.temperature}_upsample.csv")

    exp_name = f"Verified inference ({args.num_generations})\n\nModel: {args.generator_model_name}\n\nDataset: {args.dataset}"

    incidental_results_string = f"{exp_name}-incidental stats\n\nPrecision: {eval_dict['precision']}\n\nRecall: {eval_dict['recall']}\n\nF1: {eval_dict['f1']}"
    print(incidental_results_string)

    report = classification_report(eval_dict['true_labels'], eval_dict['predicted_labels'], digits=3)
    binary_results_string = f"Experiment: {exp_name}-binary stats\n\n{report}"
    print(binary_results_string)

    with open(f"{output_dir}/verified_inference_results_on_{args.dataset}-{args.num_generations}gens-temp{args.temperature}_upsample.txt", "w") as text_file:
        text_file.write(f"{incidental_results_string}\n\n{binary_results_string}")


if __name__ == '__main__':
    main()
    
