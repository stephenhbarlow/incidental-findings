import argparse
from datasets import load_dataset
from sklearn.metrics import classification_report

from model_utils import init_model_tokenizer_inference, init_hf_model_tokenizer_inference
from evaluation_utils import create_df_from_generations
from prompt_templates import *

def parse_args():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--dataset', type=str, default="validation", help="Choose 'train', 'validation', 'test' or 'ext_test'")
    parser.add_argument('--train_data_dir', type=str, default='data/incidentals_train_sents_sb_marked.json',
                        help='the path to the directory containing the training data.')
    parser.add_argument('--val_data_dir', type=str, default='data/incidentals_val_sents_sb_marked.json',
                        help='the path to the directory containing the validation data.')
    parser.add_argument('--test_data_dir', type=str, default='data/incidentals_test_sents_sb_marked.json',
                        help='the path to the directory containing the validation data.')
    parser.add_argument('--ext_test_data_dir', type=str, default='data/incidentals_rf_sents_sb_marked.json',
                        help='the path to the directory containing the validation data.')
    
    # Model settings
    parser.add_argument('--model_name', type=str, default="trained_models/generators/llama-3.1-8b_generator-model_3_epoch")
    parser.add_argument('--tokenizer', type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument('--max_seq_length', type=int, default=4096)
    parser.add_argument('--quantization', type=bool, default=True)
    parser.add_argument('--backend', type=str, default="hf", help="'hf' or 'unsloth'")

    # Generation settings
    parser.add_argument('--prompt_template_name', type=str, default="CoT")
    parser.add_argument('--generation_strategy', type=str, default="temperature")
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--early_stopping', type=bool, default=True)

    # other settings
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
    
    if args.backend == "unsloth":
        model, tokenizer = model, tokenizer = init_model_tokenizer_inference(args.generator_model_name, args.generator_tokenizer, args)

    else:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit = QUANTIZATION,
        )
        model = AutoPeftModelForCausalLM.from_pretrained(
            INFERENCE_MODEL_NAME,
            quantization_config=quantization_config,
            device_map='auto'
        )

        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
       
    ds = ds.map(args.prompt_template, fn_kwargs={"tokenizer": generator_tokenizer}, load_from_cache_file=False)
    ds.set_format("pt", columns=["prompt"], output_all_columns=True)

    generation_df = create_df_from_generations(eval_dict)
    generation_df.to_csv(f"{output_dir}/generations_on_{args.dataset}.csv")

    exp_name = f"Verified inference ({args.num_generations})\n\nModel: {args.generator_model_name}\n\nDataset: {args.dataset}"

    incidental_results_string = f"{exp_name}-incidental stats\n\nPrecision: {eval_dict['precision']}\n\nRecall: {eval_dict['recall']}\n\nF1: {eval_dict['f1']}"
    print(incidental_results_string)

    report = classification_report(eval_dict['true_labels'], eval_dict['predicted_labels'], digits=3)
    binary_results_string = f"Experiment: {exp_name}-binary stats\n\n{report}"
    print(binary_results_string)

    with open(f"{output_dir}/results_on_validation.txt", "w") as text_file:
        text_file.write(f"{incidental_results_string}\n\n{binary_results_string}")


if __name__ == '__main__':
    main()
    
