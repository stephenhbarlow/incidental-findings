from datasets import load_dataset
import random
from unsloth import FastLanguageModel
from sklearn.metrics import classification_report
import argparse
import os

from model_utils import init_model_tokenizer_trainer, init_hf_model_tokenizer_trainer
from prompt_templates import *
from evaluation_utils import evaluate_generator_model, create_df_from_generations


def parse_args():
    parser = argparse.ArgumentParser()

    # Experiment settings
    parser.add_argument('--exp_name', type=str, default="llama-2-7b_generator-model_3_epoch_unsloth")
    parser.add_argument('--evaluate_model', type=bool, default=True)
    parser.add_argument('--prompt_template_name', type=str, default="CoT")
    parser.add_argument('--quantization', type=bool, default=True)

    # Training settings
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument('--max_seq_length', type=int, default=4096)
    parser.add_argument('--lora_r', type=int, default=128)
    parser.add_argument('--lora_alpha', type=int, default=64)
    parser.add_argument('--lora_dropout', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--mini_batch_size', type=int, default=1)
    parser.add_argument('--accumulation', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--warmup_steps', type=int, default=10)
    parser.add_argument('--save_total_limit', type=int, default=4)
    parser.add_argument('--backend', type=str, default="unsloth")

    # Generation settings
    parser.add_argument('--generation_strategy', type=str, default="temperature")
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--top_p', type=float, default=0.2)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--early_stopping', type=bool, default=True)
    parser.add_argument('--do_sample', type=bool, default=True)
    parser.add_argument('--max_time', type=float, default=120.0)

    # Data input settings
    parser.add_argument('--train_data_dir', type=str, default='data/incidentals_train_sents_sb_marked.json',
                        help='the path to the directory containing the training data.')
    parser.add_argument('--val_data_dir', type=str, default='data/incidentals_val_sents_sb_marked.json',
                        help='the path to the directory containing the validation data.')
    parser.add_argument('--test_data_dir', type=str, default='data/incidentals_test_sents_sb_marked.json',
                        help='the path to the directory containing the internal test data.')
    parser.add_argument('--ext_test_data_dir', type=str, default='data/incidentals_rf_sents_sb_marked.json',
                        help='the path to the directory containing the external test data.')
    
    parser.add_argument('--base_output_dir', type=str, default='trained_models/generators/')
    
    # other settings
    parser.add_argument('--seed', type=int, default=42,
                        help='Initial random seed.')
    
    
    args = parser.parse_args()
    return args


def main():

    # parse arguments
    args = parse_args()

    # Set seed
    random.seed(args.seed)

    # Set output directory
    output_dir = f"{args.base_output_dir}/{args.exp_name}"

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
    
    # Initialise model, tokenizer, trainer and modified dataset
    if args.backend == "unsloth":
        model, tokenizer, trainer, ds = init_model_tokenizer_trainer(ds, args.prompt_template, output_dir, args)
    else:
        model, tokenizer, trainer, ds = init_hf_model_tokenizer_trainer(ds, args.prompt_template, output_dir, args)
    
    # Train model and save 
    trainer.train()
    model.save_pretrained(output_dir)

    if args.evaluate_model:
        if args.backend == "unsloth":
            FastLanguageModel.for_inference(model)
        # else:
        #     model = model.merge_and_unload()
        eval_dict = evaluate_generator_model(model, tokenizer, ds["validation"], args)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        generation_df = create_df_from_generations(eval_dict)
        generation_df.to_csv(f"{output_dir}/generations_on_validation.csv")

        incidental_results_string = f"Experiment: {args.exp_name}-incidental stats\n\nPrecision: {eval_dict['precision']}\n\nRecall: {eval_dict['recall']}\n\nF1: {eval_dict['f1']}"
        print(incidental_results_string)

        report = classification_report(eval_dict['true_labels'], eval_dict['predicted_labels'], digits=3)
        binary_results_string = f"Experiment: {args.exp_name}-binary stats\n\n{report}"
        print(binary_results_string)

        with open(f"{output_dir}/results_on_validation.txt", "w") as text_file:
            text_file.write(f"{incidental_results_string}\n\n{binary_results_string}")


if __name__ == '__main__':
    main()
    
