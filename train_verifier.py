from datasets import load_dataset
import random
from unsloth import FastLanguageModel
from sklearn.metrics import classification_report
import argparse

from prompt_templates import reward_prompt_template
from evaluation_utils import evaluate_verifier_model
from model_utils import init_model_tokenizer_trainer


def parse_args():
    parser = argparse.ArgumentParser()

    # Experiment settings
    parser.add_argument('--exp_name', type=str, default="llama-3.1-8b_verification-model_dedooped_4_epochs-1upsample-1ratio")
    parser.add_argument('--evaluate_model', type=bool, default=True)
    parser.add_argument('--quantization', type=bool, default=True)

    # Training settings
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument('--max_seq_length', type=int, default=4096)
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=64)
    parser.add_argument('--lora_dropout', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--mini_batch_size', type=int, default=1)
    parser.add_argument('--accumulation', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--warmup_steps', type=int, default=10)
    parser.add_argument('--save_total_limit', type=int, default=1)
    parser.add_argument('--unsloth_chat_template', type=str, default="llama")
    parser.add_argument('--completions_only', type=bool, default=False)

    # Generation settings
    parser.add_argument('--generation_strategy', type=str, default="temperature")
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--top_p', type=float, default=0.1)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--early_stopping', type=bool, default=True)
    parser.add_argument('--do_sample', type=bool, default=False)
    parser.add_argument('--max_time', type=float, default=120.0)

    # Data input settings
    parser.add_argument('--train_data_dir', type=str, default='processed_data/verifier_train_dataset_dedoop_upsample1_ratio1.csv',
                        help='the path to the directory containing the training data.')
    parser.add_argument('--val_data_dir', type=str, default='processed_data/verifier_val_dataset_dedoop_upsample1_ratio1.csv',
                        help='the path to the directory containing the validation data.')
    
    parser.add_argument('--base_output_dir', type=str, default='trained_models/verifiers/')
    
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

    # Load data into huggingface dataset
    ds = load_dataset("csv", 
                  data_files={"train": args.train_data_dir, 
                              "validation": args.val_data_dir,
                             }
                 )

    # Initialise model, tokenizer, trainer and modified dataset
    model, tokenizer, trainer, ds = init_model_tokenizer_trainer(ds, reward_prompt_template, output_dir, args)
    
    # Train model and save 
    trainer.train()
    model.save_pretrained(output_dir)

    if args.evaluate_model:
        FastLanguageModel.for_inference(model)
        eval_dict = evaluate_verifier_model(model, tokenizer, ds["validation"], args)

        report = classification_report(eval_dict['true_labels'], eval_dict['predicted_labels'], digits=3)
        binary_results_string = f"Experiment: {args.exp_name}-binary stats\n\n{report}"
        print(binary_results_string)

        with open(f"{output_dir}/results.txt", "w") as text_file:
            text_file.write(f"{binary_results_string}")


if __name__ == '__main__':
    main()
    
