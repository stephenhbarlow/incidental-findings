import argparse
from datasets import load_dataset
import pandas as pd

from verifier_utils import VerifiedInferencePipeline
from model_utils import init_model_tokenizer_inference
from prompt_templates import *

def parse_args():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--dataset', type=str, default="test", help="Choose 'train', 'validation', 'test' or 'ext_test'")
    parser.add_argument('--test_data_dir', type=str, default='chunk_no_6.csv',
                        help='the path to the directory containing the data.')
    parser.add_argument('--output_dir', type=str, default="data/gstt_lung_chunks/")

    
    # Model settings
    parser.add_argument('--generator_model_name', type=str, default="trained_models/generators/llama-31-8b_generator-model_3_epoch_rank16_MAIN_MODEL")
    parser.add_argument('--verifier_model_name', type=str, default="trained_models/verifiers/llama-3.1-8b_verification-model_dedooped_1_epochs-4upsample15ratio_gaussian_sigmafactor2_seed_42")
    parser.add_argument('--generator_tokenizer', type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument('--verifier_tokenizer', type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument('--max_seq_length', type=int, default=4096)
    parser.add_argument('--quantization', type=bool, default=True)
    parser.add_argument('--unsloth_chat_template', type=str, default="llama-3")


    # Generation settings
    parser.add_argument('--prompt_template_name', type=str, default="CoT")
    parser.add_argument('--generation_strategy', type=str, default="temperature")
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--top_p', type=float, default=0.5)
    parser.add_argument('--v_temperature', type=float, default=0.1)
    parser.add_argument('--v_top_p', type=float, default=0.1)
    parser.add_argument('--num_beams', type=int, default=1)
    parser.add_argument('--early_stopping', type=bool, default=True)
    parser.add_argument('--do_sample', type=bool, default=True)
    parser.add_argument('--max_time', type=float, default=120.0)

    # other settings
    parser.add_argument('--num_generations', type=int, default=10, help="Number of generations for each document.")
    parser.add_argument('--seed', type=int, default=42,
                        help='Initial random seed.')
    
    args = parser.parse_args()
    return args


def main():

    # parse arguments
    args = parse_args()

    output_dir = args.generator_model_name

    args.prompt_template= cot_prompt_template_batch_inference

    # Load data into huggingface dataset
    ds = load_dataset("csv", 
                  data_files={
                              "test": f"{args.output_dir}{args.test_data_dir}",
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
    
    aif_list, report_list, label_list = pipeline.generate_and_verify()

    data = {
        "id": ds[args.dataset]['Accession'],
        "report_text": report_list,
        "predicted_incidentals": aif_list,
        "predicted_label": label_list
    }

    df = pd.DataFrame(data)

    test_path = "data/verified_inference_generations_on_chunk_no_6.csv"

    # df.to_csv(f"{output_dir}verified_inference_generations_on_{args.test_data_dir}")

    df.to_csv(test_path)


if __name__ == '__main__':
    main()
    
