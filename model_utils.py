from unsloth import FastLanguageModel,  is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, GRPOTrainer, GRPOConfig
from transformers import AutoTokenizer, BitsAndBytesConfig, TrainingArguments, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM, LoraConfig, prepare_model_for_kbit_training, get_peft_model
import pandas as pd
import json


def generate_verifier_prediction(sample, model, tokenizer, args):
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    if args.generation_strategy == "beam":

        outputs = model.generate(
            input_ids=sample["prompt"].cuda(), 
            # max_new_tokens=args.max_seq_length,
            eos_token_id=terminators,
            num_beams=args.num_beams,
            early_stopping=args.early_stopping,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=args.do_sample,
        )
    else:
        outputs = model.generate(
            input_ids=sample["prompt"].cuda(),
            # max_new_tokens=4096,
            eos_token_id=terminators,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=args.do_sample,
        )

    prediction = tokenizer.decode(outputs[0][sample["prompt"].shape[-1]:].detach().cpu().numpy(), skip_special_tokens=True)
    prediction = prediction.strip()
    prediction = 1 if "yes" in prediction else 0

    return prediction


def generate_generator_prediction(sample, model, tokenizer, args):
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    generate = True
    count = 0
    temperature = args.temperature
    top_p = args.top_p
    do_sample = args.do_sample

    error = False
    while(generate):
        if args.generation_strategy == "temperature":
            outputs = model.generate(
                input_ids=sample["prompt"].cuda(),
                # max_new_tokens=args.max_seq_length,
                eos_token_id=terminators,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=do_sample,
                max_time=args.max_time,
            )
        else:
            outputs = model.generate(
                input_ids=sample["prompt"].cuda(), 
                # max_new_tokens=args.max_seq_length,
                eos_token_id=terminators,
                num_beams=args.num_beams,
                early_stopping=args.early_stopping,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=do_sample,
                max_time=args.max_time,
            )

        prediction = tokenizer.decode(outputs[0][sample["prompt"].shape[-1]:].detach().cpu().numpy(), skip_special_tokens=True)

        if "basic" not in args.prompt_template_name:
        # don't take any text before "{" and after "}" as this shouldn't be there by definition
            ind_end = prediction.find("}") + 1
            prediction = prediction[0:ind_end]
            ind_start = prediction.find("{")
            prediction = prediction[ind_start:]
            prediction.replace("\n", "")
            count += 1
            if count > 1:
                do_sample = True
                temperature = 0.5
                top_p = 0.5
            if verify_prediction(prediction, args) or count > 10:
                generate = False
            if count > 10:
                print(prediction)
                if (args.prompt_template_name == "CoT" or args.prompt_template_name == "CoT-long"):
                    prediction = {'sentences': [], 'label': 'negative'}
                else:
                    prediction = {'sentences': []}
                    print("***warning failed generation***")
        else:
            try:
                lines = prediction.splitlines()
                lines = [line for line in lines if line.strip()]
                if args.prompt_template_name == "basic":
                    label = "positive" if "positive" in lines[-1] else "negative"
                    prediction = {"sentences": lines[:-1], "label": label}
                else:
                    label = "positive" if lines else "negative"
                    prediction = {"sentences": lines, "label": label}
            except:
                prediction = {'sentences': [], 'label': 'negative'}
                print("***warning failed generation***")
            generate = False
                              
    if isinstance(prediction, str):        
        prediction = json.loads(prediction.strip())
    # print(count)
    if count > 1:
        error = True

    return prediction, error


def init_model_tokenizer_inference(model_name, tokenizer, args):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    tokenizer.pad_token = tokenizer.eos_token

    model, tokenizer = FastLanguageModel.from_pretrained(
                                model_name=model_name, 
                                max_seq_length=args.max_seq_length,
                                dtype=None,
                                load_in_4bit=args.quantization,
                                # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
                            )

    tokenizer = get_chat_template(
                        tokenizer,
                        chat_template = args.unsloth_chat_template, # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
                        mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
                        map_eos_token = True, # Maps <|im_end|> to </s> instead
                    )
    model = FastLanguageModel.for_inference(model)
    
    return model, tokenizer


def init_hf_model_tokenizer_inference(model_name, tokenizer, args):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer)
    if args.quantization:
        quantization_config = BitsAndBytesConfig(
                load_in_4bit = args.quantization,
        )
        model = AutoPeftModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map='auto'
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map='auto'
            )
    # model = model.merge_and_unload()
        
    return model, tokenizer

def _init_trainer(model, tokenizer, dataset, output_dir, args, hf=False):

    if hf:
        fp16 = True
        bf16 = False
    else:
        fp16 = not is_bfloat16_supported()
        bf16 = is_bfloat16_supported()

    training_args = TrainingArguments(
                                output_dir=output_dir,
                                learning_rate=args.learning_rate,
                                per_device_train_batch_size=args.mini_batch_size,
                                per_device_eval_batch_size=args.mini_batch_size,
                                logging_dir=f"{output_dir}/logs",
                                logging_strategy="epoch",
                                num_train_epochs=args.epochs,
                                eval_strategy="epoch",
                                gradient_accumulation_steps=args.accumulation,
                                warmup_steps=args.warmup_steps,
                                save_strategy="epoch",
                                save_total_limit=args.save_total_limit,
                                label_names=["labels"],
                                optim = "adamw_8bit",
                                weight_decay=0.01,
                                lr_scheduler_type = "linear",
                                fp16 = fp16,
                                bf16 = bf16,
                                )
    
    trainer = SFTTrainer(model=model,
                        train_dataset=dataset['train'],
                        eval_dataset=dataset['validation'],
                        dataset_text_field="text",
                        tokenizer=tokenizer,
                        max_seq_length=args.max_seq_length,
                        args=training_args,
                        packing=False
                        )
    return trainer 


def _init_GRPOTrainer(model, tokenizer, dataset, reward_functions, output_dir, args):
    training_args = GRPOConfig(
                        learning_rate=args.learning_rate,
                        adam_beta1=0.9,
                        adam_beta2=0.99,
                        weight_decay=0.1,
                        warmup_ratio=0.1,
                        lr_scheduler_type="cosine",
                        optim="paged_adamw_8bit",
                        logging_steps = 1,
                        per_device_train_batch_size=args.mini_batch_size,
                        gradient_accumulation_steps=args.accumulation, # Increase to 4 for smoother training
                        num_generations=8, # Decrease if out of memory
                        max_prompt_length=args.max_prompt_length,
                        max_completion_length=args.max_seq_length - args.max_prompt_length,
                        num_train_epochs=args.epochs,
                        report_to="none",
                        max_grad_norm=0.1,
                        save_strategy='epoch',
                        save_total_limit=args.save_total_limit,
                        output_dir=output_dir
                    )
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_functions,
        args=training_args,
        train_dataset=dataset['train'],
    )
    return trainer



def init_model_tokenizer_trainer(dataset, prompt_template, output_dir, args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model, tokenizer = FastLanguageModel.from_pretrained(
                                model_name=args.model_name, 
                                max_seq_length=args.max_seq_length,
                                dtype=None,
                                load_in_4bit=args.quantization,
                                # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
                            )

    tokenizer = get_chat_template(
                        tokenizer,
                        chat_template = args.unsloth_chat_template, # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
                        mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
                        map_eos_token = True, # Maps <|im_end|> to </s> instead
                    )
    
    dataset = dataset.map(prompt_template, fn_kwargs={"tokenizer": tokenizer}, load_from_cache_file=False)
    dataset.set_format("pt", columns=["prompt"], output_all_columns=True)
    
    model = FastLanguageModel.get_peft_model(
                        model,
                        r = args.lora_r, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
                        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                        "gate_proj", "up_proj", "down_proj",],
                        lora_alpha = args.lora_alpha,
                        lora_dropout = args.lora_dropout, # Supports any, but = 0 is optimized
                        bias = "none",    # Supports any, but = "none" is optimized
                        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
                        random_state = args.seed,
                        use_rslora = False,  # We support rank stabilized LoRA
                        loftq_config = None, # And LoftQ
                            )
    
    trainer = _init_trainer(model, tokenizer, dataset, output_dir, args)
    
    if args.completions_only:
        trainer = train_on_responses_only(
                    trainer,
                    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
                    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
                )
    
    return model, tokenizer, trainer, dataset


def init_hf_model_tokenizer_trainer(dataset, prompt_template, output_dir, args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
            load_in_4bit = args.quantization,
        )
    model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=quantization_config,
            device_map='auto'
        )
    lora_config = LoraConfig(
            r = args.lora_r, 
            lora_alpha = args.lora_alpha,
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
            lora_dropout = args.lora_dropout, 
            bias = 'none',
            task_type = 'CAUSAL_LM'
        )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    dataset = dataset.map(prompt_template, fn_kwargs={"tokenizer": tokenizer}, load_from_cache_file=False)
    dataset.set_format("pt", columns=["prompt"], output_all_columns=True)

    trainer = _init_trainer(model, tokenizer, dataset, output_dir, args, hf=True)

    return model, tokenizer, trainer, dataset


def init_GRPO_model_tokenizer_trainer(dataset, prompt_template, reward_functions, output_dir, args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model, tokenizer = FastLanguageModel.from_pretrained(
                                                    model_name=args.train_model_name, 
                                                    max_seq_length=args.max_seq_length, 
                                                    load_in_4bit=args.quantization, 
                                                    fast_inference=args.fast_inference,
                                                    max_lora_rank=args.lora_r,
                                                    gpu_memory_utilization=args.gpu_utilisation,
                                                )

    tokenizer = get_chat_template(
                        tokenizer,
                        chat_template = args.unsloth_chat_template, # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
                        mapping = {"role" : "from", "content" : "value", "user" : "human", "assistant" : "gpt"}, # ShareGPT style
                        map_eos_token = True, # Maps <|im_end|> to </s> instead
                    )
    
    dataset = dataset.map(prompt_template, fn_kwargs={"tokenizer": tokenizer}, load_from_cache_file=False)
    dataset.set_format("pt", columns=["prompt"], output_all_columns=True)
    
    if args.new_lora:
        model = FastLanguageModel.get_peft_model(
                            model,
                            r = args.lora_r, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
                            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                            "gate_proj", "up_proj", "down_proj",],
                            lora_alpha = args.lora_alpha,
                            lora_dropout = args.lora_dropout, # Supports any, but = 0 is optimized
                            bias = "none",    # Supports any, but = "none" is optimized
                            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
                            random_state = args.seed,
                            use_rslora = False,  # We support rank stabilized LoRA
                            loftq_config = None, # And LoftQ
                                )
    
    trainer = _init_GRPOTrainer(model, tokenizer, dataset, reward_functions, output_dir, args)
    
    return model, tokenizer, trainer, dataset


def create_df_from_generations(evaluation_dictionary):
    data = {
        "report_text": evaluation_dictionary['report_text'],
        "predicted_incidentals": evaluation_dictionary['predicted_incidentals'],
    "true_incidentals": evaluation_dictionary['true_incidentals']
    }
    df = pd.DataFrame(data)
    return df


def verify_prediction(prediction, args):
    prediction = prediction.strip()
    try: 
        prediction = json.loads(prediction)
    except:
        return False
    if (args.prompt_template_name == "CoT" or args.prompt_template_name == "CoT-long"):
        if not all(k in prediction for k in ("sentences", "label")):
            return False
        if prediction['label'] not in ("positive", "negative"):
            return False
    # else:
    #     if not all(k in prediction for k in ("sentences")):
    #         return False
    return True
