from string import punctuation
import json


sentence_format = "For each actionable incidental finding extract only the first sentence referring to it in each section of the report (e.g. 'Findings' and 'Interpretation')."

def standard_prompt_template(row, tokenizer=None):
    format_string = """The output should be a markdown code snippet formatted in the following json schema: {"sentences": list of strings // a list of the actionable incidental findings as strings, or an empty list if there are no actionable incidental findings in the report.}"""
    system_prompt = "The following text is a PET-CT report for lung cancer:"
    sents = row['sentences']
    output_dict = {"sentences": sents}
    output_json = json.dumps(output_dict)
    text = row['text'].strip().lstrip(punctuation).strip()
    row_json = [{"from": "gpt",
                "value": system_prompt,
                },
                {"from": "human",
                 "value": f"""REPORT: {text}\n\nINSTRUCTION: Extract the sentences in the report indicating actionable incidental findings requiring medical intervention.\n\n{format_string}"""
                },
               {"from": "gpt",
                "value": output_json
               }]
    row['report_text'] = row['text']
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    row["prompt"] = tokenizer.apply_chat_template(row_json[:2], add_generation_prompt=True, return_tensors="pt")
    row["prompt text"] = tokenizer.apply_chat_template(row_json[:2], tokenize=False)
    row['completion_text'] = row_json[2]["value"]
    return row


def cot_prompt_template(row, tokenizer=None):
    format_string = """The output should be a markdown code snippet formatted in the following json schema: {"sentences": list of strings // a list of the actionable incidental findings as strings, or an empty list if there are no actionable incidental findings in the report., "label": string // "positive" if there are any actionable incidental findings in the report, or "negative" only.}"""
    system_prompt = "The following text is a PET-CT report for lung cancer:"
    sents = row['sentences']
    label = "positive" if row['labels'] == 1 else "negative"
    output_dict = {"sentences": sents, "label": label}
    output_json = json.dumps(output_dict)
    text = row['text'].strip().lstrip(punctuation).strip()
    row_json = [{"from": "gpt",
                "value": system_prompt,
                },
                {"from": "human",
                 "value": f"""REPORT: {text}\n\nINSTRUCTION: Extract the sentences in the report indicating actionable incidental findings requiring medical intervention, then label the overall report "positive" (if there are any actionable incidental findings in the report), or "negative".\n\n{format_string}"""
                },
               {"from": "gpt",
                "value": output_json
               }]
    row['report_text'] = row['text']
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    row["prompt"] = tokenizer.apply_chat_template(row_json[:2], add_generation_prompt=True, return_tensors="pt")
    row["prompt text"] = tokenizer.apply_chat_template(row_json[:2], tokenize=False)
    row['completion_text'] = row_json[2]["value"]
    return row


def basic_prompt_template(row, tokenizer=None):
    system_prompt = "The following text is a PET-CT report for lung cancer:"
    sents = row['sentences']
    label = "positive" if row['labels'] == 1 else "negative"
    output_string = ""
    for i, sent in enumerate(sents):
        output_string = output_string + f"{i+1}. {sent}\n\n"
    output_string = output_string + f"""Label: "{label}".\n\n"""
    text = row['text'].strip().lstrip(punctuation).strip()
    row_json = [{"from": "gpt",
                 "value": system_prompt,
                },
                {"from": "human",
                 "value": f"""REPORT: {text}\n\nINSTRUCTION: Extract the sentences in the report indicating actionable incidental findings requiring medical intervention, then label the overall report "positive" (if there are any actionable incidental findings in the report), or "negative".\n\n"""
                },
                {"from": "gpt",
                 "value": output_string
                }]
    row['report_text'] = row['text']
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    row["prompt"] = tokenizer.apply_chat_template(row_json[:2], add_generation_prompt=True, return_tensors="pt")
    row["prompt text"] = tokenizer.apply_chat_template(row_json[:2], tokenize=False)
    row['completion_text'] = row_json[2]["value"]
    return row


def basic_standard_prompt_template(row, tokenizer=None):
    system_prompt = "The following text is a PET-CT report for lung cancer:"
    sents = row['sentences']
    label = "positive" if row['labels'] == 1 else "negative"
    output_string = ""
    for i, sent in enumerate(sents):
        output_string = output_string + f"{i+1}. {sent}\n\n"
    # output_string = output_string + f"""Label: "{label}".\n\n"""
    text = row['text'].strip().lstrip(punctuation).strip()
    row_json = [{"from": "gpt",
                 "value": system_prompt,
                },
                {"from": "human",
                 "value": f"""REPORT: {text}\n\nINSTRUCTION: Extract the sentences in the report indicating actionable incidental findings requiring medical intervention.\n\n"""
                },
                {"from": "gpt",
                 "value": output_string
                }]
    row['report_text'] = row['text']
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    row["prompt"] = tokenizer.apply_chat_template(row_json[:2], add_generation_prompt=True, return_tensors="pt")
    row["prompt text"] = tokenizer.apply_chat_template(row_json[:2], tokenize=False)
    row['completion_text'] = row_json[2]["value"]
    return row

    
def cot_prompt_template_long(row, tokenizer=None):
    format_string = """C. Finally, output steps 1 and 2 as a code snippet formatted in the following json schema: {"sentences": list of strings // a list of the actionable incidental findings as strings, or an empty list if there are no actionable incidental findings in the report., "label": string // "positive" if there are any actionable incidental findings in the report, or "negative" only.}"""
    system_prompt = "The following text is a PET-CT report for lung cancer:"
    sents = row["sentences"]
    label = "positive" if row['labels'] == 1 else "negative"
    output_dict = {"sentences": sents, "label": label}
    output_json = json.dumps(output_dict)
    output_string = ""
    if label == "positive":
        for i, sent in enumerate(sents):
            output_string = output_string + f"{i+1}. {sent}\n\n"
        output_string = output_string + f"""Therefore the report is labelled "{label}".\n\n"""
    else:
        output_string = output_string + f"""There are no actionable incidental findings in the report.\n\nTherefore the report is labelled "{label}".\n\n"""
    output_string = output_string + output_json
    text = row['text'].strip().lstrip(punctuation).strip()
    row_json = [{"from": "gpt",
                "value": system_prompt,
                },
                {"from": "human",
                 "value": f"""REPORT: {text}\n\nINSTRUCTIONS:\n\nA. Extract the sentences in the report indicating actionable incidental findings requiring medical intervention.\n\nB. Label the overall report "positive" (if there are any actionable incidental findings in the report), or "negative".\n\n{format_string}"""
                },
               {"from": "gpt",
                "value": output_string
               }]
    row['report_text'] = row['text']
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    row["prompt"] = tokenizer.apply_chat_template(row_json[:2], add_generation_prompt=True, return_tensors="pt")
    row["prompt text"] = tokenizer.apply_chat_template(row_json[:2], tokenize=False)
    row['completion_text'] = row_json[2]["value"]
    return row


def standard_prompt_template_hf(row, tokenizer=None):
    format_string = """The output should be a markdown code snippet formatted in the following json schema: {"sentences": list of strings // a list of the actionable incidental findings as strings, or an empty list if there are no actionable incidental findings in the report.}"""
    system_prompt = "The following text is a PET-CT report for lung cancer:"
    sents = row['sentences']
    output_dict = {"sentences": sents}
    output_json = json.dumps(output_dict)
    text = row['text'].strip().lstrip(punctuation).strip()
    row_json = [{"role": "system",
                "content": system_prompt,
                },
                {"role": "user",
                 "content": f"""REPORT: {text}\n\nINSTRUCTION: Extract the sentences in the report indicating actionable incidental findings requiring medical intervention.\n\n{format_string}"""
                },
               {"role": "assistant",
                "content": output_json
               }]
    row['report_text'] = row['text']
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    row["prompt"] = tokenizer.apply_chat_template(row_json[:2], add_generation_prompt=True, return_tensors="pt")
    row["prompt text"] = tokenizer.apply_chat_template(row_json[:2], tokenize=False)
    row['completion_text'] = row_json[2]["content"]
    return row


def cot_prompt_template_hf(row, tokenizer=None):
    format_string = """The output should be a markdown code snippet formatted in the following json schema: {"sentences": list of strings // a list of the actionable incidental findings as strings, or an empty list if there are no actionable incidental findings in the report., "label": string // "positive" if there are any actionable incidental findings in the report, or "negative" only.}"""
    system_prompt = "The following text is a PET-CT report for lung cancer:"
    sents = row['sentences']
    label = "positive" if row['labels'] == 1 else "negative"
    output_dict = {"sentences": sents, "label": label}
    output_json = json.dumps(output_dict)
    text = row['text'].strip().lstrip(punctuation).strip()
    row_json = [{"role": "system",
                "content": system_prompt,
                },
                {"role": "user",
                 "content": f"""REPORT: {text}\n\nINSTRUCTION: Extract the sentences in the report indicating actionable incidental findings requiring medical intervention, then label the overall report "positive" (if there are any actionable incidental findings in the report), or "negative".\n\n{format_string}"""
                },
               {"role": "assistant",
                "content": output_json
               }]
    row['report_text'] = row['text']
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    row["prompt"] = tokenizer.apply_chat_template(row_json[:2], add_generation_prompt=True, return_tensors="pt")
    row["prompt text"] = tokenizer.apply_chat_template(row_json[:2], tokenize=False)
    row['completion_text'] = row_json[2]["content"]
    return row


def basic_prompt_template_hf(row, tokenizer=None):
    system_prompt = "The following text is a PET-CT report for lung cancer:"
    sents = row['sentences']
    label = "positive" if row['labels'] == 1 else "negative"
    output_string = ""
    for i, sent in enumerate(sents):
        output_string = output_string + f"{i+1}. {sent}\n\n"
    output_string = output_string + f"""Label: "{label}".\n\n"""
    text = row['text'].strip().lstrip(punctuation).strip()
    row_json = [{"role": "system",
                 "content": system_prompt,
                },
                {"role": "user",
                 "content": f"""REPORT: {text}\n\nINSTRUCTION: Extract the sentences in the report indicating actionable incidental findings requiring medical intervention, then label the overall report "positive" (if there are any actionable incidental findings in the report), or "negative".\n\n"""
                },
                {"role": "assistant",
                 "content": output_string
                }]
    row['report_text'] = row['text']
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    row["prompt"] = tokenizer.apply_chat_template(row_json[:2], add_generation_prompt=True, return_tensors="pt")
    row["prompt text"] = tokenizer.apply_chat_template(row_json[:2], tokenize=False)
    row['completion_text'] = row_json[2]["content"]
    return row
    

def cot_prompt_template_long_hf(row, tokenizer=None):
    format_string = """C. Finally, output steps 1 and 2 as a code snippet formatted in the following json schema: {"sentences": list of strings // a list of the actionable incidental findings as strings, or an empty list if there are no actionable incidental findings in the report., "label": string // "positive" if there are any actionable incidental findings in the report, or "negative" only.}"""
    system_prompt = "The following text is a PET-CT report for lung cancer:"
    sents = row["sentences"]
    label = "positive" if row['labels'] == 1 else "negative"
    output_dict = {"sentences": sents, "label": label}
    output_json = json.dumps(output_dict)
    output_string = ""
    if label == "positive":
        for i, sent in enumerate(sents):
            output_string = output_string + f"{i+1}. {sent}\n\n"
        output_string = output_string + f"""Therefore the report is labelled "{label}".\n\n"""
    else:
        output_string = output_string + f"""There are no actionable incidental findings in the report.\n\nTherefore the report is labelled "{label}".\n\n"""
    output_string = output_string + output_json
    text = row['text'].strip().lstrip(punctuation).strip()
    row_json = [{"role": "system",
                "content": system_prompt,
                },
                {"role": "user",
                 "content": f"""REPORT: {text}\n\nINSTRUCTIONS:\n\nA. Extract the sentences in the report indicating actionable incidental findings requiring medical intervention.\n\nB. Label the overall report "positive" (if there are any actionable incidental findings in the report), or "negative".\n\n{format_string}"""
                },
               {"role": "assistant",
                "content": output_string
               }]
    row['report_text'] = row['text']
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    row["prompt"] = tokenizer.apply_chat_template(row_json[:2], add_generation_prompt=True, return_tensors="pt")
    row["prompt text"] = tokenizer.apply_chat_template(row_json[:2], tokenize=False)
    row['completion_text'] = row_json[2]["content"]
    return row
    

def reward_prompt_template(row, tokenizer=None):
    format_string = """Answer 'yes' or 'no' only."""
    system_prompt = "The following text is a PET-CT report for lung cancer:"
    aif = row['aifs']
    label = "yes" if row['labels'] == 1 else "no"
    text = row['text'].strip().lstrip(punctuation).strip()
    row_json = [{"from": "gpt",
                "value": system_prompt,
                },
                {"from": "human",
                 "value": f"""REPORT: {text}\n\nINSTRUCTION: In the context of the report would the following sentence be related to an actionable incidental finding?\n\n"{aif}"\n\n{format_string}"""
                },
               {"from": "gpt",
                "value": label
               }]
    row['report_text'] = row['text']
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    row["prompt"] = tokenizer.apply_chat_template(row_json[:2], add_generation_prompt=True, return_tensors="pt")
    row["prompt text"] = tokenizer.apply_chat_template(row_json[:2], tokenize=False)
    row['completion_text'] = row_json[2]["value"]
    return row


def reward_prompt_template_hf(row, tokenizer=None):
    format_string = """Answer 'yes' or 'no' only."""
    system_prompt = "The following text is a PET-CT report for lung cancer:"
    aif = row['aifs']
    label = "yes" if row['labels'] == 1 else "no"
    text = row['text'].strip().lstrip(punctuation).strip()
    row_json = [{"role": "system",
                "content": system_prompt,
                },
                {"role": "user",
                 "content": f"""REPORT: {text}\n\nINSTRUCTION: In the context of the report would the following sentence be related to an actionable incidental finding?\n\n"{aif}"\n\n{format_string}"""
                },
               {"role": "assistant",
                "content": label
               }]
    row['report_text'] = row['text']
    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    row["prompt"] = tokenizer.apply_chat_template(row_json[:2], add_generation_prompt=True, return_tensors="pt")
    row["prompt text"] = tokenizer.apply_chat_template(row_json[:2], tokenize=False)
    row['completion_text'] = row_json[2]["content"]
    return row




