import json


def verify_json(completion):
    try: 
        prediction = json.loads(prediction)
    except:
        return 0.0
    if not all(k in prediction for k in ("sentences", "label")):
        return 0.0
    if prediction['label'] not in ("positive", "negative"):
        return 0.0
    return 1.0 


def verify_label(completion, answer):
    true_label = json.loads(answer)['label']
    try:
        completion_label = json.loads(completion)['label']
    except:
        return 0.0
    if true_label == completion_label:
        return 1.0
    else:
        return 0.0


def verify_aifs(completion, answer):
    rewards = 0.0
    try:
        sentences = json.loads(completion)['sentences']
    except:
        rewards = -1.0
        return rewards
    true_sentences = json.loads(answer)['sentences']
    normalised_sentences = [x.lower().strip().lstrip('0123456789,;:.)(- ') for x in sentences]
    normalised_gold = [x.lower().strip().lstrip('0123456789,;:.)(- ') for x in true_sentences]
    for sentence in normalised_sentences:
        if sentence in normalised_gold:
            rewards += 1.0
            normalised_gold.remove(sentence)
        else:
            rewards -= 1.0
    rewards -= len(normalised_gold)
    return rewards


# Reward functions
def json_reward(completions, **kwargs):
    # completions = [completion[0]['content'] for completion in completions]
    completions = [completion.strip() for completion in completions]
    return [verify_json(completion) for completion in completions]
    

def label_reward(completions, answer, **kwargs):
    # completions = [completion[0]['content'] for completion in completions]
    completions = [completion.strip() for completion in completions]
    return [verify_label(completion, answer[0]) for completion in completions]

    
def aif_reward(completions, answer, **kwargs):
    # completions = [completion[0]['content'] for completion in completions]
    completions = [completion.strip() for completion in completions]
    return [verify_aifs(completion, answer[0]) for completion in completions]