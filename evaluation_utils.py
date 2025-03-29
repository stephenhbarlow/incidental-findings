from tqdm import tqdm
import pandas as pd

from model_utils import generate_verifier_prediction, generate_generator_prediction


def normalise_sentence(sentence):
    return sentence.lower().strip().lstrip('0123456789,;:.)(- ').replace("\n", " ")


def create_df_from_generations(evaluation_dictionary):
    data = {
        "report_text": evaluation_dictionary['report_text'],
        "predicted_incidentals": evaluation_dictionary['predicted_incidentals'],
    "true_incidentals": evaluation_dictionary['true_incidentals']
    }
    df = pd.DataFrame(data)
    return df


def evaluate_verifier_model(model, tokenizer, dataset, args):

    class_labels = []
    true_labels = []
    report_texts = []

    for doc in tqdm(dataset):
        gold_label = doc['labels']
        true_labels.append(gold_label)
        report_texts.append(doc['report_text'])
        class_label = generate_verifier_prediction(doc, model, tokenizer, args)   
        class_labels.append(class_label)
    return {
            "predicted_labels": class_labels,
            "true_labels": true_labels,
            "report_text": report_texts
        }


def evaluate_generator_model(model, tokenizer, dataset, args):

    tp, fp, fn = 0, 0, 0
    class_labels = []
    inc_sentences = []
    true_labels = []
    true_sentences = []
    report_texts = []
    errors = []
    for doc in tqdm(dataset):
        gold_sents = doc['sentences']
        gold_label = doc['labels']
        true_labels.append(gold_label)
        true_sentences.append(gold_sents)
        report_texts.append(doc['report_text'])
        prediction, error = generate_generator_prediction(doc, model, tokenizer, args)
        errors.append(error)
        print("PREDICTION\n\n")
        print(prediction)
        print("\n\nGOLD\n\n")
        print({"sentences": doc['sentences'], "label": "positive" if doc['labels'] == 1 else "negative"})        
        if args.prompt_template_name == 'CoT':
            class_label = 1 if prediction['label'] == "positive" else 0
        else:
            class_label = 1 if prediction['sentences'] else 0
        sentences = sorted(list(set(prediction['sentences'])))
        class_labels.append(class_label)
        inc_sentences.append(sentences)
        normalised_sentences = [normalise_sentence(x) for x in sentences]
        normalised_gold = [normalise_sentence(x)for x in gold_sents]
        for sentence in normalised_sentences:
            if sentence in normalised_gold:
                tp += 1
                normalised_gold.remove(sentence)
            else:
                fp += 1
        fn += len(normalised_gold)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)
    print(f"Errors: {sum(errors)}")
    return {"precision": precision, 
            "recall": recall, 
            "f1": f1,
            "predicted_labels": class_labels,
            "true_labels": true_labels,
            "predicted_incidentals": inc_sentences,
            "true_incidentals": true_sentences,
            "report_text": report_texts
        }
