from nltk import sent_tokenize
import pandas as pd
import random
from unsloth import FastLanguageModel
from tqdm import tqdm
from model_utils import generate_generator_prediction
from evaluation_utils import normalise_sentence


def sentence_split(doc):
    sentence_list = sent_tokenize(doc)

    # Function which fixes sentences which are just an integer (e.g "1.") 
    # and attaches them to beginning of next sentence as intended.
    for i, entry in enumerate(sentence_list):
        if len(entry) <= 2:
            try:
                sentence_list[i+1] = sentence_list[i] + " " + sentence_list[i+1]
                sentence_list.pop(i)
    
            except IndexError:
                sentence_list.pop(i)
            
    return sentence_list


class VerifierDataset(object):
     
    def __init__(self, dataframe, args):
          self.dataframe = dataframe
          self.args = args

    def create_sentence_split_df(self, dataframe):
        docs = dataframe['text'].to_list()
        aif_list = dataframe['sentences'].to_list()
        doc_sents = []
        for doc in docs:
            doc_sents.append(sentence_split(doc))
        return pd.DataFrame(zip(docs, doc_sents, aif_list), 
                            columns=['text', 'sentences', 'aifs'])


    def de_duplicate_selection(self, selection, doc_sents, aif_list):
        duplication = True
        while duplication:
            if selection in aif_list:
                selection = random.choice(doc_sents)
            else:
                duplication = False
        return selection


    def create_positive_examples_df(self, dataframe, args):
        docs = dataframe['text'].to_list()
        aif_lists = dataframe['aifs'].to_list()
        doc_sents = dataframe['sentences'].to_list()

        expanded_pos_docs = []
        aifs = []
        expanded_sents = []
        labels = []

        for doc, aif_list, doc_sents in zip(docs, aif_lists, doc_sents):
            for aif in aif_list:
                expanded_pos_docs.append(doc)
                aifs.append(aif)
                expanded_sents.append(doc_sents)
                labels.append(1)
                try:
                    doc_sents.remove(aif)
                except:
                    for sent in doc_sents:
                        if aif in sent:
                            doc_sents.remove(sent)
            expanded_pos_docs.append(doc)
            selection = random.choice(doc_sents)
            if args.de_duplicate_sentences:
                    selection = self.de_duplicate_selection(selection, doc_sents, aifs)
            aifs.append(selection)
            expanded_sents.append(doc_sents)
            labels.append(0)
            
        return pd.DataFrame(zip(expanded_pos_docs, expanded_sents, aifs, labels), columns=['text', 'sentences', 'aifs', 'labels'])


    def create_negative_examples_df(self, dataframe, args):
        docs = dataframe['text'].to_list()
        aif_list = dataframe['aifs'].to_list()
        doc_sents = dataframe['sentences'].to_list()

        expanded_neg_docs = []
        aifs = []
        expanded_sents = []
        labels = []

        if args.stratify_by_report:
            for doc, aif_list, doc_sents in zip(docs, aif_list, doc_sents):
                for i in range(args.num_neg_examples_per_report):
                    expanded_neg_docs.append(doc)
                    selection = random.choice(doc_sents)
                    if args.de_duplicate_sentences:
                        selection = self.de_duplicate_selection(selection, doc_sents, aifs)
                    aifs.append(selection)
                    expanded_sents.append(doc_sents)
                    labels.append(0)
                    doc_sents.remove(selection)
            
            dataframe = pd.DataFrame(zip(expanded_neg_docs, expanded_sents, aifs, labels), 
                                    columns=['text', 'sentences', 'aifs', 'labels'])


        else:
            print("*WARNING* Stratify is set to false so not de-duplication will happen for negative samples. *WARNING*")
            for doc, aif_list, doc_sents in zip(docs, aif_list, doc_sents):
                for sent in doc_sents:
                    expanded_neg_docs.append(doc)
                    aifs.append(sent)
                    expanded_sents.append(doc_sents)
                    labels.append(0)
            
            dataframe = pd.DataFrame(zip(expanded_neg_docs, expanded_sents, aifs, labels), 
                                    columns=['text', 'sentences', 'aifs', 'labels'])
            dataframe = dataframe.sample(frac=args.negative_samples, random_state=args.seed)
                
        return dataframe


    def create_verifier_df(self):
        pos_df = self.dataframe[self.dataframe['labels'] == 1]
        neg_df = self.dataframe[self.dataframe['labels'] == 0]

        pos_sent_df = self.create_sentence_split_df(pos_df)
        neg_sent_df = self.create_sentence_split_df(neg_df)

        train_df_pos = self.create_positive_examples_df(pos_sent_df, self.args)
        train_df_neg = self.create_negative_examples_df(neg_sent_df, self.args)

        final_reward_df = pd.concat([train_df_pos, train_df_neg]).sample(frac=1, random_state=self.args.seed)

        return final_reward_df
     

class VerifiedInferencePipeline(object):

    def __init__(self, g_model, g_tokenizer, v_model, v_tokenizer, dataset, args):
          self.g_model = FastLanguageModel.for_inference(g_model)
          self.g_tokenizer = g_tokenizer
          self.v_model = FastLanguageModel.for_inference(v_model)
          self.v_tokenizer = v_tokenizer
          self.dataset = dataset
          self.args = args

    def verify_sentence(self, sentence, report):
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>The following text is a PET-CT report for lung cancer:<|eot_id|><|start_header_id|>user<|end_header_id|>REPORT: {report}\n\nINSTRUCTION: In the context of the report would the following sentence be related to an actionable incidental finding?\n\n"{sentence}"\n\nAnswer 'yes' or 'no' only.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        terminators = [self.v_tokenizer.eos_token_id, self.v_tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        tokenized_prompt = self.v_tokenizer(prompt, return_tensors='pt')
        input_ids = tokenized_prompt['input_ids'].cuda()
        outputs = self.v_model.generate(
                    input_ids=input_ids,
                    max_new_tokens=10,
                    eos_token_id=terminators,
                    do_sample=False,
                    # temperature=self.args.v_temperature,
                    # top_p=self.args.v_top_p,
                    pad_token_id=self.v_tokenizer.eos_token_id,
                )
        prediction = self.v_tokenizer.decode(outputs[0][input_ids.shape[-1]:].detach().cpu().numpy(), skip_special_tokens=True)

        prediction = prediction.strip()[:5]
        verification = True if "yes" in prediction else False 
        return verification


    def select_candidates(self, candidates, report):
        verified_sents = []
        for sent in candidates:
            if self.verify_sentence(sent, report):
                verified_sents.append(sent)
        return verified_sents


    def create_list_of_aifs(self, candidate_lists, report_list):
        list_of_aifs = []
        for candidates, report in tqdm(zip(candidate_lists, report_list)):
            list_of_aifs.append(self.select_candidates(candidates, report))
        return list_of_aifs
            

    # Generate NUM_GENERATIONS iterations of sentences, deduplicate and return
    def generate_candidate_batch(self, doc):
        candidates = []
        refined_candidates = []
        for n in range(self.args.num_generations):
            prediction, _ = generate_generator_prediction(doc, self.g_model, self.g_tokenizer, self.args)
            sents = prediction['sentences']
            candidates.extend(sents)
        candidates = list(set(candidates))
        normalised_candidates = list(set([normalise_sentence(x) for x in candidates]))
        for candidate in candidates:
            if normalise_sentence(candidate) in normalised_candidates:
                refined_candidates.append(candidate)
                normalised_candidates.remove(normalise_sentence(candidate))
        return refined_candidates


    # get list of unique candidate AIFs for each document in a dataset
    def generate_candidates(self):
        candidate_lists = []
        report_list = []
        for doc in tqdm(self.dataset):
            candidate_lists.append(self.generate_candidate_batch(doc))
            report_list.append(doc['report_text'])
        return candidate_lists, report_list


    def generate_and_verify(self):
        candidate_lists, report_list = self.generate_candidates()
        aif_list = self.create_list_of_aifs(candidate_lists, report_list)
        print(aif_list)
        label_list = []
        for aif in aif_list:
            if aif:
                label_list.append(1)
            else:
                label_list.append(0)
        return aif_list, report_list, label_list


    def evaluate(self):
        tp, fp, fn = 0, 0, 0
        aif_list, report_list, label_list = self.generate_and_verify()
        true_sentences = self.dataset['sentences']
        for aif_sents, true_sents in tqdm(zip(aif_list, true_sentences)):
            normalised_sentences = [normalise_sentence(x)for x in aif_sents]
            normalised_gold = [normalise_sentence(x)for x in true_sents]
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
        return {"precision": precision, 
                "recall": recall, 
                "f1": f1,
                "predicted_labels": label_list,
                "predicted_incidentals": aif_list,
                "true_labels": self.dataset['labels'],
                "true_incidentals": true_sentences,
                "report_text": report_list
            }
            
