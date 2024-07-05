import json
import nltk
import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import tqdm

nltk.download('punkt')

model_map = {
    "snli-base": {"model_card": "boychaboy/SNLI_roberta-base", "entailment_idx": 0, "contradiction_idx": 2},
    "snli-large": {"model_card": "boychaboy/SNLI_roberta-large", "entailment_idx": 0, "contradiction_idx": 2},
    "mnli-base": {"model_card": "ckpt/deberta-base-mnli", "entailment_idx": 2, "contradiction_idx": 0},
    "mnli": {"model_card": "ckpt/roberta-large-mnli", "entailment_idx": 2, "contradiction_idx": 0},
    "anli": {"model_card": "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli", "entailment_idx": 0,
             "contradiction_idx": 2},
    "vitc-base": {"model_card": "tals/albert-base-vitaminc-mnli", "entailment_idx": 0, "contradiction_idx": 1},
    "vitc": {"model_card": "ckpt/albert-xlarge-vitaminc-mnli", "entailment_idx": 0, "contradiction_idx": 1},
    "vitc-only": {"model_card": "tals/albert-xlarge-vitaminc", "entailment_idx": 0, "contradiction_idx": 1},
    # "decomp": 0,
}


def batcher(iterator, batch_size=4, progress=False):
    if progress:
        iterator = tqdm.tqdm(iterator)

    batch = []
    for elem in iterator:
        batch.append(elem)
        if len(batch) == batch_size:
            final_batch = batch
            batch = []
            yield final_batch
    if len(batch) > 0:  # Leftovers
        yield batch


def card_to_name(card):
    return card


def name_to_card(name):
    if name in model_map:
        return model_map[name]["model_card"]
    return name


def get_neutral_idx(ent_idx, con_idx):
    return list(set([0, 1, 2]) - set([ent_idx, con_idx]))[0]


class SummaCImager:
    def __init__(self, model_name_nli="mnli", model_name_sent="all-mpnet-base-v2", device="cuda"):
        self.tokenizer_nli = None
        self.tokenizer_sem = None
        assert model_name_nli in model_map.keys(), "Unrecognized model_nli name: `%s`" % (model_name_nli)

        self.model_name_nli = model_name_nli
        self.model_name_sent = model_name_sent
        if model_name_nli != "decomp":
            self.model_card = name_to_card(model_name_nli)
            self.entailment_idx = model_map[model_name_nli]["entailment_idx"]
            self.contradiction_idx = model_map[model_name_nli]["contradiction_idx"]
            self.neutral_idx = get_neutral_idx(self.entailment_idx, self.contradiction_idx)

        self.max_input_length = 500
        self.device = device
        self.cache = {}
        self.model_nli = None  # Lazy loader
        self.model_sem = None  # Lazy loader

    def load_nli(self):
        self.tokenizer_nli = AutoTokenizer.from_pretrained(self.model_card)
        self.model_nli = AutoModelForSequenceClassification.from_pretrained(self.model_card).eval()
        self.model_nli.to(self.device)
        if self.device == "cuda":
            self.model_nli.half()

    def load_sem(self):
        self.tokenizer_sem = AutoTokenizer.from_pretrained(self.model_name_sent)
        self.model_sem = SentenceTransformer(self.model_name_sent, device=self.device)

    def split_sentences(self, text):
        sentences = nltk.tokenize.sent_tokenize(text)
        sentences = [sent for sent in sentences if len(sent) > 10]
        return sentences

    def build_chunk_dataset(self, original, generated, pair_idx=None):
        original_chunks = self.split_sentences(original)
        generated_chunks = self.split_sentences(generated)

        N_ori, N_gen = len(original_chunks), len(generated_chunks)
        dataset = [{"premise": original_chunks[i], "hypothesis": generated_chunks[j], "doc_i": i, "gen_i": j,
                    "pair_idx": pair_idx} for i in range(N_ori) for j in range(N_gen)]
        return dataset, N_ori, N_gen, original_chunks, generated_chunks

    def build_image(self, original, generated):
        dataset, N_ori, N_gen, doc_chunks, sum_chunks = self.build_chunk_dataset(original, generated)

        if len(dataset) == 0:
            return np.zeros((1, 1))

        entail = np.zeros((N_ori, N_gen))
        const = np.zeros((N_ori, N_gen))

        if self.model_nli is None:
            self.load_nli()
        if self.model_sem is None:
            self.load_sem()

        # get doc embeddings
        doc_emb = self.model_sem.encode(doc_chunks, convert_to_tensor=True).to(self.device)
        sum_emb = self.model_sem.encode(sum_chunks, convert_to_tensor=True).to(self.device)
        cos_similarities = torch.zeros((doc_emb.shape[0], sum_emb.shape[0]), device=self.device)
        for i in range(cos_similarities.shape[0]):
            for j in range(cos_similarities.shape[1]):
                cos_similarities[i, j] = torch.cosine_similarity(doc_emb[i].unsqueeze(0), sum_emb[j].unsqueeze(0)).squeeze()
        cos_similarities = cos_similarities.cpu().numpy()
        for batch in batcher(dataset, batch_size=20):
            batch_prems = [b["premise"] for b in batch]
            batch_hypos = [b["hypothesis"] for b in batch]
            batch_tokens_nli = self.tokenizer_nli.batch_encode_plus(list(zip(batch_prems, batch_hypos)), padding=True,
                                                                    truncation=True, max_length=self.max_input_length,
                                                                    return_tensors="pt",
                                                                    truncation_strategy="only_first")
            # TODO: Add semantic
            with torch.no_grad():
                model_outputs = self.model_nli(**{k: v.to(self.device) for k, v in batch_tokens_nli.items()})
            batch_probs = torch.nn.functional.softmax(model_outputs["logits"], dim=-1)
            batch_evids = batch_probs[:, self.entailment_idx].tolist()
            batch_conts = batch_probs[:, self.contradiction_idx].tolist()

            for b, evid, cont in zip(batch, batch_evids, batch_conts):
                entail[b["doc_i"], b["gen_i"]] = evid
                const[b["doc_i"], b["gen_i"]] = cont
        return entail, const, doc_chunks, sum_chunks, cos_similarities
