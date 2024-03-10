from typing import Any, List, Union
from tqdm import tqdm

from typing import List, Union
from itertools import groupby

import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification
)

import spacy
import json

def sentencize(text):
    """Split text into sentences"""
    global nlp
    if not nlp:
        nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [sent for sent in doc.sents]

def split_text(text, segment_len=200):
    """Split text into segments according to sentence boundaries."""
    segments, seg = [], []
    sents = [[token.text for token in sent] for sent in sentencize(text)]
    for sent in sents:
        if len(seg) + len(sent) > segment_len:
            segments.append(" ".join(seg))
            seg = sent
            # single sentence longer than segment_len
            if len(seg) > segment_len:
                # split into chunks of segment_len
                seg = [
                    " ".join(seg[i:i+segment_len])
                    for i in range(0, len(seg), segment_len)
                ]
                segments.extend(seg)
                seg = []
        else:
            seg.extend(sent)
    if seg:
        segments.append(" ".join(seg))
    return segments

LABELS = ["Entailment", "Neutral", "Contradiction"]

def merge_ret(ret):
    """Merge results from multiple paragraphs"""
    if "Entailment" in ret:
        return "Entailment"
    if "Contradiction" in ret:
        return "Contradiction"
    return "Neutral"


def merge_multi_psg_ret(ret):
    """Merge results from multiple passages
    TODO: consider possible cases where the results are inconsistent.
    """
    if "Entailment" in ret:
        return "Entailment"
    if "Contradiction" in ret:
        return "Contradiction"
    return "Neutral"

class CheckerBase:
    def __init__(self) -> None:
        """
        Initializer for the CheckerBase class.

        Initialize labels for 'Entailment', 'Neutral', and 'Contradiction'.
        Also initializes a list of all labels.
        """

        self.label_entailment = 'Entailment'
        self.label_neutral = 'Neutral'
        self.label_contradiction = 'Contradiction'
        self.labels = ["Entailment", "Neutral", "Contradiction"]

    def check(
        self, 
        claim: List[List[Union[str, List[str]]]],
        reference: Union[List[str], List[List[str]]],
        response: List[str] = None,
        question: List[str] = None,
        max_reference_segment_length: int = 200, 
    ):
        """
        Check claims against references.

        Parameters
        ----------
        claim : List[List[Union[str, List[str]]]]
            List consists of the triplets extracted from each given example.
        reference : Union[List[str], List[List[str]]]
            List of reference passages for each given example.
        response : List[str], optional
            List of model response texts, defaults to None.
        question : List[str], optional
            List of questions for each given example, defaults to None.
        max_reference_segment_length : int, optional
            Maximum length of each reference segment, defaults to 200.

        Returns
        -------
        ret_group_triplet : List[List[str]]
            Grouped triplet checking results corresponding to each given example.

        """

        if response is None:
            response = [None] * len(claim)
        if question is None:
            question = [None] * len(claim)
        input_flattened = []
        input_ids = []
        for idx, (c, ref, res, q) in enumerate(zip(claim, reference, response, question)):
            if isinstance(ref, str):
                ref = [ref]
            segments_all_psg = []
            for psg in ref:
                if max_reference_segment_length > 0:
                    segments = split_text(psg, max_reference_segment_length)
                else:
                    segments = [psg]
                segments_all_psg.append(segments)
            for c_idx, t in enumerate(c):
                for idx_psg, seg_psg in enumerate(segments_all_psg):
                    for seg in seg_psg:
                        input_flattened.append([t, seg, res, q])
                        input_ids.append([idx, c_idx, idx_psg])
        ret = self._check(
                claims=[inp[0] for inp in input_flattened],
                references=[inp[1] for inp in input_flattened],
                responses=[inp[2] for inp in input_flattened],
                questions=[inp[3] for inp in input_flattened],
            )

        ret = [[x] + y for x, y in zip(ret, input_ids)]
        ret_merge_seg = [[merge_ret([item[0] for item in group])] + key[:-1] for key, group in groupby(ret, key=lambda x: x[1:])]
        ret_merge_psg = [[merge_multi_psg_ret([item[0] for item in group])] + key[:-1] for key, group in groupby(ret_merge_seg, key=lambda x: x[1:])]
        ret_group_triplet = [[item[0] for item in group] for key, group in groupby(ret_merge_psg, key=lambda x: x[1:])]

        return ret_group_triplet

    def _check(
        self,
        claims: List[Union[str, List[str]]],
        references: List[str],
        responses: List[str],
        questions: List[str]
    ):
        """
        Internal method for checking claims against references.

        This method should be implemented by subclasses.

        Parameters
        ----------
        claims : List[Union[str, List[str]]]
            List of claims to be checked.
        references : List[str]
            List of reference passages.
        responses : List[str]
            List of model response texts.
        questions : List[str]
            List of questions.

        Returns
        -------
        List[str]
            List of checking results.
        """

        raise NotImplementedError



class NLIChecker(CheckerBase):
    def __init__(
        self, 
        model='ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli',
        device=0,   # Todo: support distributed inference
        batch_size=1
    ):
        """
        Initializes the NLIChecker with the specified model, device, and batch size.

        Parameters
        ----------
        model : str, optional
            The name or identifier of the model to use, defaults to 'ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli'.
        device : int, optional
            The device to run inference on, defaults to 0.
        batch_size : int, optional
            The batch size for inference, defaults to 16.
        """

        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model).to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.device = device
        self.batch_size = batch_size

    @torch.no_grad()
    def _check(
        self,
        claims: List[Union[str, List[str]]],
        references: List[str],
        responses: List[str],
        questions: List[str],
    ):
        """
        Batch checking claims against references.

        Parameters
        ----------
        claims : List[Union[str, List[str]]]
            List of claim triplets.
        references : List[str]
            List of reference passages (split according to 'max_reference_segment_length').
        responses : List[str]
            List of model response texts.
        questions : List[str]
            List of questions corresponding to each triplet.

        Returns
        -------
        ret : List[str]
            List of labels for the checking results.

        """

        N1, N2 = len(references), len(claims)
        assert N1 == N2, f"Batches must be of the same length. {N1} != {N2}"
        if isinstance(claims[0], list):
            assert len(claims[0]) == 3
            claims = [f"{c[0]} {c[1]} {c[2]}" for c in claims]
        batch_preds = []
        for i in tqdm(range(0, len(claims), self.batch_size)):
            batch_claims = claims[i:i + self.batch_size]
            batch_references = references[i:i + self.batch_size]

            inputs = self.tokenizer(
                batch_references, batch_claims, max_length=512, truncation=True,
                return_tensors="pt", padding=True, return_token_type_ids=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            output = self.model(**inputs).logits.softmax(dim=-1).cpu()  # [batch_size, 3]
            preds = output.argmax(dim=-1)
            batch_preds.extend(preds)
        ret = [LABELS[p] for p in batch_preds]

        print("Length of ret", len(ret))
        return ret

def check(args):
    # initialize models
    if args.checker_name in ["gpt4", "claude2"]:
        checker = LLMChecker(model=args.checker_name, batch_size=args.batch_size_checker)
    elif args.checker_name == "nli":
        checker = NLIChecker(batch_size=1)
    elif args.checker_name == "alignscore":
        checker = AlignScoreChecker(batch_size=args.batch_size_checker)
    elif args.checker_name == "repc":
        checker = RepCChecker(classifier=args.repc_classifier_name, batch_size=args.batch_size_checker)
    else:
        raise NotImplementedError
    
    retriever = None
    if args.use_retrieval:
        if args.retriever_name == "google":
            retriever = GoogleRetriever(args.cache_dir)
        else:
            raise NotImplementedError
    
    if args.aggregator_name == "strict":
        agg_fn = strict_agg
    elif args.aggregator_name == "soft":
        agg_fn = soft_agg
    elif args.aggregator_name == "major":
        agg_fn = major_agg
    else:
        raise NotImplementedError
    
    # load data
    with open(args.input_path, "r") as fp:
        input_data = json.load(fp)

    input_data = input_data # Only 10 for testing
    
    # check triplets
    print('Checking')
    triplet_list = []
    reference_list = []
    question_list = []
    for item in input_data:
        assert "triplets" in item, "triplets field is required"
        triplets = item["triplets"]
        if args.use_retrieval:
            reference = retriever.retrieve(item["response"])
            item["reference"] = reference
        else:
            assert "reference" in item, \
                "reference field is required if retriever is not used."
            reference = item["reference"]
        question = item.get("question", None)
        triplet_list.append(triplets)
        reference_list.append(reference)
        question_list.append(question)

    # print("Print Triplet List", len(triplet_list))
    # print("Print Reference List", len(reference_list))
    # print("Print Question List", len(question_list))

    # Print number of triplets in total
    total_triplets = 0
    for triplet in triplet_list:
        total_triplets += len(triplet)
    # print("Total Triplets:", total_triplets)

    results = checker.check(triplet_list, reference_list, question=question_list)
    # print("Length of results:", len(results))
    agg_results = [agg_fn(r) for r in results]

    # print("Length of input data:", len(input_data))

    for i in range(len(input_data)):
        # Check if triplets are empty
        print("Input_keys", input_data[i].keys())
        break

    print("Length of input data:", len(input_data))
    print("Length of results:", len(results))

    output_data = []

    count = 0
    # In the input data remove all triplets, that are empty
    input_data_new = []
    for item in input_data:
        if item["triplets"] != []:
            input_data_new.append(item)

    print("!!!!!!!!!!!", len(input_data_new))

    # Basically the mismatch is due to input data, and results -> 27 don't have any triplets

    output_data = [{
        **input_data_new[i],
        **{
            "Y": agg_results[i],
            "ys": results[i],
        }
    } for i in range(len(input_data_new))]
    with open(args.output_path, "w") as fp:
        json.dump(output_data, fp, indent=2)
