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

import os
from typing import List, Union
from tqdm import tqdm

from .checker_base import CheckerBase
from ..utils import get_model_batch_response


LLM_CHECKING_PROMPT_Q = \
"""I have a claim that made by a language model to a question, please help me for checking whether the claim can be entailed according to the provided reference which is related to the question. 
The reference is a list of passages, and the claim is represented as a triplet formatted with ("subject", "predicate", "object").

If the claim is supported by ANY passage in the reference, answer 'Entailment'. 
If NO passage in the reference entail the claim, and the claim is contradicted with some passage in the reference, answer 'Contradiction'.
If NO passage entail or contradict with claim, or DOES NOT contain information to verify the claim, answer 'Neutral'. 

Please DO NOT use your own knowledge for the judgement, just compare the reference and the claim to get the answer.

### Question:
{question}

### Reference:
{reference}

### Claim:
{claim}

Your answer should always be only a single word in ['Entailment', 'Neutral', 'Contradiction']. DO NOT add explanations or you own reasoning to the output.
"""

LLM_CHECKING_PROMPT = \
"""I have a claim that made by a language model, please help me for checking whether the claim can be entailed according to the provided reference. 
The reference is a list of passages, and the claim is represented as a triplet formatted with ("subject", "predicate", "object").

If the claim is supported by ANY passage in the reference, answer 'Entailment'. 
If NO passage in the reference entail the claim, and the claim is contradicted with some passage in the reference, answer 'Contradiction'.
If NO passage entail or contradict with claim, or DOES NOT contain information to verify the claim, answer 'Neutral'. 

Please DO NOT use your own knowledge for the judgement, just compare the reference and the claim to get the answer.

### Reference:
{reference}

### Claim:
{claim}

Your answer should always be only a single word in ['Entailment', 'Neutral', 'Contradiction']. DO NOT add explanations or you own reasoning to the output.
"""


class LLMChecker(CheckerBase):
    def __init__(
        self,
        model,
        batch_size=16
    ) -> None:
        """
        Initializer for the LLMChecker class.

        Initializes LLMChecker with the provided model and batch size.

        Parameters:
        -----------
        model : str
            The name or identifier of the language model to use.
        batch_size : int, optional
            Batch size for checking, defaults to 16.
        """

        super().__init__()
        self.prompt_temp = LLM_CHECKING_PROMPT
        self.prompt_temp_wq = LLM_CHECKING_PROMPT_Q
        self.batch_size = batch_size
        if model not in ['gpt4', 'claude2']:
            self.model = model
        elif model == 'gpt4':
            self.model = 'gpt-4'
        elif model == 'claude2':
            self.model = 'bedrock/anthropic.claude-v2' if os.environ.get('AWS_REGION_NAME') else 'claude-2'
        else:
            raise ValueError('The model you specified is not supported.')

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

        ret_labels = []
        prompt_list = []
        for claim, reference, question in zip(claims, references, questions):
            if isinstance(claim, list):
                assert len(claim) == 3
                claim = f"({claim[0]}, {claim[1]}, {claim[2]})"
            if question is None:
                prompt = self.prompt_temp.format(
                    reference=reference,
                    claim=claim
                )
            else:
                prompt = self.prompt_temp_wq.format(
                    question=question,
                    reference=reference,
                    claim=claim
                )
            prompt_list.append(prompt)

        for i in tqdm(range(0, len(prompt_list), self.batch_size)):
            batch_prompts = prompt_list[i:i + self.batch_size]
            llm_responses = get_model_batch_response(
                prompts=batch_prompts,
                temperature=0,
                model=self.model,
                max_new_tokens=10,
            )
            for llm_response in llm_responses:
                if llm_response and len(llm_response):
                    label = None
                    if self.label_contradiction.lower() in llm_response.lower():
                        label = self.label_contradiction
                    elif self.label_entailment.lower() in llm_response.lower():
                        label = self.label_entailment
                    else:
                        label = self.label_neutral
                    ret_labels.append(label)
                else:
                    raise 'API returns None or empty string'
        return ret_labels
