import os, sys

sys.path.append(os.path.split(sys.path[0])[0])
from abc import ABC, abstractmethod
from util import *
import time

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import logging
from transformers import (
    AutoTokenizer,
    AutoModel,
    T5Tokenizer,
    T5ForConditionalGeneration,
    AutoModelForCausalLM,
)
import torch
from tqdm import tqdm

from config import device

"""
abstract class Decomposer to define the interface for decomposing texts into atom facts.
"""


class Decomposer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def decompose(self, texts: List[str]) -> List[List[str]]:
        """
        Decompose the input texts into atomic facts.

        This abstract method should be implemented by subclasses to process a list of input texts,
        where each text is a paragraph, and return a list of lists containing decomposed atomic facts
        for each input text.

        Args:
            texts (List[str]): A list of input texts, where each text is a paragraph.

        Returns:
            List[List[str]]: A list of lists, where each inner list contains the decomposed atomic facts
                             corresponding to each input text.
        """
        pass


decompose_prompt = """
Decompose the following sentences into atom facts, response only the decompositions(use JSON format):
Sentence: Tim Finchem (born August 24, 1947) is an American businessman and former Commissioner of the PGA Tour. He served as Commissioner from 1994 to 2016.\n\nFinchem was born in Washington, D.C., and grew up in Bethesda, Maryland. He graduated from the University of Maryland in 1969 with a degree in business administration. He then attended the University of Virginia School of Law, where he earned his Juris Doctor degree in 1972.\n\nFinchem began his career in the golf industry in 1975, when he joined the PGA Tour as its first full-time legal counsel. He was promoted to Deputy Commissioner in 1988 and Commissioner in 1994. During his tenure, the PGA Tour grew from a domestic tour to an international tour, with events in more than 30 countries. He also oversaw the introduction of the FedEx Cup, a season-long points competition that culminates in a four-tournament playoff.\n\nFinchem retired as Commissioner in 2016 and was succeeded by Jay Monahan. He was inducted into the World Golf Hall of Fame in 2017.
Answer: ["Tim Finchem was born on August 24, 1947.","Tim Finchem is an American businessman.","Tim Finchem is a former Commissioner of the PGA Tour.","Tim Finchem served as Commissioner of the PGA Tour from 1994 to 2016.","Tim Finchem was born in Washington, D.C.","Tim Finchem grew up in Bethesda, Maryland.","Tim Finchem graduated from the University of Maryland in 1969.","Tim Finchem earned a degree in business administration in 1969.","Tim Finchem attended the University of Virginia School of Law.","Tim Finchem earned a Juris Doctor degree from the University of Virginia School of Law in 1972.","Tim Finchem began his career in the golf industry in 1975.","Tim Finchem joined the PGA Tour in 1975.","Tim Finchem was the first full-time legal counsel for the PGA Tour.","Tim Finchem was promoted to Deputy Commissioner of the PGA Tour in 1988.","Tim Finchem was promoted to Commissioner of the PGA Tour in 1994.","During Tim Finchem's tenure, the PGA Tour grew from a domestic tour to an international tour.","During Tim Finchem's tenure, the PGA Tour held events in more than 30 countries.","Tim Finchem oversaw the introduction of the FedEx Cup.","The FedEx Cup is a season-long points competition.","The FedEx Cup culminates in a four-tournament playoff.","Tim Finchem retired as Commissioner of the PGA Tour in 2016.","Jay Monahan succeeded Tim Finchem as Commissioner of the PGA Tour.","Tim Finchem was inducted into the World Golf Hall of Fame in 2017."]
Sentence: John Russell Reynolds (1820–1876) was an English lawyer, judge, and author. He was born in London, the son of a barrister, and was educated at Eton College and Trinity College, Cambridge. He was called to the bar in 1845, and became a Queen's Counsel in 1859. He was appointed a judge of the Court of Common Pleas in 1867, and was knighted in 1871.\n\nReynolds was a prolific author, writing on a wide range of topics. He wrote several books on legal topics, including The Law of Libel and Slander (1863), The Law of Copyright (1865), and The Law of Patents for Inventions (1868). He also wrote on a variety of other topics, including history, biography, and literature. He was a frequent contributor to the Saturday Review, and wrote several books on Shakespeare, including The Mystery of William Shakespeare (1848) and The Authorship of Shakespeare (1875). He also wrote a biography of the poet John Keats (1848).
Answer: ["John Russell Reynolds was born in 1820.","John Russell Reynolds died in 1876.","John Russell Reynolds was an English lawyer.","John Russell Reynolds was an English judge.","John Russell Reynolds was an English author.","John Russell Reynolds was born in London.","John Russell Reynolds was the son of a barrister.","John Russell Reynolds was educated at Eton College.","John Russell Reynolds was educated at Trinity College, Cambridge.","John Russell Reynolds was called to the bar in 1845.","John Russell Reynolds became a Queen\'s Counsel in 1859.","John Russell Reynolds was appointed a judge of the Court of Common Pleas in 1867.","John Russell Reynolds was knighted in 1871.","John Russell Reynolds was a prolific author.","John Russell Reynolds wrote on a wide range of topics.","John Russell Reynolds wrote several books on legal topics.","John Russell Reynolds wrote \'The Law of Libel and Slander\' in 1863.","John Russell Reynolds wrote \'The Law of Copyright\' in 1865.","John Russell Reynolds wrote \'The Law of Patents for Inventions\' in 1868.","John Russell Reynolds wrote on topics including history, biography, and literature.","John Russell Reynolds was a frequent contributor to the Saturday Review.","John Russell Reynolds wrote several books on Shakespeare.","John Russell Reynolds wrote \'The Mystery of William Shakespeare\' in 1848.","John Russell Reynolds wrote \'The Authorship of Shakespeare\' in 1875.","John Russell Reynolds wrote a biography of John Keats in 1848."]
Sentence: {sentence}
Answer: """

prefix_prompt = """
Decompose the following sentences into atom facts: {sentence}
"""


class APIDecomposer(Decomposer):

    def __init__(self, api_name: str = "openai", verbose: bool = False):
        super().__init__()
        self.api_name = api_name
        self.verbose = verbose

    def decompose(self, texts: List[str]) -> List[List[str]]:
        from json_repair import (
            repair_json,
        )  # to avoid json decode error due to format mistake

        results = []
        for text in tqdm(texts, disable=not self.verbose):
            response = query_gpt(
                prompt=decompose_prompt.format(sentence=text),
                api_name=self.api_name,
            )
            try:
                results.append(json.loads(response))
            except Exception as e:
                results.append(json.loads(repair_json(response)))

        return results


single_decompose_prompt = """
Decompose the following sentences into atom facts, response only the decompositions: 
Sentence: Crawford was selected by the Tampa Bay Devil Rays in the second round (52nd overall) of the 1999 Major League Baseball Draft and made his MLB debut in 2002. 
Answer: Crawford was selected by the Tampa Bay Devil Rays in the second round of the 1999 Major League Baseball Draft.  \nCrawford was selected 52nd overall in the 1999 Major League Baseball Draft.  \nCrawford made his MLB debut in 2002. 
Sentence: Paul Wilson Brooks (28 May 1921 – 26 January 1946) was an English cricketer. 
Answer: Paul Wilson Brooks was born on 28 May 1921.  \nPaul Wilson Brooks died on 26 January 1946.  \nPaul Wilson Brooks was an English cricketer. 
Sentence: {sentence}
Answer: """


class APISingleDecomposer(Decomposer):

    def __init__(self, api_name: str = "openai", verbose: bool = False):
        super().__init__()
        self.api_name = api_name
        self.verbose = verbose
        self.prompt = single_decompose_prompt

    def _split_text(self, text: str) -> List[str]:
        return stanza_split_sentence(text)

    def decompose(self, texts: List[str]) -> List[List[str]]:
        splitted_texts = [
            self._split_text(text) for text in texts
        ]  # each paragraph is split into sentences

        splitted_len = [
            len(text) for text in splitted_texts
        ]  # store the length of each paragraph

        splitted_texts = [
            sentence for text in splitted_texts for sentence in text
        ]  # flatten the list of lists

        decomposed_texts = []

        for text in tqdm(splitted_texts, disable=not self.verbose):
            response = query_gpt(
                prompt=self.prompt.format(sentence=text),
                api_name=self.api_name,
            )
            decomposed_texts.append(response)

        reconstructed_texts = []
        index = 0
        for length in splitted_len:
            paragraph = "  \n".join(decomposed_texts[index : index + length])
            reconstructed_texts.append(paragraph)
            index += length

        assert len(reconstructed_texts) == len(texts)
        return [self._split_text(text) for text in reconstructed_texts]


class QwenDecomposer(Decomposer):

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        batch_size: int = 1,
        vllm_boost: bool = False,
        lora_path: str = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.vllm_boost = vllm_boost
        self.lora_path = lora_path
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        logging.info(f"Loading model from {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.vllm_boost:
            from vllm import LLM

            self.model = LLM(
                model=self.model_name, enable_lora=self.lora_path is not None
            )
        else:
            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    self.model_name, torch_dtype="auto", device_map="auto"
                )
                .to(device)
                .eval()
            )

    def _format_input(self, text: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prefix_prompt.format(sentence=text)},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    @torch.no_grad()
    def decompose(self, texts: List[str]) -> List[List[str]]:
        decomposed_texts = []
        if self.vllm_boost:
            from vllm import SamplingParams
            from vllm.lora.request import LoRARequest

            sampling_params = SamplingParams(
                max_tokens=2048,
            )
            if self.lora_path:
                generated_ids = self.model.generate(
                    [self._format_input(text) for text in texts],
                    sampling_params,
                    lora_request=LoRARequest("lora", 1, self.lora_path),
                )
            else:
                generated_ids = self.model.generate(
                    texts,
                    sampling_params,
                )
            decomposed_texts.extend(
                [stanza_split_sentence(item.outputs[0].text) for item in generated_ids]
            )
        else:
            for i in tqdm(
                range(0, len(texts), self.batch_size), postfix=self.model_name
            ):
                batch_texts = texts[i : i + self.batch_size]
                batch_texts = [self._format_input(text) for text in batch_texts]
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=2048,
                    return_tensors="pt",
                ).to(device)
                generated_ids = self.model.generate(**inputs, max_new_tokens=2048)
                generated_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
                ]
                response = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                decomposed_texts.extend(
                    [stanza_split_sentence(text) for text in response]
                )
        return decomposed_texts


class QwenSingleDecomposer(Decomposer):

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        batch_size: int = 1,
        vllm_boost: bool = False,
        lora_path: str = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.vllm_boost = vllm_boost
        self.lora_path = lora_path
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        logging.info(f"Loading model from {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote=True,
            padding_side="left",
            max_length=512,
            # trucation=True,
            # padding=True,
        )
        if self.vllm_boost:
            from vllm import LLM

            self.model = LLM(
                model=self.model_name, enable_lora=self.lora_path is not None
            )
        else:
            self.model = (
                AutoModelForCausalLM.from_pretrained(
                    self.model_name, torch_dtype="auto", device_map="auto"
                )
                .to(device)
                .eval()
            )
        # logging.info(f"Model Architecture: {self.model}")

    def _format_input(self, text: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prefix_prompt.format(sentence=text)},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _split_text(self, text: str) -> List[str]:
        return stanza_split_sentence(text)

    @torch.no_grad()
    def decompose(self, texts: List[str]) -> List[List[str]]:
        splitted_texts = [
            self._split_text(text) for text in texts
        ]  # each paragraph is split into sentences

        splitted_len = [
            len(text) for text in splitted_texts
        ]  # store the length of each paragraph

        splitted_texts = [
            self._format_input(sentence) for text in splitted_texts for sentence in text
        ]  # flatten the list of lists

        decomposed_texts = []

        if self.vllm_boost:
            from vllm import SamplingParams
            from vllm.lora.request import LoRARequest

            sampling_params = SamplingParams(
                max_tokens=512,
                repetition_penalty=1.2,
            )
            if self.lora_path:
                generated_ids = self.model.generate(
                    splitted_texts,
                    sampling_params,
                    lora_request=LoRARequest("lora", 1, self.lora_path),
                )
            else:
                generated_ids = self.model.generate(
                    splitted_texts,
                    sampling_params,
                )
            decomposed_texts.extend([item.outputs[0].text for item in generated_ids])
        else:
            for i in tqdm(
                range(0, len(splitted_texts), self.batch_size), postfix=self.model_name
            ):
                batch_texts = splitted_texts[i : i + self.batch_size]

                inputs = self.tokenizer(
                    batch_texts,
                    max_length=512,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                ).to(device)
                generated_ids = self.model.generate(
                    **inputs, max_new_tokens=512, repetition_penalty=1.2
                )
                generated_ids = [
                    output_ids[len(input_ids) :]
                    for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
                ]
                response = self.tokenizer.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                decomposed_texts.extend(response)

        # Reconstruct paragraphs
        reconstructed_texts = []
        index = 0
        for length in splitted_len:
            paragraph = "  \n".join(decomposed_texts[index : index + length])
            reconstructed_texts.append(paragraph)
            index += length
        logging.info(f"Input texts length: {len(texts)}")
        logging.info(f"Reconstructed texts length: {len(reconstructed_texts)}")
        logging.info(f"Decomposed texts length: {len(decomposed_texts)}")
        logging.info(f"Sum of splitted lengths: {sum(splitted_len)}")
        assert len(reconstructed_texts) == len(texts)
        return [self._split_text(text) for text in reconstructed_texts]


class T5Decomposer(Decomposer):
    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        batch_size: int = 32,
        prefix: str = "Decompose the following sentences into atom facts: ",
    ):
        super().__init__()
        self.model_name: str = model_name
        self.batch_size: int = batch_size
        self.model: T5ForConditionalGeneration = None
        self.tokenizer: T5Tokenizer = None
        self.prefix: str = prefix
        self._load_model()

    def _load_model(self):
        logging.info(f"Loading model from {self.model_name}")
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = (
            T5ForConditionalGeneration.from_pretrained(self.model_name)
            .to(device)
            .eval()
        )

    def _split_text(self, text: str) -> List[str]:
        return stanza_split_sentence(text)

    @torch.no_grad()
    def decompose(self, texts: List[str]) -> List[List[str]]:

        splitted_texts = [
            self._split_text(text) for text in texts
        ]  # each paragraph is split into sentences

        splitted_len = [
            len(text) for text in splitted_texts
        ]  # store the length of each paragraph

        splitted_texts = [
            sentence for text in splitted_texts for sentence in text
        ]  # flatten the list of lists

        decomposed_texts = []
        for i in tqdm(
            range(0, len(splitted_texts), self.batch_size), postfix=self.model_name
        ):
            batch_texts = splitted_texts[i : i + self.batch_size]
            batch_texts = [self.prefix + text for text in batch_texts]
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=512,
                    num_beams=5,
                    no_repeat_ngram_size=2,
                    num_return_sequences=1,
                )
            batch_decomposed_texts = self.tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )
            decomposed_texts.extend(batch_decomposed_texts)

        # Reconstruct paragraphs
        reconstructed_texts = []
        index = 0
        for length in splitted_len:
            paragraph = "\n\n".join(decomposed_texts[index : index + length])
            reconstructed_texts.append(paragraph)
            index += length

        assert len(reconstructed_texts) == len(texts)
        return [self._split_text(text) for text in reconstructed_texts]


def generate_decompose(
    decomposer: Decomposer,
    output: str,
    input: str = None,
):
    if input:
        data = read_json(input)
    else:
        data = read_json("data/wiki_bio_hallu/wiki_bio_simple.json")

    texts = [d["wiki_bio_text"] for d in data]
    error_texts = [d["text_with_error"] for d in data]
    texts.extend(error_texts)
    start = time.time()

    decomposed = decomposer.decompose(texts)
    end = time.time()
    logging.info(f"Time: {end - start:.4f}")

    decomposed_texts = decomposed[: len(data)]
    error_decomposed_texts = decomposed[len(data) :]
    out = []
    for dec, err, dat in zip(decomposed_texts, error_decomposed_texts, data):
        out.append(
            {
                "idx": dat["idx"],
                "wiki_bio_text": dat["wiki_bio_text"],
                "text_with_error": dat["text_with_error"],
                "splitted_wiki": dat["splitted_wiki"],
                "wiki_decomposed": dec,
                "text_with_error_decomposed": err,
            }
        )

    write_json(out, output)


if __name__ == "__main__":
    # generate_decompose(
    #     decomposer=T5Decomposer("results/finetuned_t5", batch_size=1),
    #     output="results/t5_decompose.json",
    # )
    # generate_decompose(
    #     decomposer=QwenSingleDecomposer(
    #         model_name="results/qwen2/full/ln=2e5", batch_size=64
    #     ),
    #     input="results/qwen_coref_resolved_ln2e5.json",
    #     output="results/qwen_full_coref_decompose_test.json",
    # )
    # generate_decompose(
    #     decomposer=QwenSingleDecomposer(
    #         model_name="experiment/results/qwen2/full/ln=2e5", batch_size=16
    #     ),
    #     output="experiment/results/qwen_full_decompose_ln2e5_bs_16.json",
    # )

    generate_decompose(
        decomposer=QwenSingleDecomposer(
            model_name="experiment/results/qwen2/kd", batch_size=1
        ),
        output="experiment/results/qwen_kd.json",
    )

    # generate_decompose(
    #     decomposer=APISingleDecomposer(api_name="local", verbose=True),
    #     output="experiment/results/qwen2_7b_decompose.json",
    # )
    pass
