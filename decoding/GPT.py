import torch
import numpy as np
import json
import os
import socket

import config
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaForCausalLM,
)
from torch.nn.functional import softmax

HOSTNAME = socket.gethostname()


class GPT:
    """wrapper for https://huggingface.co/openai-gpt"""

    def __init__(self, llm, device="cpu", gpt="perceived", not_load_model=False):
        self.device = device
        self.llm = llm
        self.outputs = None
        if llm == "original":
            self.model = (
                (
                    AutoModelForCausalLM.from_pretrained(config.MODELS[llm](gpt))
                    .eval()
                    .to(self.device)
                )
                if not not_load_model
                else None
            )
            with open(os.path.join(config.DATA_LM_DIR, gpt, "vocab.json"), "r") as f:
                self.vocab = json.load(f)
            self.word2id = {w: i for i, w in enumerate(self.vocab)}
            self.UNK_ID = self.word2id["<unk>"]
        elif "llama3" in llm:
            self.model = (
                LlamaForCausalLM.from_pretrained(
                    config.MODELS[llm], device_map="balanced"
                ).eval()
                if not not_load_model
                else None
            )
            self.tokenizer = AutoTokenizer.from_pretrained(config.MODELS[llm])
            self.word2id = self.tokenizer.vocab
            self.vocab = [
                word for word, _ in sorted(self.word2id.items(), key=lambda x: x[1])
            ]
            self.UNK_ID = (
                self.tokenizer.unk_token_id if self.tokenizer.unk_token_id else 0
            )
        elif llm == "llama70b":
            device_map = {
                'model.embed_tokens': 0,
                'model.embed_dropout': 0,
            }
            for i in range(0, 3):
                device_map[f'model.layers.{i}'] = 0
            for i in range(3, 10):
                device_map[f'model.layers.{i}'] = 1
            for i in range(10, 28):
                device_map[f'model.layers.{i}'] = 2
            for i in range(28, 46):
                device_map[f'model.layers.{i}'] = 3
            for i in range(46, 64):
                device_map[f'model.layers.{i}'] = 4
            for i in range(62, 80):
                device_map[f'model.layers.{i}'] = 5
            # for i in range(0, 20):
            #     device_map[f'model.layers.{i}'] = 2
            # for i in range(20, 40):
            #     device_map[f'model.layers.{i}'] = 3
            # for i in range(40, 60):
            #     device_map[f'model.layers.{i}'] = 4
            # for i in range(60, 80):
                # device_map[f'model.layers.{i}'] = 5
            device_map['model.norm'] = 4
            device_map['lm_head'] = 5
            self.model = (
                (
                    LlamaForCausalLM.from_pretrained(config.MODELS[llm], device_map=device_map)#"auto")#, max_memory={0: "20GB", 1: "20GB", 2: "80GB", 3: "80GB", 4: "80GB", 5: "80GB"})
                    .eval()
                )
                if not not_load_model
                else None
            )
            self.tokenizer = AutoTokenizer.from_pretrained(config.MODELS[llm])
            self.word2id = self.tokenizer.vocab
            self.vocab = [
                word for word, _ in sorted(self.word2id.items(), key=lambda x: x[1])
            ]
            self.UNK_ID = (
                self.tokenizer.unk_token_id if self.tokenizer.unk_token_id else 0
            )
        elif llm == "gpt":
            self.model = (
                AutoModelForCausalLM.from_pretrained(config.MODELS[llm])
                .eval()
                .to(self.device)
                if not not_load_model
                else None
            )
            self.tokenizer = AutoTokenizer.from_pretrained(config.MODELS[llm])
            self.word2id = self.tokenizer.vocab
            self.vocab = [
                word for word, _ in sorted(self.word2id.items(), key=lambda x: x[1])
            ]
            self.UNK_ID = (
                self.tokenizer.unk_token_id if self.tokenizer.unk_token_id else 0
            )
        else:
            if "boole" in HOSTNAME:
                self.model = (
                    AutoModelForCausalLM.from_pretrained(
                        config.MODELS[llm], device_map="balanced"
                    ).eval()
                    if not not_load_model
                    else None
                )
            elif "riemann" in HOSTNAME:
                self.model = (
                    AutoModelForCausalLM.from_pretrained(
                        config.MODELS[llm], device_map="balanced"
                    ).eval()
                    if not not_load_model
                    else None
                )
            else:
                self.model = (
                    AutoModelForCausalLM.from_pretrained(
                        config.MODELS[llm], device_map="balanced"
                    ).eval()
                    if not not_load_model
                    else None
                )
            self.tokenizer = AutoTokenizer.from_pretrained(config.MODELS[llm])
            self.word2id = self.tokenizer.vocab
            self.vocab = [
                word for word, _ in sorted(self.word2id.items(), key=lambda x: x[1])
            ]
            self.UNK_ID = (
                self.tokenizer.unk_token_id if self.tokenizer.unk_token_id else 0
            )

    def encode(self, words, is_test):
        """map from words to ids"""
        if is_test or (self.llm == "original"):
            return [
                self.word2id[x] if x in self.word2id else self.UNK_ID for x in words
            ], list(range(len(words)))
        else:
            word_i = 0
            wordind2tokind, ids = [], []
            ids = self.tokenizer.encode(" ".join(words))
            for id_i in range(len(ids)):
                tok = self.tokenizer.decode(ids[id_i])
                words[word_i] = words[word_i].replace(" ", "")
                if (word_i >= len(words)) or (
                    tok in [" ", "", self.tokenizer.eos_token]
                ):
                    wordind2tokind.append(word_i - 1)
                    continue
                # Skip blank in original script
                while words[word_i] == "":
                    if tok.replace(" ", "") == "":
                        break
                    else:
                        word_i += 1

                if tok in [self.tokenizer.bos_token]:
                    append = word_i
                elif words[word_i] in [tok.replace(" ", ""), tok]:  # The normal case
                    append = word_i
                    word_i += 1
                # if a token is continuation of previous token
                elif tok[0] != " ":
                    tmp_i = 0
                    # Try to find how many tokens previous continuation
                    for tmp_i in range(id_i - 1, max(-1, id_i - 11), -1):
                        if self.tokenizer.decode(ids[tmp_i]) == "":
                            continue
                        # For GPT models
                        if words[word_i] == self.tokenizer.decode(
                            ids[tmp_i : id_i + 1]
                        ).replace(" ", ""):
                            break
                        # For Llama3 models
                        if self.tokenizer.decode(ids[tmp_i])[0] == " ":
                            break
                    # The case where a word is formed
                    if (
                        (
                            (" " if (tmp_i or (words[word_i - 1] == "")) else "")
                            + words[word_i]
                        )
                        == self.tokenizer.decode(ids[tmp_i : id_i + 1])
                        or words[word_i]
                        == self.tokenizer.decode(ids[tmp_i : id_i + 1]).replace(" ", "")
                        or (
                            (
                                self.tokenizer.decode(ids[tmp_i])
                                == self.tokenizer.bos_token
                            )
                            and (
                                words[word_i]
                                == self.tokenizer.decode(ids[tmp_i + 1 : id_i + 1])
                            )
                        )
                    ):
                        append = word_i
                        word_i += 1
                    # The case where a token include "'s"
                    elif self.tokenizer.decode(ids[tmp_i : id_i + 1]) == "'s" and (
                        " " + words[word_i]
                    ) == self.tokenizer.decode(ids[tmp_i]) + self.tokenizer.decode(
                        ids[id_i]
                    ):
                        append = word_i
                        word_i += 1
                    # The case where a token is still not formed
                    elif tok.replace(" ", "") in words[word_i]:
                        append = word_i
                elif tok.replace(" ", "") in words[word_i]:
                    append = word_i
                else:
                    raise
                wordind2tokind.append(append)
        if (len(ids) != len(wordind2tokind)) or (len(words) - 1 > word_i):
            print(ids)
            print(wordind2tokind)
            print(words[word_i - 3 : words[word_i] + 3])
        assert len(ids) == len(wordind2tokind), f"{len(ids)} != {len(wordind2tokind)}"
        assert len(words) - 1 <= word_i, f"{len(words)} != {word_i}"
        return ids, wordind2tokind

    def get_story_array(self, words, context_words):
        """get word ids for each phrase in a stimulus story"""
        nctx = context_words + 1
        story_ids, wordind2tokind = self.encode(words, is_test=False)
        story_array = np.zeros([len(story_ids), nctx]) + self.UNK_ID
        for i in range(len(story_array)):
            segment = story_ids[i : i + nctx]
            story_array[i, : len(segment)] = segment
        return torch.tensor(story_array).long(), wordind2tokind

    def get_context_array(self, contexts):
        """get word ids for each context"""
        context_array = np.array(
            [self.encode(words, is_test=True)[0] for words in contexts]
        )
        return torch.tensor(context_array).long()

    def get_hidden(self, ids, layer, story=None):
        """get hidden layer representations"""
        mask = torch.ones(ids.shape).int()
        if story is None:
            outputs = self.model(
                input_ids=ids.to(self.device),
                attention_mask=mask.to(self.device),
                output_hidden_states=True,
            )
            return outputs.hidden_states[layer].detach().cpu().numpy()
        else:
            with torch.no_grad():
                save_location = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "features", self.llm)
                if self.llm in ["falcon", "falcon7b", "llama70b", "llama3", "llama3.1"]:
                    os.makedirs(save_location, exist_ok=True)
                if os.path.exists(os.path.join(save_location, story+"_layer"+str(layer)+".npy")):
                    return np.load(os.path.join(save_location, story+"_layer"+str(layer)+".npy"))
                result = [[] for _ in range(len(config.GPT_LAYERS[self.llm]))]
                for i in range(ids.shape[0]-ids.shape[-1]+1):
                    id = ids[i].reshape(1, -1)
                    outputs = self.model(
                        input_ids=id.to(self.device),
                        output_hidden_states=True,
                    )
                    for j in range(len(config.GPT_LAYERS[self.llm])):
                        if i == 0:
                            result[j] = outputs.hidden_states[config.GPT_LAYERS[self.llm][j]].detach().cpu().numpy()[0,:,:]
                        else:
                            output = outputs.hidden_states[config.GPT_LAYERS[self.llm][j]].detach().cpu().numpy()[0,-1,:].reshape(1, -1)
                            result[j] = np.concatenate([result[j], output])
                            # output = outputs.hidden_states[layer].detach().cpu().numpy()[0,-1,:].reshape(1, -1)
                if self.llm in ["falcon", "falcon7b", "llama70b", "llama3", "llama3.1"]:
                    for i in range(len(config.GPT_LAYERS[self.llm])):
                        np.save(
                            os.path.join(save_location, story+"_layer"+str(config.GPT_LAYERS[self.llm][i])),
                            result[i]
                        )
                result = result[config.GPT_LAYERS[self.llm].index(layer)]
            print(result.shape, "this must be (len_words, hidden).", story)
            return result

    def get_probs(self, ids):
        """get next word probability distributions"""
        mask = torch.ones(ids.shape).int()
        with torch.no_grad():
            outputs = self.model(
                input_ids=ids.to(self.device), attention_mask=mask.to(self.device)
            )
        probs = softmax(outputs.logits, dim=2).detach().cpu().numpy()
        if (self.llm != "original") and (self.tokenizer.eos_token_id):
            probs[:, :, self.tokenizer.eos_token_id] = 1e-10
        return probs

    def decode_misencoded_text(self, words):
        if self.llm in ["llama3", "llama3.1", "opt", "llama70b"]:
            return [
                w.replace("Ġ", " ")
                .replace("âĢĻ", "'")
                .replace("Ċ", "\n")
                .replace("âĢľ", "“")
                .replace("âĢĿ", "”")
                .replace("<|end_of_text|>", "")
                .replace("<|begin_of_text|>", "")
                .replace("Âł", " ")
                .replace("âĢ", "")
                .replace("ĺ", "")
                .replace("Ãł", "à")
                .replace("Ã©", "é")
                for w in words
            ]
        elif self.llm in ["gpt"]:
            return [w.replace("</w>", " ") for w in words]
        return words
