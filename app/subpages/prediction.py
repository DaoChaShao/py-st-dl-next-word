#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/12/5 20:20
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   prediction.py
# @Desc     :   

from pathlib import Path
from re import match
from random import randint
from stqdm import stqdm
from streamlit import (empty, sidebar, subheader, session_state,
                       button, container, rerun, columns,
                       data_editor, markdown, write, number_input,
                       caption)
from torch import load, device, Tensor, no_grad, nn

from src.configs.cfg_rnn import CONFIG4RNN
from src.nets.rnn4classification import NormalRNNForClassification
from src.utils.helper import Timer
from src.utils.PT import item2tensor
from src.utils.SQL import SQLiteIII
from src.utils.stats import load_json, create_full_data_split
from src.utils.THU import cut_only

empty_messages: empty = empty()
bars = container(width="stretch")
out_title: empty = empty()
out = container(border=1, width="stretch")
display_title: empty = empty()
display = container(border=1, width="stretch")
left, right = columns(2, width="stretch", gap="small")

session4init: list[str] = ["dictionary", "re_dict", "model", "data", "items", "timer4init"]
for session in session4init:
    session_state.setdefault(session, None)
session4pick: list[str] = ["seq_token", "X", "idx", "timer4pick"]
for session in session4pick:
    session_state.setdefault(session, None)
session4pred: list[str] = ["words", "timer4pred"]
for session in session4pred:
    session_state.setdefault(session, None)

with sidebar:
    subheader("Test Settings")

    params: Path = Path(CONFIG4RNN.FILEPATHS.SAVED_NET)
    dic: Path = Path(CONFIG4RNN.FILEPATHS.DICTIONARY)
    if params.exists() and params.is_file():
        empty_messages.warning("The model & dictionary file already exists. You can initialise model first.")

        if session_state["dictionary"] is None and session_state["model"] is None and session_state["data"] is None:
            if button("Initialise Model & Dictionary & Data", type="primary", width="stretch"):
                with Timer("Next Word Prediction") as session_state["timer4pick"]:
                    # Initialise the dictionary and convert dictionary
                    session_state["dictionary"]: dict[str, int] = load_json(dic)
                    session_state["re_dict"]: dict = {idx: word for word, idx in session_state["dictionary"].items()}

                    # Initialise a model and load saved parameters
                    session_state["model"] = NormalRNNForClassification(
                        len(session_state["dictionary"]),
                        embedding_dim=CONFIG4RNN.PARAMETERS.EMBEDDING_DIM,
                        hidden_size=CONFIG4RNN.PARAMETERS.HIDDEN_SIZE,
                        num_layers=CONFIG4RNN.PARAMETERS.LAYERS,
                        num_classes=len(session_state["dictionary"]),
                        dropout_rate=CONFIG4RNN.PREPROCESSOR.DROPOUT_RATIO,
                        accelerator=CONFIG4RNN.HYPERPARAMETERS.ACCELERATOR,
                    )
                    dict_state: dict = load(params, map_location=device(CONFIG4RNN.HYPERPARAMETERS.ACCELERATOR))
                    session_state["model"].load_state_dict(dict_state)

                    # Initialise the test data from sqlite database
                    with SQLiteIII(CONFIG4RNN.DATABASE.TABLE, CONFIG4RNN.DATABASE.COL) as db:
                        session_state["data"] = db.get_all_data()
                        # print(len(session_state["data"]))

                    # Separate the data
                    _, _, sentences = create_full_data_split(session_state["data"])

                    # Tokenise the data
                    with bars:
                        amount: int | None = 100
                        # amount: int | None = None
                        session_state["items"]: list[str] = []
                        if amount is None:
                            for line in stqdm(sentences, total=len(sentences), desc="Tokenizing Test Data"):
                                for item in cut_only(line):
                                    if item in CONFIG4RNN.PUNCTUATIONS.CN or match(r"^[\u4e00-\u9fff]+$", item):
                                        session_state["items"].append(item)
                        else:
                            for line in stqdm(sentences[:amount], total=amount, desc="Tokenizing Test Data"):
                                for item in cut_only(line):
                                    if item in CONFIG4RNN.PUNCTUATIONS.CN or match(r"^[\u4e00-\u9fff]+$", item):
                                        session_state["items"].append(item)
                        # print(items)
                        rerun()
        else:
            empty_messages.info(f"Initialisation completed! {session_state["timer4pick"]} Pick up a data to test.")

            with left:
                markdown(f"**Dictionary {len(session_state['dictionary'])}**")
                data_editor(session_state["dictionary"], hide_index=False, disabled=True, width="stretch")
            with right:
                markdown(f"**Reversed Dictionary {len(session_state['re_dict'])}**")
                data_editor(session_state["re_dict"], hide_index=False, disabled=True, width="stretch")

            if session_state["seq_token"] is None and session_state["X"] is None:
                if button("Pick up a Data", type="primary", width="stretch"):
                    with Timer("Pick a piece of data") as session_state["timer4pick"]:
                        # Convert the whole sentence to sequence using dictionary
                        UNK_IDX: int = session_state["dictionary"]["<UNK>"]
                        sequence: list[int] = [
                            session_state["dictionary"].get(item, UNK_IDX) for item in session_state["items"]
                        ]
                        # print(sequence)
                        # print()

                        # Pick up a random sequence token
                        session_state["idx"]: int = randint(
                            0, len(sequence) - 1 - CONFIG4RNN.PREPROCESSOR.MAX_SEQUENCE_LEN
                        )
                        session_state["seq_token"]: list[int] = sequence[
                            session_state["idx"]: session_state["idx"] + CONFIG4RNN.PREPROCESSOR.MAX_SEQUENCE_LEN
                        ]
                        # Convert the token to a tensor
                        X: Tensor = item2tensor(
                            session_state["seq_token"], embed=True, accelerator=CONFIG4RNN.HYPERPARAMETERS.ACCELERATOR
                        )
                        # Add batch size
                        session_state["X"] = X.unsqueeze(0)
                        rerun()
            else:
                empty_messages.warning(
                    f"You selected a data for prediction. {session_state['timer4pick']} You can repick if needed."
                )

                display_title.markdown(f"**The data you selected**")
                with display:
                    write("".join(
                        session_state["items"][
                            session_state["idx"]: session_state["idx"] + CONFIG4RNN.PREPROCESSOR.MAX_SEQUENCE_LEN
                        ]
                    ))
                    write(session_state["seq_token"])
                    write(session_state["X"])

                top_k: int = number_input(
                    "TOP K", 1, 10, 5, 1, disabled=True, width="stretch", help="Higher Probability"
                )

                if session_state["words"] is None:
                    temperature: float = number_input(
                        "Temperature", 0.5, 1.6, 1.0, 0.1, disabled=True, width="stretch",
                        help="Set a temperature scaler controlling categorical randomness, lower values, lower randomness"
                    )
                    caption("The range is between 0.5 and 1.5.")

                    if button("Predict", type="primary", width="stretch"):
                        with Timer("Predict") as session_state["timer4pred"]:
                            session_state["model"].eval()
                            with no_grad():
                                predictions = session_state["model"](session_state["X"])
                                # print(predictions.shape, predictions)
                                # print()

                                # Set a temperature scaler controlling categorical randomness, lower values, lower randomness
                                temperature: float = CONFIG4RNN.PARAMETERS.TEMPERATURE
                                if 0.5 <= temperature <= 1.5:
                                    predictions = predictions / temperature
                                else:
                                    raise ValueError("Temperature should be between 0.5 and 1.5.")

                                # Convert the output to probabilities
                                probs: Tensor = nn.functional.softmax(predictions, dim=-1)
                                # print(probs.shape, probs)
                                # print()

                                # Get top k value from probabilities (values, indices)
                                _, indices = probs.topk(top_k, dim=-1, sorted=True)
                                # print(indices)  # size: (batches, top_k)
                                # print()

                                session_state["words"]: list[str] = [
                                    session_state["re_dict"].get(idx) for idx in indices.squeeze().tolist()
                                ]
                                rerun()

                    if button("Repick", type="secondary", width="stretch"):
                        for key in session4pick:
                            session_state[key] = None
                        rerun()
                else:
                    empty_messages.success(f"Prediction Complete. {session_state['timer4pred']} You can repredict.")
                    out_title.markdown(f"**Prediction Result**")
                    with out:
                        write(", ".join(session_state["words"]))

                    if button("Repredict", type="secondary", width="stretch"):
                        for key in session4pred:
                            session_state[key] = None
                        for key in session4pick:
                            session_state[key] = None
                        rerun()




    else:
        empty_messages.error("The model & dictionary file does NOT exist.")
