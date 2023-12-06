import numpy as np
import torch
from peft import LoraConfig, PeftModel
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

from constants import *
from postprocessing import postprocess
from utils import set_seed, device

set_seed()

## Inference

# Load model
def load_model_and_tokenizer(path):
    config = LoraConfig.from_pretrained(path)
    model = AutoModelForQuestionAnswering.from_pretrained(
        config.base_model_name_or_path
    )  # base_model will just smth like "t5-small"
    model = PeftModel.from_pretrained(model, path)
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name_or_path, use_fast=True
    )
    model.to(device)
    return model, tokenizer


model, tokenizer = load_model_and_tokenizer(f"{PATH}/t5-squad-tuned")

## Inference


def predict(model, tokenizer, question, context):
    """
    Uses the specified QA model and tokenizer to perform extractive question answering on a given question and context string.
    1. Tokenizes the question and context
    2. Makes predictions using the model and postprocesses the output
    3. Returns the model prediction

    Input:
        -model: QA model to be used for prediction
        -tokenizer: Tokenizer to be used for tokenizing the question and context
        -question (str): Question string
        -context (str): Context string
    Returns:
        -Model prediction (str)

    """

    # Tokenize the question and context
    input = tokenizer(
        question,
        context,
        max_length=SEQ_LENGTH,
        truncation="only_second",
        padding="max_length",
    )

    # convert from native Python lists to tensors
    # also add a dimension since we are not feeding in a batch but model expects extra batch dim
    for key in input:
        input[key] = torch.tensor(input[key], dtype=torch.int64).unsqueeze(0)

    # convert sequence_ids to tensors and add new dim as well, assign to new attribute
    input["sequence_ids"] = torch.tensor(
        np.array(input.sequence_ids(), dtype=float)
    ).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        inference_output = model(
            input["input_ids"].to(device),
            attention_mask=input["attention_mask"].to(device),
        )
    pred = postprocess(input, inference_output, inference=True)[
        0
    ]  # extract from list of length 1 cause we only feed it one example

    return pred


#### Test case 1:

question = "How many weeks are in an academic quarter at UCLA?"
context = "UCLA operates on a quarter system, which means that the academic year is divided into three 10-week quarters. \
On the other hand, UC Berkeley operates on a semester system, which means that the academic year is divided into two 15-week semesters. \
This can affect the pace of classes and the timing of exams and breaks. Additionally, some students may find that they have more opportunities \
to take classes with a quarter system, while others may prefer the longer semester format. Overall, the main difference is the length of the \
academic terms, but it also affects the pace of classes and the timing of exams and breaks."

pred = predict(model, tokenizer, question, context)
print(f"Question: {question} \n\n Context: {context[:100]}... \n\n Prediction: {pred}")

#### Test case 2:

question = "Name the musicians that Laufey have cited as influential in her music."
context = "Laufey Lín Jónsdóttir (/ˈleɪveɪ/ LAY-vay, Icelandic: [ˈlœyːvei ˈliːn ˈjounstouhtɪr̥]; born 23 April 1999), known by the mononym Laufey, \
is an Icelandic singer-songwriter.[2] She achieved notability in the early 2020s, and describes her musical style, a mixture of \
jazz pop and bedroom pop, as 'modern jazz'.[3] Having performed as a cello soloist with the Iceland Symphony Orchestra at age 15, Laufey was a \
finalist on Ísland Got Talent (2014) and semi-finalist on The Voice Iceland (2015). In 2021, she released her debut EP, Typical of Me, and graduated \
from the Berklee College of Music in Boston, United States. Laufey's debut album, Everything I Know About Love, was released in 2022 and charted in \
Iceland and the US. It was followed by Bewitched, which was released in 2023 to critical acclaim and earned her a Best Traditional Pop Vocal Album \
nomination at the 2024 Grammy Awards; the single 'From the Start' found moderate success in Canada, New Zealand and the United Kingdom. Laufey has \
cited Ella Fitzgerald, Billie Holiday, Chet Baker and Taylor Swift as sources of influence."

pred = predict(model, tokenizer, question, context)
print(f"Question: {question} \n\n Context: {context[:100]}... \n\n Prediction: {pred}")

### Test case 3:

question = "Name the group members that worked on this project."
context = "The grading scheme of Math 156 --Machine Learning-- during \
Fall 2023 consists of three assignments, two midterms, and a project. One group in the class focused \
their project on natural language processing (NLP). The \
names of the members of that group are Terry Ming, Seita Yoshifusa, Christopher Fu, and Nam Truong. \
They started working on the project since week 3 of the quarter and \
will be assigned a random date to present their project to the class."

pred = predict(model, tokenizer, question, context)
print(f"Question: {question} \n\n Context: {context[:100]}... \n\n Prediction: {pred}")

### Test case 4:

question = "What new model is believed to represent a closer step towards AGI?"
context = "Ahead of OpenAI CEO Sam Altman’s four days in exile, several staff researchers wrote a letter to the board of directors \
warning of a powerful artificial intelligence discovery that they said could threaten humanity, two people familiar with the matter told Reuters. \
The previously unreported letter and AI algorithm were key developments before the board's ouster of Altman, the poster child of generative AI, \
the two sources said. Prior to his triumphant return late Tuesday, more than 700 employees had threatened to quit and join backer Microsoft (MSFT.O) \
in solidarity with their fired leader. After being contacted by Reuters, OpenAI, which declined to comment, acknowledged in an internal message to \
staffers a project called Q* and a letter to the board before the weekend's events, one of the people said. Some at OpenAI believe Q* \
(pronounced Q-Star) could be a breakthrough in the startup's search for what's known as artificial general intelligence (AGI), \
one of the people told Reuters."
# https://www.reuters.com/technology/sam-altmans-ouster-openai-was-precipitated-by-letter-board-about-ai-breakthrough-2023-11-22/

pred = predict(model, tokenizer, question, context)
print(f"Question: {question} \n\n Context: {context[:100]}... \n\n Prediction: {pred}")

### Test case 5:

question = (
    "Which electives from the Statistics department can an Applied Math major take?"
)
context = "Required: Mathematics 31A or 31AL, 31B, 32A, 32B, 33A, 33B, Physics 1A, 1B, Program in Computing 10A, and one course \
from Chemistry and Biochemistry 20A, 20B, Physics 1C. \n\n Required: Mathematics 115A, 131A, either 131B or 132, 142; two two-term \
sequences from two of the following categories: numerical analysis—courses 151A and 151B, probability and statistics—courses 170A and 170B, \
or Statistics 100A and 100B, differential equations—courses 134 and 135; four courses from 106 through 199 and Statistics 100A through 102C \
(appropriate courses from other departments may be substituted for some of the additional courses provided departmental consent is \
given before such courses are taken). Mathematics 115A is required of all majors and is intended to be the first upper-division course taken. \
It is strongly advised that students take Mathematics 115A as soon as the major is declared, if not earlier."

pred = predict(model, tokenizer, question, context)
print(f"Question: {question} \n\n Context: {context[:100]}... \n\n Prediction: {pred}")
