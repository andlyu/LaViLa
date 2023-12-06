import json
from tqdm import tqdm
import os
from openai import OpenAI


def process_caption(caption, n_lines=3):
    """Limits the number of captions per chunk"""
    lines = caption.split("\n")
    new_caption = ""
    for line in lines:
        if line.startswith("10 captions for chunk"):
            new_caption += str(n_lines) + line[2:] + "\n"
        elif line and line[0].isdigit() and int(line[0]) < n_lines:
            new_caption += line + "\n"
    return new_caption


def process_answer(answer):
    """
    Find the index of FINAL_ANSWER: in the string answer
    Then return the first digit after that.
    if FINAL_ANSWER is not found, return -1
    """
    if "FINAL_ANSWER:" in answer:
        index = answer.index("FINAL_ANSWER:")
        for i in range(index, len(answer)):
            if answer[i].isdigit():
                return int(answer[i])
    elif "final answer:" in answer.lower():
        index = answer.lower().index("final answer:")
        for i in range(index, len(answer)):
            if answer[i].isdigit():
                return int(answer[i])
    elif "answer:" in answer.lower():
        print(answer)
        index = answer.lower().index("answer:")
        for i in range(index, len(answer)):
            if answer[i].isdigit():
                return int(answer[i])
    print(answer)
    return -1


client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

caption_path = "assets/reproduce_v2/"


if __name__ == "__main__":
    # load subset answers from /workspace/questions.json
    with open("/workspace/questions.json") as f:
        questions = json.load(f)

    # load subset answers from /workspace/questions.json
    with open("/workspace/subset_answers.json") as f:
        subset_answers = json.load(f)

    tokens_used = []
    answers = []
    # for question in tqdm(questions[:2]):
    for q_uid in tqdm(list(subset_answers.keys())[:50]):
        question = [q for q in questions if q["q_uid"] == q_uid][0]
        q_uid = question["q_uid"]

        # load caption from /workspace/captions.json
        with open(f"{caption_path}{q_uid}.txt") as f:
            caption = f.read()

        intro_text = """
            Given a few different captions for a video split into 10 chunks, please answer the following multiple choice question.
            Please look at the captions carefully when answering the questions given five possible options.
            For the question, please think about it first, write your logic for choosing the answer in a few sentences, and then
            answer in the format: FINAL_ANSWER: 0 for option 0, or FINAL_ANSWER: 3 if the correct option is 3.
        """

        question_text = "Question: " + question["question"] + "\n"
        options_text = (
            "Options: " + str([question[f"option {i}"] for i in range(5)]) + "\n"
        )
        caption = process_caption(caption)
        caption = "Captions: " + caption + "\n"

        prompt = f"{intro_text} \n {caption} \n {question_text} \n {options_text}"

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-3.5-turbo",
        )
        answer = chat_completion.choices[0].message.content
        answer_num = process_answer(answer)
        tokens_used.append(chat_completion.usage.total_tokens)
        answers.append(
            {
                "q_uid": q_uid,
                "pred_answer": answer_num,
                "true_answer": subset_answers[q_uid],
            }
        )

        # print accuracy
        print(
            "Accuracy:",
            sum([1 for a in answers if a["pred_answer"] == a["true_answer"]])
            / len(answers),
        )
        print("Answers unparsible", sum([1 for a in answers if a["pred_answer"] == -1]))
        print("Average tokens used:", sum(tokens_used) / len(tokens_used))
        print(
            "total tokens used (in thousands): (price: .3c/1k)", sum(tokens_used) / 1000
        )
