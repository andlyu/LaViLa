import json
from tqdm import tqdm
import os
from openai import OpenAI


def process_caption(caption, n_lines=5):
    """Limits the number of captions per chunk"""
    lines = caption.split("\n")
    new_caption = ""
    for line in lines:
        if line.startswith("10 captions for chunk"):
            new_caption += str(n_lines) + line[2:] + "\n"
        elif line and line[0].isdigit() and int(line[0]) < n_lines:
            new_caption += "    " + line[5:] + "\n"
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
        index = answer.lower().index("answer:")
        for i in range(index, len(answer)):
            if answer[i].isdigit():
                return int(answer[i])
    elif "final_answer" in answer.lower():
        index = answer.lower().index("final_answer")
        for i in range(index, len(answer)):
            if answer[i].isdigit():
                return int(answer[i])
    print(answer)
    return -1


client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

caption_path = "/Users/andrew/Desktop/final/"


if __name__ == "__main__":
    # load subset answers from /workspace/questions.json
    with open("/Users/andrew/Desktop/final/questions.json") as f:
        questions = json.load(f)

    # load subset answers from /workspace/questions.json
    with open("/Users/andrew/Desktop/final/subset_answers.json") as f:
        subset_answers = json.load(f)
        
    # load subset answers from /workspace/questions.json
    with open("/Users/andrew/Desktop/final/subset_answers.json") as f:
        subset_answers = json.load(f)

    question_answers = []
    with open("/Users/andrew/Desktop/final/answers.json") as f:
        question_answers = [json.loads(line) for line in f.readlines()]
        
    with open("/Users/andrew/Desktop/final/answers2.json") as f:
        new_question_answers = [json.loads(line) for line in f.readlines()]
        for q in new_question_answers:
            question_answers.append(q)

    tokens_used = []
    answers = []
    
    
    
    # for question in tqdm(questions[4120:]):
    #for question in tqdm(questions[4120:]):
    for question_answer in tqdm(question_answers):
        if(question_answer["pred_answer"] != -1):
            continue
        question = [q for q in questions if q["q_uid"] == question_answer["q_uid"]][0]
        # for idx, q_uid in tqdm(enumerate(list(subset_answers.keys())[:500])):
        #question = [q for q in questions if q["q_uid"] == q_uid][0]
        q_uid = question["q_uid"]

        # load caption from /workspace/captions.json
        with open(f"{caption_path}{q_uid}.txt") as f:
            caption = f.read()

#         intro_text = """
# You are asked to address the task of Video Question Ansewring by only looking at captions of the video.

# We first split a video into 10 chunks and then caption each chunk with three captions. Based on these captions, you will need to ansewr the multiple choice questoin provided. 
# Please look at the captions carefully when answering the questions given five possible options.
# For the question, please summarize the question and the answers, write your logic for choosing the answer in a few sentences, and then
# answer in the format: FINAL_ANSWER: x for option x. Where x is a value from 0 to 4. 
# Also, Note: "C" is the person observing in the ego-centric videos. 
#         """
        
        intro_text = """
            We first split a video into 10 chunks and then caption each chunk with five captions, please answer the following multiple choice question.
            Please look at the captions carefully when answering the questions given five possible options.
            For the question, please summarize the question and the answers, write your logic for choosing the answer in a few sentences, and then
            answer in the format: FINAL_ANSWER: x for Option x. Where x is a value from 0 to 4. 
            If here is no answer, please make an educated guess.
        """

        #question_text = "Question: " + question["question"] + "\n"
        copy_question = question.copy()
        del copy_question["q_uid"], copy_question["google_drive_id"]
        copy_question = (
            f"Question: {copy_question['question']} \n" +
            f"Option 0: {copy_question['option 0']} \n" +
            f"Option 1: {copy_question['option 1']} \n" +
            f"Option 2: {copy_question['option 2']} \n" +
            f"Option 3: {copy_question['option 3']} \n" +
            f"Option 4: {copy_question['option 4']} \n"
        )
        # options_text = (
        #     "Options: " + str([f"option {i} " + question[f"option {i}"] for i in range(5)]) + "\n"
        # )
        caption = process_caption(caption)
        caption = "The following are Captions to the Video Chunks: \n" + caption + "\n"

        prompt = f"{intro_text} \n\n {copy_question} \n\n {caption}"

    
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
                #"true_answer": subset_answers[q_uid],
            }
        )
        # if(idx == 0):
        #     print(prompt)
            
        # append the last answer to answers.json
        with open("/Users/andrew/Desktop/final/answers3.json", "a") as f:
            json.dump(answers[-1], f)
            f.write("\n")

        # print accuracy
        # print(
        #     "Accuracy:",
        #     sum([1 for a in answers if a["pred_answer"] == a["true_answer"]])
        #     / len(answers),
        # )
        print("Answers unparsible", sum([1 for a in answers if a["pred_answer"] == -1]))
        print("Average tokens used:", sum(tokens_used) / len(tokens_used))
        print(
            "total tokens used (in thousands): (price: .3c/1k)", sum(tokens_used) / 1000
        )
