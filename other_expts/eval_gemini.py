import vertexai
from vertexai.preview.generative_models import GenerativeModel, Part
import os
import pandas as pd
import json


def generate_text(project_id: str, location: str, video_id: str, question: str) -> str:
    """Ask a video question to the Gemini model and return the answer

    Args:
        project_id (str): project id to use
        location (str): GCP Lcation
        video_id (str): Videos were previously uploaded to "gs://video-qa/{video_id}.mp4"
        question (str): The text question to ask the model

    Returns:
        str: the text answer returned by the model
    """
    # Initialize Vertex AI
    vertexai.init(project=project_id, location=location)
    # Load the model
    multimodal_model = GenerativeModel("gemini-pro-vision")
    # Query the model
    print(f"gs://video-qa/{video_id}.mp4")
    response = multimodal_model.generate_content(
        [
            # Add an example image
            Part.from_uri(f"gs://video-qa/{video_id}.mp4", mime_type="video/mp4"),
            # Add an example query
            question,
        ],
    )
    multimodal_model.generate_content(
        [
            Part.from_uri(
                f"gs://video-qa/sped_up_4x_263666b2-3229-429a-b5a7-defc6433dc29.mp4",
                mime_type="video/mp4",
            ),
            "What is this video about",
        ]
    )

    print(response)
    return response.text


def ask_video_qa(
    question: str, video_id: str, pred_df: pd.DataFrame, gt_answer: int, verbose=True
) -> None:
    """Ask a question about a video and add it to the gt_answer dataframe

    Args:
        question (str): Text question that gets sent to the model
        video_id (str): path to video is retrieved from -> "gs://video-qa/{video_id}.mp4"
        pred_df (pd.DataFrame): df to which the prediction is added
        gt_answer (int): int answer to the question (0-4 inclusive)
    """
    text_answer = generate_text(
        project_id="nlptrading",
        location="us-central1",
        video_id=video_id,
        question=question,
    )
    # get the last index of "Option"
    last_opt_idx = text_answer.rfind("Option")
    option_answer = text_answer[last_opt_idx:]

    pred_answer = -1
    if "Option 0" in option_answer:
        pred_answer = 0
    elif "Option 1" in option_answer:
        pred_answer = 1
    elif "Option 2" in option_answer:
        pred_answer = 2
    elif "Option 3" in option_answer:
        pred_answer = 3
    elif "Option 4" in option_answer:
        pred_answer = 4

    pred_df = pred_df.append(
        {
            "video_id": video_id,
            "question": question,
            "true answer": gt_answer,
            "text answer": text_answer,
            "prediction": pred_answer,
            "correct": int(pred_answer) == (true_answer),
        },
        ignore_index=True,
    )
    pred_df.to_csv("other_expts/gemini_pred.csv", index=False)

    if verbose:
        print("video_id: ", video_id)
        print("text answer: ", text_answer)
        print("pred answer: ", pred_answer)
        print("true answer: ", true_answer)
        print("question: ", question)
        print("-------------------------------------------------------")

    return pred_df


def calculate_and_print_percentage_correct(dataframe: pd.DataFrame, speed_prefix: str):
    """Calculate the percentage of correct answers for a given speed prefix

    Args:
        dataframe (_type_): _description_
        speed_prefix (_type_): _description_
    """
    filtered_videos = dataframe[dataframe["video_id"].str.contains(speed_prefix)]
    if len(filtered_videos) > 0:
        correct_sum = filtered_videos["correct"].sum()
        percentage_correct = correct_sum / len(filtered_videos)
        print(f"{speed_prefix} percentage correct: {percentage_correct}")
    else:
        print(f"No videos found with speed prefix {speed_prefix}")


if __name__ == "__main__":
    """
    This script creates a csv file with gemini predictins for the EgoSchema dataset.
    It evaluates the model on videos of different speeds, and saves the results to gemini_pred.csv

    Data: The videos were previously uploaded to GCP, and questions are loaded from
    questions.json (questions for the whole dataset) + subset_answers.json (quesitons + answers for a subset)
    The script only looks at 100 videos from the subset, due to funding limitatins.

    The prompt is the question + Answer Chices, and a custom post string asking to identify the ID (not tuned much).
    After the results are saved
    """

    # Load prev results if they exist (We don't want to rerun the whole thing in case of failure)
    if os.path.exists("other_expts/gemini_pred.csv"):
        gemini_pred = pd.read_csv("other_expts/gemini_pred.csv")
    else:
        gemini_pred = pd.DataFrame(
            columns=[
                "video_id",
                "question",
                "true answer",
                "text answer",
                "prediction",
                "correct",
            ]
        )
        gemini_pred.to_csv("other_expts/gemini_pred.csv", index=False)

    # Post string after the question and the options from the dataset
    post_string = (
        "\n First justify the answer, and then Please answer either "
        + '"Option 0", "Option 1", "Option 2", "Option 3", "Option 4".'
    )

    # Load the questions
    with open("subset_answers.json") as f:
        subset_answers = json.load(f)
    with open("datasets/questions.json") as f:
        questions = json.load(f)

    # Iterate over the first 100 videos in the subset
    for key in list(subset_answers.keys())[:101]:
        print(key)

        # skip the video that was blocked by Gemini due to safety concerns
        if key == "263666b2-3229-429a-b5a7-defc6433dc29":
            continue

        # check if datasets/{key}.mp4 exists
        if not os.path.exists(f"datasets/{key}.mp4"):
            print(f"File {key}.mp4 does not exist")
            continue

        # Get the question from the questions.json file
        question = [q for q in questions if q.get("q_uid", "") == key][0]
        del question["q_uid"]
        del question["google_drive_id"]
        question = str(question) + post_string

        true_answer = subset_answers[key]

        # Ask the question for the different speed prefixes
        if key not in gemini_pred["video_id"].values:
            gemini_pred = ask_video_qa(
                question, key, gemini_pred, true_answer, verbose=True
            )
        if f"sped_up_4x_{key}" not in gemini_pred["video_id"].values:
            gemini_pred = ask_video_qa(
                question, f"sped_up_4x_{key}", gemini_pred, true_answer, verbose=True
            )
        if f"sped_up_10x_{key}" not in gemini_pred["video_id"].values:
            gemini_pred = ask_video_qa(
                question, f"sped_up_10x_{key}", gemini_pred, true_answer, verbose=True
            )

        # Call the method for different speed prefixes
        calculate_and_print_percentage_correct(gemini_pred, "sped_up_4x_")
        calculate_and_print_percentage_correct(gemini_pred, "sped_up_10x_")
        print()


# generate_text(project_id="nlptrading", location="us-central1", video_id ="00b9a0de-c59e-49cb-a127-6081e2fb8c8e", question=question)
