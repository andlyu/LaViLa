from demo_narrator import main as demo_narrator_main
from types import SimpleNamespace
import json
from tqdm import tqdm

if __name__ == "__main__":
    args = SimpleNamespace(
        cuda=True,
        video_path="/workspace/videos/videos/0a01d7d0-11d6-4af6-abd9-2025656d3c63.mp4",
    )

    # load subset answers from /workspace/questions.json
    with open("/workspace/questions.json") as f:
        questions = json.load(f)

    # load subset answers from /workspace/questions.json
    with open("/workspace/subset_answers.json") as f:
        subset_answers = json.load(f)

    model = None
    # for question in tqdm(questions[:50]):
    for q_uid in tqdm(list(subset_answers.keys())[:50]):
        question = [q for q in questions if q["q_uid"] == q_uid][0]
        q_uid = question["q_uid"]
        args.video_path = f"/workspace/videos/videos/{q_uid}.mp4"

        model = demo_narrator_main(args, model)
