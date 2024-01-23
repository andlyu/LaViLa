# Overview
This project evaluates the performance of the Gemini model, a video question-answering AI from Vertex AI, on the EgoSchema dataset. This is a long-form video QA, where videos are up to 3 minutes long, with questions that require reasoning. This makes the Gemini model expensive to run (~ .36$ per video) The model is tested on videos played at normal, 4x, and 10x speeds to determine the performance drop form speeding up videos.

## Dataset
EgoSchema Dataset: A collection of videos with associated questions and answers.
Video Speed Variations: Each video in the dataset has been modified to play at normal, 4x, and 10x speeds. These are uploaded to GCP for the model to load them. 

## Files Description
- `download_file.py`: Python script for downloading videos from Google Drive into a local dataset folder. 
- `eval_gemini.py`: Script that interfaces with the Gemini model to ask questions about the videos and saves the answers to a csv file.
- `process_gemini_pred.ipynb`: Jupyter notebook that processes the results from gemini_pred.csv, calculating accuracy metrics and confidence intervals for each video speed category.

## Metrics
Accuracy of predictions for each video speed.
Confidence intervals for the proportion of correct answers.
Percentage of chagne in accuracy from normal to 4x speed, and from normal to 10x speed.

## Running the Experiment
Set the `ACCESS_TOKEN` environment variable for Google Drive API access.
Run `download_file.py` to fetch videos, and upload them to the propper GCP file (will need to modify in `eval_gemini.py`).
Use `eval_gemini.py` to query the model and record answers.
Open `process_gemini_pred.ipynb` in a Jupyter environment to analyze results.

## Conclusions + Future Work
Even though the Gemini model samples images, and asks questions about them, decreasing the number of images leads to a drop in performance. (Seeimgly a ~5 % when with a 10x speedup). While this may be a cost effective method to run the model, it deoes sacrifice performance.

## Future work includes:
Training on more images to improve the confidence of the experiments.