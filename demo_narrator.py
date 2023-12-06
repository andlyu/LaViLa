# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import os
import urllib.request
from collections import OrderedDict

import torch
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
import decord

from lavila.data.video_transforms import Permute
from lavila.data.datasets import get_frame_ids, video_loader_by_frames
from lavila.models.models import VCLM_OPENAI_TIMESFORMER_LARGE_336PX_GPT2_XL
from lavila.models.tokenizer import MyGPT2Tokenizer
from eval_narrator import decode_one

EXPT_NAME = "final"


def main(args, model=None):
    vr = decord.VideoReader(args.video_path)
    num_seg = 4
    num_chunks = 10
    video_id = args.video_path.split("/")[-1].split(".")[0]

    chunks_frames = []

    total_frames = len(vr)
    segment_length = total_frames // num_chunks

    for chunk in range(num_chunks):
        # Calculate start and end frame for each segment
        start_frame = chunk * segment_length
        end_frame = (
            (chunk + 1) * segment_length if chunk < num_chunks - 1 else total_frames
        )

        # Generate frame IDs for the current segment
        frame_ids = get_frame_ids(
            start_frame, end_frame, num_segments=num_seg, jitter=False
        )

        # Load frames for the current segment
        frames = video_loader_by_frames("./", args.video_path, frame_ids)
        chunks_frames.append(frames)

    ckpt_name = "vclm_openai_timesformer_large_336px_gpt2_xl.pt_ego4d.jobid_246897.ep_0003.md5sum_443263.pth"
    ckpt_path = os.path.join("modelzoo/", ckpt_name)
    os.makedirs("modelzoo/", exist_ok=True)
    if not os.path.exists(ckpt_path):
        print("downloading model to {}".format(ckpt_path))
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/lavila/checkpoints/narrator/{}".format(
                ckpt_name
            ),
            ckpt_path,
        )
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = OrderedDict()
    for k, v in ckpt["state_dict"].items():
        state_dict[k.replace("module.", "")] = v

    if model is None:
        # instantiate the model, and load the pre-trained weights
        model = VCLM_OPENAI_TIMESFORMER_LARGE_336PX_GPT2_XL(
            text_use_cls_token=False,
            project_embed_dim=256,
            gated_xattn=True,
            timesformer_gated_xattn=False,
            freeze_lm_vclm=False,  # we use model.eval() anyway
            freeze_visual_vclm=False,  # we use model.eval() anyway
            num_frames=4,
            drop_path_rate=0.0,
        )
        model.load_state_dict(state_dict, strict=True)
    if args.cuda:
        model.cuda()
    model.eval()

    # transforms on input frames
    crop_size = 336
    val_transform = transforms.Compose(
        [
            Permute([3, 0, 1, 2]),
            transforms.Resize(crop_size),
            transforms.CenterCrop(crop_size),
            transforms_video.NormalizeVideo(
                mean=[108.3272985, 116.7460125, 104.09373615000001],
                std=[68.5005327, 66.6321579, 70.32316305],
            ),
        ]
    )
    print("Loading tokenizer")
    tokenizer = MyGPT2Tokenizer("gpt2-xl", add_bos=True)
    output_str = ""

    for frame_idx, frames in enumerate(chunks_frames):
        frames = val_transform(frames)
        frames = frames.unsqueeze(0)  # fake a batch dimension

        with torch.no_grad():
            if args.cuda:
                frames = frames.cuda(non_blocking=True)
            image_features = model.encode_image(frames)
            generated_text_ids, ppls = model.generate(
                image_features,
                tokenizer,
                target=None,  # free-form generation
                max_text_length=77,
                top_k=None,
                top_p=0.95,  # nucleus sampling
                num_return_sequences=10,  # number of candidates: 10
                temperature=1.3,
                early_stopping=True,
            )

        output_str += f"10 captions for chunk {frame_idx}:\n"
        print(f"10 captions for chunk {frame_idx}:\n")
        for i in range(10):
            generated_text_str = decode_one(generated_text_ids[i], tokenizer)
            print("{}: {}".format(i, generated_text_str))
            output_str += "{}: {}\n".format(i, generated_text_str)

    # check if assets/EXPT_NAME exists
    if not os.path.exists(f"assets/{EXPT_NAME}"):
        os.makedirs(f"assets/{EXPT_NAME}")
    with open(f"assets/{EXPT_NAME}/{video_id}.txt", "w") as f:
        f.write(output_str)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser("lavila narrator demo")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument(
        "--video-path",
        default="/workspace/videos/videos/0a01d7d0-11d6-4af6-abd9-2025656d3c63.mp4",
        type=str,
        help="video path",
    )
    args = parser.parse_args()
    main(args)
