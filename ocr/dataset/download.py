import argparse
import base64
from pathlib import Path
import json

import requests
import tqdm
from openai import OpenAI
from pydantic import BaseModel
import numpy as np


class CaptchaLabel(BaseModel):
    text: str


def download_data(num: int, endpoint: str, output_dir: str, autolabel: bool):
    print(f"Downloading {num} captcha images...")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    client = OpenAI() if autolabel else None

    for i in tqdm.tqdm(range(num)):
        response = requests.get(endpoint)
        ext = response.headers["Content-Type"].split("/")[-1]
        filename = output / f"_{i:04d}.{ext}"
        with open(filename, "wb") as f:
            f.write(response.content)

        if client is not None:
            base64_image = base64.b64encode(response.content).decode("utf-8")
            completion = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an OCR model. Please label the captcha image either as 4 lowercase letters (abcd) or a formula with 2 numbers, an operator, and an equal sign (a@b=).",
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                            }
                        ],
                    },
                ],
                response_format=CaptchaLabel,
                logprobs=True,
            )

            skip = False
            for p in completion.choices[0].logprobs.content:
                prob = np.exp(p.logprob)
                if prob < 0.5:
                    print(f"Low probability {p.token}: {prob:.2f}")
                    skip = True
                    break
            if skip:
                print(f"Invalid label for {filename}")
                continue

            label = json.loads(completion.choices[0].message.content)["text"]
            if len(label) != 4:
                print(f"Invalid label {label} for {filename}")
                continue

            new_filename = output / f"{label}.{ext}"
            if new_filename.exists():
                print(f"File {new_filename} already exists")
                continue

            filename.rename(new_filename)
            print(f"Labelled {filename} as {new_filename}")

    print(f"Downloaded {num} captcha images to {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num", type=int, help="Number of captcha images to download", default=10
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        help="Endpoint to download captcha images",
        default="https://cos1s.ntnu.edu.tw/AasEnrollStudent/RandImage",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output directory to save captcha images",
        default="data",
    )
    parser.add_argument(
        "--autolabel",
        action="store_true",
        help="Automatically label the captcha images",
    )
    args = parser.parse_args()

    download_data(args.num, args.endpoint, args.output, args.autolabel)


if __name__ == "__main__":
    main()
