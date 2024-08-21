# NTNU Captcha OCR Challenge 2024

Training the OCR model.

## Setup

Use micromamba to create a new environment and install the required packages.

```bash
micromamba create -n ntnu-captcha-ocr python=3.11 pytorch torchvision accelerate datasets tensorboard openai -c pytorch -c conda-forge
```

Activate the environment.

```bash
micromamba activate ntnu-captcha-ocr
```

## Prepare the Dataset

Please use the following command to download some captcha images.

```bash
python -m ocr.dataset.download --num 100
```

And label the images by renaming the images with the correct text. (You can also use `--autolabel` to let GPT-4o Mini label the images for you.)

Push to Hugging Face dataset.

```bash
python -m ocr.dataset.push --repo JacobLinCool/ntnu-captcha-ocr-dataset
```

## Train the Model

```bash
python -m ocr.train
```

## Inference

```bash
python -m ocr.infer path/to/image1.png path/to/image2.jpg
```
