# Coding a Transformer from Scratch

I started this project to get more familiar with Transformer models by coding one from scratch, following Umar Jamil's helpful tutorial! While this is just an implementation of a vanilla Transformer model for translation, as described in the landmark [Attention is All You Need paper](https://arxiv.org/abs/1706.03762), it was a very fun exercise to get intimately familiar with the inner workings of the model.

It was also a great opportunity to learn the process of going from custom code to [hosting the model on HuggingFace](https://huggingface.co/gretakate/english-to-italian-transformer-from-scratch) for the community to use. Check out the [huggingface_upload](https://github.com/gretakate/coding-transformer/tree/main/huggingface_upload) folder for an example of how to set this up!

# Contents

### [1. Getting Started](#getting-started)
### [2. Training](#training)
### [3. Inference](#inference)
### [4. Roadmap](#roadmap)
### [5. Acknowledgements](#acknowldegements)

# Getting Started

For any local development, training, or inference, create a python 3.9 environment and install the requirements.

```
# After creating your 3.9 virtual environment
source venv/bin/activate

pip install -r requirements.txt
```

To run the tests, run
```
pytest tests/
```

# Training

### Train Locally

Use `local_train.ipynb` as a guide for training locally. You probably only want to do this if you have GPU access. I have found that MPS does not play nicely with PyTorch and cpu is too slow to be useful for this large of a training job.

### Train on Google Colab

I used Google Colab for all of my training! You can use `colab_train.ipynb` as a reference. With an account on Google Colab you can get access to T4 GPUs for free (although this depends on availablility). For reference, it seemed to take about 22 minutes per epoch for a batch size of 24 and learning rate 10^-4 on a T4 and about 5.5 minutes per epoch on an A100.

# Inference

## Local Inference

Use `local_inference.ipynb` as a guide for running inference locally. Make sure to set your `model_folder` and `preload` configs to point to where you stored your model weights after training.

## Colab Inference (from HuggingFace weights)

If you want to try out inference from the model I uploaded to HuggingFace, open this Colab notebook! (TODO)

Or, try it out in the interactive app [here]()! (TODO)


# Roadmap

- [ ] Compare the difference in using a single tokenizer for both the source and target languages.
- [ ] Experiment with different hyperparameter settings to improve performance

# Acknowldegements

I would like to thank Umar Jamil for his work in putting out such fantastic tutorials on Transformer models. This project was started by following along his youtube video: [Coding a Transformer from scratch on PyTorch, with full explanation, training and inference](https://www.youtube.com/watch?v=ISNdQcPhsts&t=7790s). I found that the best way to learn from the video was to watch segments at a time, then code each module myself and check back on his implementation for reference.

Umar Jamil publishes the code used in the video tutorial here: https://github.com/hkproj/pytorch-transformer 

In addition to implementing and refactoring the Transformer model and training loop that Umar Jamil teaches, I...
- Added modifications to the tokenization of SOS,EOS,PAD tokens to align with source and target tokenizations in the correct places. In the original code there were some places where SOS/EOS/PAD tokens for the encoder input were tokenized in the target language for training but for inference tokenized with the source language tokenizer. Loss was being calculated by ignorning the PAD token in the source language, when it should have ignored the PAD token in the target language. When I re-trained the model after this fix, performance improved greatly.
- Added tests
- Added HuggingFace integration with a custom model class and configuration
- Hosted the model weights on HuggingFace
- Hosted an interactive app on HuggingFace spaces