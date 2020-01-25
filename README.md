<p align="center">
    <br>
    <img src="https://raw.githubusercontent.com/huggingface/transformers/master/docs/source/imgs/transformers_logo_name.png" width="400"/>
    <br>
<p>
<p align="center">
    <a href="https://circleci.com/gh/huggingface/transformers">
        <img alt="Build" src="https://img.shields.io/circleci/build/github/huggingface/transformers/master">
    </a>
    <a href="https://github.com/huggingface/transformers/blob/master/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/github/license/huggingface/transformers.svg?color=blue">
    </a>
    <a href="https://huggingface.co/transformers/index.html">
        <img alt="Documentation" src="https://img.shields.io/website/http/huggingface.co/transformers/index.html.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/huggingface/transformers/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/huggingface/transformers.svg">
    </a>
</p>

<h3 align="center">
<p>State-of-the-art Natural Language Processing for TensorFlow 2.0 and PyTorch
</h3>

🤗 Transformers (formerly known as `pytorch-transformers` and `pytorch-pretrained-bert`) provides state-of-the-art general-purpose architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet, CTRL...) for Natural Language Understanding (NLU) and Natural Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between TensorFlow 2.0 and PyTorch.

### Features

This is a clone of HuggingFace transformers where a new pooling mechanism class called `AveragePooler` has been added to `modeling_bert.py` that could substitute `BertPooler`. Different from `BertPooler` that relies only a fully connected layer on top of the final layer hidden state of `[CLS]` for classification tasks, `AveragePooler` leverages all the final layer hidden states using a customized attention layer. In particular, this customized attention layer, uses the linearly transformed of sum of final layer non `[PAD]` token embeddings as the single query vector. This cause that this customized attention layer only outputs a single output vector. This single output vector could be used for different sequence classification tasks. 