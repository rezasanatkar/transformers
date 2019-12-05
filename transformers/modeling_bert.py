# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model. """

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import math
import os
import sys

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss

from .modeling_utils import PreTrainedModel, prune_linear_layer
from .configuration_bert import BertConfig
from .file_utils import add_start_docstrings

logger = logging.getLogger(__name__)

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    'bert-base-german-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
    'bert-base-german-dbmdz-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-pytorch_model.bin",
    'bert-base-german-dbmdz-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-pytorch_model.bin",
}


def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model.
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error("Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, l[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new}


BertLayerNorm = torch.nn.LayerNorm


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()

        #nn.Embedding encapsulates a tensor that represent embeddings and provides some functionalities on top of the
        #embedding tensor in order to slice it in order to return tensor embeddings corresponding to specific vocab indices
        #In particular, if you look into pytorch/torch/nn/modules/sparse.py which contains the implementation of Embedding
        #class, you notice that class Embedding is also derived from nn.Module. This means that the Embedding class only
        #needs to define the forward method and the backprob fn is provided by nn.Module.

        #To instantiate an object of nn.Embeddings, you nedd provide num_embeddings which is the vocab.size, embedding_dim
        #which is the hidden_size in the case of transformers like BERT. One optional argument that is used in below for
        #word_embeddings is padding_idx. In NLU applications, in a lot of situations, we end up with padding the sequences with
        #a special padding token in order to fasciliate batch processing of sequences. For example, if you want to do sequence
        #classification and in a given mini-batch, some sequences are shorter than the longest sequnece, you will end up wih
        #padding the shorter sequences in the minibatch to match the lenght of the longest sequence. In order to ensure that this
        #padded tokens do not impact the decision and states of the network for those shorter sequences, the embedding corresponding
        #to the padding token must be all zero vector. Using padding_idx, we are telling nn.Embedding which vocab index is corresponding
        #to the especial padding token. Here, for the BERT vocan, the index vocab of the special padding token is 0 and therefore,
        #padding_idx=0. This means that everytime that you provide token indices to nn.Embeddings to receive embeddings in return,
        #for tokens with vocab indices of 0, you will get all-zero vectors. The vocab index of the padding token being 0 for BERT
        #is not random and is by design. If you look into the BERT vocab file at ~/.cache/torch/transformers/26bc1ad6c0ac742e9b52263248f6d0f00068293b33709fae12320c0e35ccfbbb.542ce4285a40d23a559526243235df47c5f75c197f04f37d1a0c124c32c9a084, you will see that the first
        #token is [PAD].
        
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        #for bert-base-uncased, config.vocab_size is 30522, config.hidden_size is 768.
        
        
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        #for position_embeddings, the number of embedding vectors is equal to max_position_embeddings which is equal to 512
        #It is because the input context window size is 512. The embedding size for these embeddings is equal to 768.
        
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        #for BERT, type_vocab_size is 2. Therefore, token_type_embeddings will have two vectors of dimension 768

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        #BertLayerNorm is simply torch.nn.LayerNorm. Different from BatchNormalization, the normalization in the case of
        #LayerNorm does not occur over the batch dimension and statistics are not aggregated across the examples of a minibatch.
        #In particualr, you have the option to specify the exact dimensions that you want to perform normalization over using
        #the first argument that you pass to torch.nn.LayerNorm. Here, we are passing config.hidden_size which is equalt to 768
        #to LayerNorm which is telling LayerNorm that it needs to do normalization by aggregating statistics across only the last
        #dimension that has size of 768. That being said, you have the option to pass LayerNorm a tuple like (10, 768) which will
        #force LayerNorm to aggregate statistics across the last two dimensions where the last dimension has size 768 and the dimension
        #before the last dimenstion has size 10. Moreover, similar to BatchNorm, LayerNorm has learnable parameters alpha and beta
        #by default. You can disable these learnable parameters by passing the following keyword argument to LayerNorm:
        #elementwise_affine = False. Another optional argument for LayerNorm is eps which is used in the dominator of normalization
        #to avoid division by zero. layer_norm_eps is equal to 1e-12

        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #hidden_dropout_prob is equal to 0.1. nn.DropOut is derived from nn.Module
        #during training, DropOut zeros out the activations with propbability 0.1. This means that the activations stay intact
        #with probability 0.9. To hemogonize training with inference, during training, the activations that are not zero-out, will
        #be multipled by 1 / .9 = 10 / 9. This normalization during training will let us to simply turn off (drop out probability
        #of zero) the dropout for inference. 

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        #the most confusing part about this forward method is the fact that it takes both input_ids and input_embeds which seems to
        #be redundant since if we know already the embeddings for inputs, what is the reason to use an object of this class. In fact,
        #that is the case, we only retrieve input emebedding using input_ids from word_embeddings if input_embeds is not provided.

        #if input_embds is provided which means that we do not need word_embeddings to extract input_embds, the BertEmbedding
        #object is used to only extract the position embeddings and add them to the passed input embeddings. 
        
        if input_ids is not None:
            input_shape = input_ids.size()
            #here, the assumption is that input_ids is a tensor(batch_size, max_len_sequence) where the shorter sequences are padded
            #by the special padding token [PAD]. In particular, input_ids are the vocab indices of tokens
        else:
            input_shape = inputs_embeds.size()[:-1]
            #here, input_embeds is already the embeddings of the input sequences. Therefore, it will be a tensor of shape
            #(batch_size, max_len_sequence, embedding_size = 768). Therefore, the input_shape that is supposed to be
            #(batch_size, max_len_sequnece) will be extracted from input_embeds as follows: input_embeds.size()[:-1]

        seq_length = input_shape[1]
        #it is obvious now that the seq_length will be the second element of input_shape
        
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        #here, we want to decide what should be the device of position_ids and token_type_ids in the case if they are not given to
        #us and we plan to instantiate them here. In order to extract the device, the firs step is to check which one of input_ids and
        #input_embeds is available and use its device as the device for position_ids and token_type_ids that are about to be created.
        
        if position_ids is None:
            #here, we check it position_ids are not given, we need to manually create them.
            
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            #given that for all the sequences in this minibatch, they have same seq_lenght (shorter ones are padded), they
            #essentially share a same 1-dimensional position ids tensor that can be created using torch.arange.
            #torch.arange will return a sequence of numbers as follows [0, 1, 2, ..., seq_lenght - 1]. Note this is a 1-dimensional
            #tensor
            
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
            #in above, position_ids.unsqueeze(0) will add a dimension of size 1 at the position 0. Therefore, it transforms
            #position_ids from a 1d tensor(seq_length) to a 2d tensor(1, seq_lenght). That is transforming
            #[0, 1, 2, ..., seq_lenght - 1] to [[0, 1, 2, ..., seq_lenght - 1]]

            #the second transformation applied to position_ids is expand method which is similar to tile operation in the sense it
            #increases the size of a dimension of a tensor by copying its values across the other dimension. For example, you
            #can expand the position_ids from a tensor of shape(1, seq_lenght) to a tensor of shape(batch_size, seq_lenght) by applying
            #expand([batch_size, seq_lenght]) to the unaqueezed position_ids tensor. The final position_ids tensor will be the
            #following: [[0, 1, 2, ..., seq_lenght - 1], [0, 1, 2, ..., seq_lenght - 1], ..., [0, 1, 2, ..., seq_length - 1]]

            
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
            #here, if token_type_ids is not provided, we create a zero tensor of shape(batch_size, seq_lenght)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
            #this is the main section of this forward method. If input_embeds is not passed as an argument of the forwrard method,
            #then we use word_embedding to extract the embedding vectors corresponding to these input_ids. Therefore, if
            #input_ids is a 2d tensor(batch_size, seq_lenght), then input_embeds will be a
            #3d tensor(batch_size, seq_lenght, embedding_size)

            
        position_embeddings = self.position_embeddings(position_ids)
        #here, position_ids is a 2d tensor(batch_size, seq_lenght) and position_embeddings will be a
        #3d tensor(batch_size, seq_lenght, embedding_size)
        
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        #here, token_type_ids is a all-zero tensor(batch_size, seq_lenght) and token_type_embeddings will be
        #a 3d tensor(batch_size, seq_lenght, embedding_size)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        #here, given all the above tensors are 3d tensor(batch_size, seq_lenght, embedding_size), we simply add them together
        #to get a unified 3d tensor(batch_size, seq_lenght, embedding_size)

        
        #after computing embeddings in above, the we apply LayerNorm on top of these embeddings that aggregate the statistics across
        #only the last dimension of embeddings which has size of embedding_size = 768. It means that normalization happens per each
        #embedding vector separately.
        embeddings = self.LayerNorm(embeddings)

        #finally, we apply dropout with the dropout probability of 0.1
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()

        #for bert-base-uncased, hidden_size is 768 and num_attention_heads is 12. Which means that the internal hidden size for each
        #head will be 768 / 12 = 64

        #in below, we check if hidden_size is a factor of num_attention_heads
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))

        #output_attentions is False for bert-base-uncase
        self.output_attentions = config.output_attentions

        #num_attention_heads is 12 for bert-base-uncased
        self.num_attention_heads = config.num_attention_heads

        #attention_head_size which the size of internal hidden state for each head is equal to 768 / 12 = 64
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)

        #all_head_size will be equal to 768 which is the concatenation of the ouput of the 12 heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        #in below, all the three query, key and value linear layers take the input embeddings with dimension size 768 and
        #generate query, key and value vectors for all the 12 heads simultanously. In particular, each input embedding is a
        #row vector of 1 x 768 which is multiplied by 768 x 768 query, key and value weight matrices. Therefore, the output vector
        #corresponding to each of this weigth matrices will be 1 x 768 row vector as well. Then each such 1 x 768 row vector is
        #divided into 12 slices of 64 embedding size and will be used by one of the 12 heads. This means that query, key and
        #value weight 768 x 768 matrices are formed by concatenation of query, key and value 768 x 64 weight matrices of 12 heads.
        #in other words, for each 768 x 768 matrix, the first 64 columns are correspodning to the first head, the second 64 columns
        #are corresponding to the second head, and so on 

        #both hidden_size and all_head_size are 768 where hidden_size is in_features and all_head_size is out_features
        self.query = nn.Linear(config.hidden_size, self.all_head_size) 
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        #the attention probability is 0.1 for bert-base-uncased
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        #here, x is a 3d tensor (batch_size, seq_length, 768) where it could be a 3d tensor corresponding eitehr to keys, values or queries.
        #for each given token, all of the 3 query, value and key vectors are row vectors of size (1 x 768). However, we know that in the case of
        #multi-head attentions, we need to divide all these 3 vectors into 12 subvectors of size (1 x 64). It is the main functionality of this method.
        #In particular, this method realizes this functionality by reshapeing these three row vectors of size (1 x 768). The last dimensions of query,
        #value and key tensors have size of 768, and we need to break it into two dimensions of sizes 12 and 64 where 12 is the numebr of heads and
        #64 is the internal embedding size of each individual head. Therefore, the reshaping will transform the tensors from the
        #size(batch_size, seq_lenght, 768) to size(batch_size, seq_lenght, 12, 64). After this reshaping operation and before returnning the resahped
        #tensor, we need to switch the second and third dimension so that we get a tensor of size(batch_size, 12, seq_lenght, 64). This means that before
        #fir each given token, we had all its correspodning keys, values and queries next to each other, and now after this permutation, for each given
        #head, we have all of its corresponding keys, values and queries corresponding to different tokens next to each other. 
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        #here, hidden_states is the input to the self-attention layer that needs to be processed by this layer. In other words,
        #hidden_states is the ouput embedding vectors generated by the previous self-attention layer. Therefore, hidden_states
        #must be a tensor(batch_size, seq_length, embedding_size)

        #note: query matrix is 768 x 768 and hidden_states is a tensor(batch_size, seq_lenght, 768). Therefore, we have
        #hidden_states x query_matrix = queries which is a 3d tenor(batch_size, seq_lenght, 768). The multiplication of hidden_states
        #and query_matrix is a normal matrix multiplication since hidden_states is a batch_size x seq_lenght x 768 tensor and
        #query matrix is 768 x 768 tensor. This means that each input embedding row vector of 1 x 768 will be transformed into
        #a 1 x 768 row vector by multiplication by the 768 x 768 query weight matrix. 
        mixed_query_layer = self.query(hidden_states)
        #the above mixed_query_layer is a tensor(batch_size, seq_lenght, 768) that containes the query vector for tokens of different sequences belonging
        #to a minibatch for all the heads. In particual, you need divide each query row vector of (1 x 768) into 12 parts and each part with dimension of
        #64 is corresponding to a head.


        # ** using self-attention layer in decoder part of seq2seq
        #the below if condition is very interesting in the sense that it enables this self attention layer also to be used as a decoder self-attention layer
        #in seq2seq modeles and not just as a self-attention layer in encoders. For this self-attention layer to be used in a decoder part of a seq2seq
        #model, its keys and values are not transformation of its hidden_states input (the embedding output vectors from the previous self-attention layer
        #in decoder) and only its queries are transformation of its hidden states input. In particualr, a self-attention in decoder part of seq2seq model
        #generates its keys and values as transformation of hidden states (embedding vectors) produced by a self-attention layer in encoder part of the
        #seq2seq model. This makes sense for the decoder to generate a new sequence by looking at the encoded input sequence. Using self-attention layer
        #in decoder part of a seq2seq model, there is no a specific requirement that the seq_length of decoder is same as the seq_lenght of encoder.
        #In particular, different sequence lenghts of decoder and encoder do not make any issue for a self-attention layer in decoder since the number of
        #query vectors could be different from the number of value and key vectors. However, the number of value vectors must be same as the number of
        #key vectors. This holds for a self-attention layer being used in decoder since both key and value vectors come from a self-attention layer in
        #encoder.

        # ** what is attention mask?
        #in order to be able to process sequences with different lenghts in a given minibatch, we need to pad them using [PAD] token to make shorter
        #sequences to have same lenght as the longer ones. Also, we will truncate those sequences that are longer than the max_seq_lenght of 512.
        #that being said, we know that the nn.Embedding used here, will return vectors of all-zeros for [PAD] tokens. Therefore, when a self-attention
        #layer process its inpur embedding vectors, it only makes sense it only involves the non-PAD tokens to compute output embedding vectors corresponding
        #to non-PAD tokens. That being said, you expect from a self-attention layer to ensure that the number of ouput embedding vectors is equal to the
        #number of input embedding vectors. This means that a self-attention layer should not use the embeddings of PAD tokens to compute the output
        #embedding vectors but it needs to bypass them to output. The attention mask simply specifies which tokens are actual tokens and which ones are
        #PAD tokens.

        # ** what changes need to be made to attention mask for a self-attention layer in decoder part of a seq2seq model?
        #since for a self-attention layer in decoder part of a seq2seq model, the keys and values are from a seprate self-attention layer in encoder
        #part of the seq2seq model, the attention-mask needs to be replaced from the one that distinguish between actual tokens and PAD tokens in
        #the self-attention layer in encoder. This change in attention-mask being applied in below. 
        
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            #if encoder_hidden_states is not None, it means that we are using this self-attention layer in decoder part of a seq2seq model, and
            #encoder_hidden_states are the output vectors from a self-attention layer in encoder part of the seq2seq model.
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        #transpose_for_scores reshape the tensors of shape(batch_size, seq_lenght, 768) to tensors of size(batch_size, 12, seq_length, 64) where
        #12 denotes the number of heads and 64 is the size of the internal embedding of each head.

        # ** torch.matmul is a batched matrix multiplication if at least one of the tensors has rank of at least three:
        #this behavior is by design in pytorch. In a batched matrix multiplication, only the last two dimensions are involved in 2d matrix multiplication
        #and all the other dimensions are considered as batch dimensions. An example of matmul batched multiplication is the following:
        
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        #in above, both key_layer and query_layer are (batch_size, 12, seq_lenght, 64). First, we transpose key_layer by swapping its 3rd and 4th
        #dimension. This means that the transposed version of ley_layer will be (batch_size, 12, 64, seq_lenght). Also, since both of these tensors
        #have at least rank of 3, matmul will perform a batched norm multiplication. That is the output of the following multiplication
        #query(batch_size, 12, seq_lengh, 64) x key(batch_size, 12, 64, seq_length) = attention(batch_size, 12 , seq_lenght, seq_length) where each
        #attention(i, j, :, :) is a seq_length x seq_length 2d tensor that is computed by 2d matrix multiplication of
        #query(i, j, :, :) and key(i, j, :, :). In particular, attention(i, j, k, l) refers the inner product similarity between kth and lth tokens
        #of ith sequnce corresponding to the jth head.


        # ** why normalization of attention_scores?
        #attention_scores is a tensor(batch_size, 12, seq_lenght, seq_lengh) where each entry represents the inner product similarity between tokens.
        #also, attentions_scores(i, j, k, :) is a row vector that contains the attention scores for kth token of ith sequence corresponding to jth
        #head over all the other tokens in ith sequence. After describing, what attention_scores contains, the next step is to discuss why normalization
        #of attention scores is required. The first reason is that each row vector attention_scores(i, j, k, :) is the logit attentions for kt token in
        #ith sequence for jth head, and it needs to be transformed into probability using softmax. Now, if one of the scores in the row vector
        #attention_scores(i, j, k, :) is much larger than the others, it will dominate the softmax probability and prevents the self-attention layer to
        #pay attention to other tokens. Normalization of the attention_scores by the square root of the size of internal emebedding of each head which
        #is 16, will distibute the probaility mass more evenly among tokens. So, we divide each attention_score by 8.

        # ** why normalization of attention_scores by the square root of the embedding size?
        #the main reason stems from the way that these attention scores are computed at the first place. Each single attention score is computed by inner
        #product of query vector and key vector. Therefore, you can imagine that the inner product of query and key vectors that are aligned, will scale
        #linearly by the dimension of those vectors. So, in order to ensure that the attention scores are invariant with respect to the dimensions of
        #those vectors, we need to divide them by the embedding size of queries and values, and not the square root of the embedding size of queries and
        #values.
        
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) #here, attention_head_size will be 64 so its square root will be 8

        # ** When attention mask is not required?
        #in below, we apply attention_mask that is supposed to tell us which tokens are actual tokens and which ones are PAD tokens. There are a number of
        #applications that such an attention_mask is not required. For example, when we train BERT as a lonaguage model on corpus of text where all the
        #sequences in a mini-batch have lenght of 512. However, for most of NLU tasks, we can assume that the sequences belonging to a sequence have
        #different lenghts and therefore such attention_mask is required.

        # ** How attention mask garauntees zero-weight for PAD tokens?
        #first, the tensor of attention_mask need to be the same dimension as attention_scores(batch_size, 12, seq_lenght, seq_length). Also, it has
        #to be -inf for all PAD tokens and zero for non-PAD tokens to ensure that the softmax weight for PAD tokens will be zero, and no impact for
        #non-PAD tokens. This means that if k is a PAD token in ith sequence of minibatch, then we need to have the following:
        #mask_attention(i, :, :, k) = -inf. That is the kth column of attention matrices across all heads need to be equal to -inf. 
        
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        #the final step to convert the attention_scores(batch_size, 12, seq_lenght, seq_lengh) to softmax vector by applyting softmax over the last
        #dimension as above. Therefore, attention_probs will be a tensor(batch_size, 12, seq_lenght, seq_lenght) and each row vector
        #tensor(i, j, k, :) will be a softmax vector.
        

        # ** How dropout is applied to attention_probs?
        #Below is the decription of HuggingFace about dropout being applied to attention_probs which is hard to understand 
        # {This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.}
        #This is what I think they meant: applying dropout as normal results in droping (zeroing out) one-by-one activations and not as a whole.
        #However, if you zero-out the weight corresponding to a token, it means that you will drop that token in that head completely since its
        #contributing weight is zero.
        
        attention_probs = self.dropout(attention_probs)

        # ** what is the difference between attention_mask and head_mask?
        #the main difference is that attention_mask is being applied before softmax layer which garauntees that the softmax probabilities are actually
        #representing a probability vector and they add to one, whereas head_mask is applied after softmax layer and therefore the softmax vector is not
        #a probability vector and it doesn't add up to one. The other difference between these two masks are their objectives. attention_mask is to
        #represent the PAD tokens via -inf entries, whereas head_mask's objective is not to represent the PAD tokens but probably a particular masking
        #structure that the architecture requires. head_mask will represent the tokens that it wants to mask by zero entries and no-change tokens by
        #one entries
        
        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
            #the above multiplication is point-wise multiplication and therefore head_mask should have the same dimension as the attention_probs which
            #is (batch_size, 12, seq_lenght, seq_lenght)

        # ** How to use attention_probs to compute weighted average output embeddings for the self-attention layer?
        #we know that attention_probs is a tensor(batch_size, 12, seq_lenght, seq_lenght) where attention_probs(i, j, k, :) is a row vector with the
        #lenght of the seq_lenght and represent the attention weights for the kth token of ith sequence for jth head. value_layer is a tensor
        #of size(batch_size, 12, seq_lenght, 64) where value_layer(i, j, k, :) dontes the value row vector with the lenght of 64 for the token
        #kth of ith sequence for jth head. Therefore, torch.matmul that becomes a batched-norm matrix multiplication since at least of these two
        #tensors has rank greater and equal to 3, results in a tensor of size(batch_size, 12, seq_lenght, 64) where context_layer(i, j, k, :)
        #is the ouput row vector with the lenght 64 corresponding to kth token of ith sequence for jth head.
        context_layer = torch.matmul(attention_probs, value_layer)

        #context_layer is a tensor(batch_size, 12, seq_lenght, 64). In order to generate the final output embeddings for the self-attention layer,
        #we need to conctatenate the 12 single-head vectors of size 64 to create a single vector of size 768. Therefore, the first step is swap
        #seq_lenght and head dimension to get a context_layer tensor of (batch_size, seq_lenght, 12, 64)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # ** why do we need to call contiguous after permute and transpose methods?
        #The operations like transpose, permute, view, exapand do not result in generating new tensors with the asked dimensions but only modifies
        #the meta information in Tensor object to provide the caller with the dimension that it asked for. To force torch to actually change the
        #underlying memory of tensor and not only the meta data of tensor, you need to call contiguous method over the tensor.


        #context_layer is a tensor(batch_size, seq_lenght, 12, 64). Here, we want to concatenate the 12 vectors of lenght 64. Therefore, the new shape
        #for the context_layer tensor must be (batch_size, seq_lenght, 768)
        
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        #the above reshape operation leads to context_layer have shape of (batch_size, seq_lenght, 768)

        #if we have asked this self-attention layer to output attention_probs as well as the ouput embeddings, then we return both the output
        #embeddings and attenion_probs
        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()

        # ** what is the main goal of BertSelfOutput?
        #this block is the second block of BertAttention and takes tensors of size(batch_size, seq_lenght, 768) and ouputs sensors of
        #size(batch_size, seq_lenght, 768). In particular, it applies linear matrix multiplication of size 768 x 768 to each row vector
        #of the input tensor(i, j, :). It does such a batched matrix multiplication using the internal batched matrix multiplication.
        #also, another functionality of this block is to implement the skip-connection from the input of the self-attention block to its
        #ouputs. 
        
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        #here, hidden_size is 768.
        
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        #BertLayerNorm is just nn.LayerNorm where passing config.hidden_size as the first argument of nn.LayerNorm forces that the empirical
        #means and variances are aggregated across only the last dimention of size of 768 of its input tensors. 
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #the hidden_dropout_prob is 0.1

    def forward(self, hidden_states, input_tensor):
        #here, hidden_states is the output embedding vectors generated by the self-attention block whereas input_tensor is the the input
        #embeddings to the self-attention block.
        
        hidden_states = self.dense(hidden_states)
        #here, hidden_states is a tensor(batch_size, seq_lenght, 768) and the ouput hidden_states is a tensor(batch_size, seq_lenght, 768)
        #self.dense is an instance of nn.Linear which encapsulates the weight tensor of size 768 x 768, and based on its internal nn.MatMul
        #will perform batched matrix multiplication which means that each row vector tensor(i, j, :) is multplied by the 768 x 768 weight matrix.
        
        hidden_states = self.dropout(hidden_states)
        #the above dropout layer applied dropout with probability of 0.1

        # ** Why we should apply LayerNorm after DropOut?
        #here, LayerNorm is supposed to compute statistics for each row output embedding vector of size(1, 768). Assume, that you apply LayerNorm
        #before DropOut, then there will be some cases that an activation is dropped from the output embedding vector of a token, but its value
        #is used to compute the statistics by LayerNorm. Therefore, you can see that if you apply LayerNorm before DropOut, its computed
        #statistics could potentially be invalied based on the dropped activations. 

        #here, we want to perform two actions: skip-connection and layer-normalization. The skip-connection part is implemented by adding
        #the input embedding to self-attention layer (input_tensor) by its ouput (hidden_states) before being passed to normalization layer.
        #the normalization layer which is nn.LayerNorm is instantiated here such that it aggregates the empirical mean and varaince across the last
        #dimension of size 768. In other words, LayerNorm statistics are being computed for each token separately. 
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()

        # ** what are the two main blocks of BertAttention?
        #the BertAttention layer consists of two main blocks: (1) the multi-head self attention layer that takes tensors of size
        #(btach_size, seq_lenght, 768) and outputs the tensors of size(batch_size, seq_lenght, 768) where each input token emebedding row vector
        #(1, 768) is mapped to a row ouput embedding vector of size(1, 768). The issue with this output embedding vector of size(1, 768) is that
        #it is composed by concatenation of ouput embeddings of size 64 ot 12 heads. (2) the second block in BertAttention aims at adressing
        #the issue of the first block with the main objective of resolving this alliasing issue. This second block applies a single layer of neural
        #network with input size of 768 and the output size of 768, which is applied to each token output embedding vector separately. This second
        #block is called BertSelfOutput. BertSelfOutput, after applying this batched matrix multiplication of 768 x 768, it does the following steps
        #as well in this order: (i) droput (ii) skip-connection (iii) LayerNorm
        
        self.self = BertSelfAttention(config)
        
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and remove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        
        self_outputs = self.self(hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)

        #is_decoder is False for bert-base-uncased which makes sence since we want to used BERT as encoder
        self.is_decoder = config.is_decoder
        
        if self.is_decoder: 
            self.crossattention = BertAttention(config)
            
        self.intermediate = BertIntermediate(config)
        
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask)
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()

        #for bert-base-uncased, output_attentions is False
        self.output_attentions = config.output_attentions

        #for bert-base-uncased, output_hidden_states is False
        self.output_hidden_states = config.output_hidden_states


        #here, num_hidden_layers is 12
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        #note: nn.ModuleList is different from nn.Sequential. In particular, nn.ModuleList is like a python list and doesn't
        #provide any additional functionality. This means that inside the forward method of BertEncoder, you need to manually
        #glue them together by passing the ouput of ith one as the input of (i+1)th layer. However, nn.Sequential does the glueing
        #for you, which means that you can invoke the module returned by nn.Sequential as a single unified module

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size,
                                 config.vocab_size,
                                 bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


BERT_START_DOCSTRING = r"""    The BERT model was proposed in
    `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`_
    by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. It's a bidirectional transformer
    pre-trained using a combination of masked language modeling objective and next sentence prediction
    on a large corpus comprising the Toronto Book Corpus and Wikipedia.

    This model is a PyTorch `torch.nn.Module`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matter related to general usage and behavior.

    .. _`BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`:
        https://arxiv.org/abs/1810.04805

    .. _`torch.nn.Module`:
        https://pytorch.org/docs/stable/nn.html#module

    Parameters:
        config (:class:`~transformers.BertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Inputs:
        **input_ids**: ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of input sequence tokens in the vocabulary.
            To match pre-training, BERT input sequence should be formatted with [CLS] and [SEP] tokens as follows:

            (a) For sequence pairs:

                ``tokens:         [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]``

                ``token_type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1``

            (b) For single sequences:

                ``tokens:         [CLS] the dog is hairy . [SEP]``

                ``token_type_ids:   0   0   0   0  0     0   0``

            Bert is a model with absolute position embeddings so it's usually advised to pad the inputs on
            the right rather than the left.

            Indices can be obtained using :class:`transformers.BertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.convert_tokens_to_ids` for details.
        **attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        **token_type_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token
            (see `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding`_ for more details).
        **position_ids**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.
        **head_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``:
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            ``1`` indicates the head is **not masked**, ``0`` indicates the head is **masked**.
        **inputs_embeds**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, embedding_dim)``:
            Optionally, instead of passing ``input_ids`` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        **encoder_hidden_states**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``:
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if the model
            is configured as a decoder.
        **encoder_attention_mask**: (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
"""

@add_start_docstrings("The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
                      BERT_START_DOCSTRING, BERT_INPUTS_DOCSTRING)
class BertModel(BertPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        #BertPreTrainedModel is derived from PreTrainedModel and the __ini__ method of PreTrainedModel will store the config
        #object of this model as its config field member
        self.config = config

        self.embeddings = BertEmbeddings(config)
        #BertEmbeddings is directly derived from nn.Module
        #the self.embeddings nn.module here is a torch neural network that encapsulates the embeddings for the tokens in vocab
        #as well as 512 positional embeddings. For each input tensor of minibatch of sequences [batch_size, seq_length] which
        #contains the vocab indices of input sequences, it will return a tensor embedding of
        #size(batch_size, seq_lenght, embedding_size)
        
        self.encoder = BertEncoder(config)
        #BertEncoder is also derived from nn.Module
        
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None):
        """ Forward pass on the Model.

        The model can behave as an encoder (with only self-attention) as well
        as a decoder, in which case a layer of cross-attention is added between
        the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
        Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

        To behave as an decoder the model needs to be initialized with the
        `is_decoder` argument of the configuration set to `True`; an
        `encoder_hidden_states` is expected as an input to the forward pass.

        .. _`Attention is all you need`:
            https://arxiv.org/abs/1706.03762

        """
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]

        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if attention_mask.dim() == 2:
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        encoder_outputs = self.encoder(embedding_output,
                                       attention_mask=extended_attention_mask,
                                       head_mask=head_mask,
                                       encoder_hidden_states=encoder_hidden_states,
                                       encoder_attention_mask=encoder_extended_attention_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model with two heads on top as done during the pre-training:
                       a `masked language modeling` head and a `next sentence prediction (classification)` head. """,
                      BERT_START_DOCSTRING,
                      BERT_INPUTS_DOCSTRING)
class BertForPreTraining(BertPreTrainedModel):
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        **next_sentence_label**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see ``input_ids`` docstring)
            Indices should be in ``[0, 1]``.
            ``0`` indicates sequence B is a continuation of sequence A,
            ``1`` indicates sequence B is a random sequence.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when both ``masked_lm_labels`` and ``next_sentence_label`` are provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total loss as the sum of the masked language modeling loss and the next sequence prediction (classification) loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **seq_relationship_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, 2)``
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForPreTraining.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        prediction_scores, seq_relationship_scores = outputs[:2]

    """
    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,
                masked_lm_labels=None, next_sentence_label=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        outputs = (prediction_scores, seq_relationship_score,) + outputs[2:]  # add hidden states and attention if they are here

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            outputs = (total_loss,) + outputs

        return outputs  # (loss), prediction_scores, seq_relationship_score, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model with a `language modeling` head on top. """,
                      BERT_START_DOCSTRING,
                      BERT_INPUTS_DOCSTRING)
class BertForMaskedLM(BertPreTrainedModel):
    r"""
        **masked_lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        **lm_labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the left-to-right language modeling loss (next word prediction).
            Indices should be in ``[-1, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-1`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **masked_lm_loss**: (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        **ltr_lm_loss**: (`optional`, returned when ``lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Next token prediction loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, masked_lm_labels=input_ids)
        loss, prediction_scores = outputs[:2]

    """
    def __init__(self, config):
        super(BertForMaskedLM, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyMLMHead(config)

        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,
                masked_lm_labels=None, encoder_hidden_states=None, encoder_attention_mask=None, lm_labels=None, ):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds,
                            encoder_hidden_states=encoder_hidden_states,
                            encoder_attention_mask=encoder_attention_mask)

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        # Although this may seem awkward, BertForMaskedLM supports two scenarios:
        # 1. If a tensor that contains the indices of masked labels is provided,
        #    the cross-entropy is the MLM cross-entropy that measures the likelihood
        #    of predictions for masked words.
        # 2. If `lm_labels` is provided we are in a causal scenario where we
        #    try to predict the next token for each input in the decoder.
        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)  # -1 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        if lm_labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            lm_labels = lm_labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), lm_labels.view(-1))
            outputs = (ltr_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), (ltr_lm_loss), prediction_scores, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model with a `next sentence prediction (classification)` head on top. """,
                      BERT_START_DOCSTRING,
                      BERT_INPUTS_DOCSTRING)
class BertForNextSentencePrediction(BertPreTrainedModel):
    r"""
        **next_sentence_label**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see ``input_ids`` docstring)
            Indices should be in ``[0, 1]``.
            ``0`` indicates sequence B is a continuation of sequence A,
            ``1`` indicates sequence B is a random sequence.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``next_sentence_label`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Next sequence prediction (classification) loss.
        **seq_relationship_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, 2)``
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        seq_relationship_scores = outputs[0]

    """
    def __init__(self, config):
        super(BertForNextSentencePrediction, self).__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,
                next_sentence_label=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        seq_relationship_score = self.cls(pooled_output)

        outputs = (seq_relationship_score,) + outputs[2:]  # add hidden states and attention if they are here
        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            outputs = (next_sentence_loss,) + outputs

        return outputs  # (next_sentence_loss), seq_relationship_score, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of
                      the pooled output) e.g. for GLUE tasks. """,
                      BERT_START_DOCSTRING,
                      BERT_INPUTS_DOCSTRING)
class BertForSequenceClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

    """
    def __init__(self, config):
        #for bert-base-uncased, the config is the following
        #{
        #"attention_probs_dropout_prob": 0.1,
        #"finetuning_task": "mrpc",
        #"hidden_act": "gelu",
        #"hidden_dropout_prob": 0.1,
        #"hidden_size": 768,
        #"initializer_range": 0.02,
        #"intermediate_size": 3072,
        #"is_decoder": false,
        #"layer_norm_eps": 1e-12,
        #"max_position_embeddings": 512,
        #"num_attention_heads": 12,
        #"num_hidden_layers": 12,
        #"num_labels": 2,
        #"output_attentions": false,
        #"output_hidden_states": false,
        #"output_past": true,
        #"pruned_heads": {},
        #"torchscript": false,
        #"type_vocab_size": 2,
        #"use_bfloat16": false,
        #"vocab_size": 30522
        #}
        super(BertForSequenceClassification, self).__init__(config)
        #the above super init method hits the init method of PreTrainedModel inside modeling_utils.py that inherents from
        #nn.Module. PreTrainedModel stores the config object of this model as its config member field.
        #BertForSequenceClassification is derived from BertPreTrainedModel and BertPreTrainedModel is derived from
        #PreTrainedModel. BertPreTrainedModel only has one method called _init_weights and it doesn't have __init__ method
        self.num_labels = config.num_labels #2

        self.bert = BertModel(config)
        #BertModel also is derived from BertPreTrainedModel. This is kinda weird since both this class
        #BertForSequenceClassification and its member field BertModel are derived from BertPreTrained. It seems kinda
        #redundant.

        #in particular, BertModel is the bare bert model transformer without any head for either sequence classification or
        #question answering. In other words, BertModel only ouputs the hidden states. 
        
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model with a multiple choice classification head on top (a linear layer on top of
                      the pooled output and a softmax) e.g. for RocStories/SWAG tasks. """,
                      BERT_START_DOCSTRING,
                      BERT_INPUTS_DOCSTRING)
class BertForMultipleChoice(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **classification_scores**: ``torch.FloatTensor`` of shape ``(batch_size, num_choices)`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above).
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForMultipleChoice.from_pretrained('bert-base-uncased')
        choices = ["Hello, my dog is cute", "Hello, my cat is amazing"]
        input_ids = torch.tensor([tokenizer.encode(s, add_special_tokens=True) for s in choices]).unsqueeze(0)  # Batch size 1, 2 choices
        labels = torch.tensor(1).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, classification_scores = outputs[:2]

    """
    def __init__(self, config):
        super(BertForMultipleChoice, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):
        num_choices = input_ids.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model with a token classification head on top (a linear layer on top of
                      the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. """,
                      BERT_START_DOCSTRING,
                      BERT_INPUTS_DOCSTRING)
class BertForTokenClassification(BertPreTrainedModel):
    r"""
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification loss.
        **scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.num_labels)``
            Classification scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForTokenClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1] * input_ids.size(1)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, scores = outputs[:2]

    """
    def __init__(self, config):
        super(BertForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), scores, (hidden_states), (attentions)


@add_start_docstrings("""Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of
                      the hidden-states output to compute `span start logits` and `span end logits`). """,
                      BERT_START_DOCSTRING,
                      BERT_INPUTS_DOCSTRING)
class BertForQuestionAnswering(BertPreTrainedModel):
    r"""
        **start_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        **end_positions**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        **start_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-start scores (before SoftMax).
        **end_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length,)``
            Span-end scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
        question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
        input_text = "[CLS] " + question + " [SEP] " + text + " [SEP]"
        input_ids = tokenizer.encode(input_text)
        token_type_ids = [0 if i <= input_ids.index(102) else 1 for i in range(len(input_ids))] 
        start_scores, end_scores = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([token_type_ids]))
        all_tokens = tokenizer.convert_ids_to_tokens(input_ids)  
        print(' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1]))
        # a nice puppet


    """
    def __init__(self, config):
        super(BertForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,
                start_positions=None, end_positions=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
