import torch
from transformers import *

# Transformers has a unified API
# for 8 transformer architectures and 30 pretrained weights.
#          Model          | Tokenizer          | Pretrained weights shortcut
MODELS = [(BertModel,       BertTokenizer,       'bert-base-uncased'),
          (OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt'),
          (GPT2Model,       GPT2Tokenizer,       'gpt2'),
          (CTRLModel,       CTRLTokenizer,       'ctrl'),
          (TransfoXLModel,  TransfoXLTokenizer,  'transfo-xl-wt103'),
          (XLNetModel,      XLNetTokenizer,      'xlnet-base-cased'),
          (XLMModel,        XLMTokenizer,        'xlm-mlm-enfr-1024'),
          (DistilBertModel, DistilBertTokenizer, 'distilbert-base-uncased'),
          (RobertaModel,    RobertaTokenizer,    'roberta-base')]

MODELS = [(OpenAIGPTModel,  OpenAIGPTTokenizer,  'openai-gpt'),]

# To use TensorFlow 2.0 versions of the models, simply prefix the class names with 'TF', e.g. `TFRobertaModel` is the TF 2.0 counterpart of the PyTorch model `RobertaModel`

# Let's encode some text in a sequence of hidden-states using each model:
for model_class, tokenizer_class, pretrained_weights in MODELS:
    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    #OpenAIGPTTokenier is a BytePair encoding
    model = model_class.from_pretrained(pretrained_weights)

    # Encode text
    input_ids = torch.tensor([tokenizer.encode("Here is some text to encode", add_special_tokens=True)])  # Add special tokens takes care of adding [CLS], [SEP], <s>... tokens in the right way for each model.

    #tokenizer.encode returns a python list, therefore, you need to transform it to a torch tensor using torch.tensor
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
        #in above, last_hidden_states is the output of the last self-attention block which will be a tensor(input_lenght, embedding_size)
        #in particular, model(input_ids) here will return a tuple with size of 1 that is the tenosr corresponding to the output of the last
        #self-attention block. However, you can ask the model to ouput not only the output of the last self-attention block but the outputs of all the
        #slef-attention blocks. You can ask the model to do such by passing output_hidden_states = True while instantiating the model as follows:
        #model = OpenAIGPTModel.from_pretrained("openai-gpt", ouput_hidden_states = True). Doing so, if you execute result = model(input_ids),
        #then result[0] will be the torch tensor(input_lenght, 768)  ouput of the last self-attention block and result[1] will be a tuple of lenght 13,
        #where each element of this tuple is a torch tensor(input_lenght, 768). In particular, the first tensor is corresponding to the embedding layer
        #and the rest 12 tensors are corresponding to the 12 self-attention blocks. Also, you have the option to ask the model to ouput its attentions.
        #You can ask the model to do a such as follows: model = OpenAIGPTModel.from_pretrain("openai-gpt", output_hidden_states = True,
        #ouput_attentions = True). Then, the result will be a tuple with 3 elemnts where the first 2 are same as above and the third element itself
        #will be a python tuple with 12 torch tensors correspondings to the 12 self-attention blocks. Each of this tensor will be a
        #tensor(1, 12, input_length, input_lenght) where 12 here refers to 12 head of the self-attention blocks and tensor(0, i, :, :) refers to
        #the inner-product similarity based attention matrix corresponding to the ith head of this block.
        
        print(last_hidden_states.size())


# Each architecture is provided with several class for fine-tuning on down-stream tasks, e.g.
BERT_MODEL_CLASSES = [BertModel, BertForPreTraining, BertForMaskedLM, BertForNextSentencePrediction,
                      BertForSequenceClassification, BertForTokenClassification, BertForQuestionAnswering]

# All the classes for an architecture can be initiated from pretrained weights for this architecture
# Note that additional weights added for fine-tuning are only initialized
# and need to be trained on the down-stream task
pretrained_weights = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
for model_class in BERT_MODEL_CLASSES:
    # Load pretrained model/tokenizer
    model = model_class.from_pretrained(pretrained_weights)

    # Models can return full list of hidden-states & attentions weights at each layer
    model = model_class.from_pretrained(pretrained_weights,
                                        output_hidden_states=True,
                                        output_attentions=True)
    input_ids = torch.tensor([tokenizer.encode("Let's see all hidden-states and attentions on this text")])
    all_hidden_states, all_attentions = model(input_ids)[-2:]

    # Models are compatible with Torchscript
    model = model_class.from_pretrained(pretrained_weights, torchscript=True)
    traced_model = torch.jit.trace(model, (input_ids,))

    # Simple serialization for models and tokenizers
    model.save_pretrained('./directory/to/save/')  # save
    model = model_class.from_pretrained('./directory/to/save/')  # re-load
    tokenizer.save_pretrained('./directory/to/save/')  # save
    tokenizer = BertTokenizer.from_pretrained('./directory/to/save/')  # re-load

    # SOTA examples for GLUE, SQUAD, text generation...
