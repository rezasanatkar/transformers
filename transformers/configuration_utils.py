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
""" Configuration base class and utilities."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import copy
import json
import logging
import os
from io import open

from .file_utils import cached_path, CONFIG_NAME

logger = logging.getLogger(__name__)

class PretrainedConfig(object):
    r""" Base class for all configuration classes.
        Handles a few parameters common to all models' configurations as well as methods for loading/downloading/saving configurations.

        Note:
            A configuration file can be loaded and saved to disk. Loading the configuration file and using this file to initialize a model does **not** load the model weights.
            It only affects the model's configuration.

        Class attributes (overridden by derived classes):
            - ``pretrained_config_archive_map``: a python ``dict`` of with `short-cut-names` (string) as keys and `url` (string) of associated pretrained model configurations as values.

        Parameters:
            ``finetuning_task``: string, default `None`. Name of the task used to fine-tune the model. This can be used when converting from an original (TensorFlow or PyTorch) checkpoint.
            ``num_labels``: integer, default `2`. Number of classes to use when the model is a classification model (sequences/tokens)
            ``output_attentions``: boolean, default `False`. Should the model returns attentions weights.
            ``output_hidden_states``: string, default `False`. Should the model returns all hidden-states.
            ``torchscript``: string, default `False`. Is the model used with Torchscript.
    """
    pretrained_config_archive_map = {}

    def __init__(self, **kwargs):
        self.finetuning_task = kwargs.pop('finetuning_task', None)
        self.num_labels = kwargs.pop('num_labels', 2)
        self.output_attentions = kwargs.pop('output_attentions', False)
        self.output_hidden_states = kwargs.pop('output_hidden_states', False)
        self.output_past = kwargs.pop('output_past', True)  # Not used by all models
        self.torchscript = kwargs.pop('torchscript', False)  # Only used by PyTorch models
        self.use_bfloat16 = kwargs.pop('use_bfloat16', False)
        self.pruned_heads = kwargs.pop('pruned_heads', {})
        self.is_decoder = kwargs.pop('is_decoder', False)

    def save_pretrained(self, save_directory):
        """ Save a configuration object to the directory `save_directory`, so that it
            can be re-loaded using the :func:`~transformers.PretrainedConfig.from_pretrained` class method.
        """
        assert os.path.isdir(save_directory), "Saving path should be a directory where the model and configuration can be saved"

        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, CONFIG_NAME)
        #inside file_utils.py, CONFIG_NAME is set to be config.json

        self.to_json_file(output_config_file)
        logger.info("Configuration saved in {}".format(output_config_file))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r""" Instantiate a :class:`~transformers.PretrainedConfig` (or a derived class) from a pre-trained model configuration.

        Parameters:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model configuration to load from cache or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing a configuration file saved using the :func:`~transformers.PretrainedConfig.save_pretrained` method, e.g.: ``./my_model_directory/``.
                - a path or url to a saved configuration JSON `file`, e.g.: ``./my_model_directory/configuration.json``.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            kwargs: (`optional`) dict: key/value pairs with which to update the configuration object after loading.

                - The values in kwargs of any keys which are configuration attributes will be used to override the loaded values.
                - Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled by the `return_unused_kwargs` keyword parameter.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override the cached versions if they exists.

            resume_download: (`optional`) boolean, default False:
                Do not delete incompletely recieved file. Attempt to resume the download if such a file exists.

            proxies: (`optional`) dict, default None:
                A dictionary of proxy servers to use by protocol or endpoint, e.g.: {'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}.
                The proxies are used on each request.

            return_unused_kwargs: (`optional`) bool:

                - If False, then this function returns just the final configuration object.
                - If True, then this functions returns a tuple `(config, unused_kwargs)` where `unused_kwargs` is a dictionary consisting of the key/value pairs whose keys are not configuration attributes: ie the part of kwargs which has not been used to update `config` and is otherwise ignored.

        Examples::

            # We can't instantiate directly the base class `PretrainedConfig` so let's show the examples on a
            # derived class: BertConfig
            config = BertConfig.from_pretrained('bert-base-uncased')    # Download configuration from S3 and cache.
            config = BertConfig.from_pretrained('./test/saved_model/')  # E.g. config (or model) was saved using `save_pretrained('./test/saved_model/')`
            config = BertConfig.from_pretrained('./test/saved_model/my_configuration.json')
            config = BertConfig.from_pretrained('bert-base-uncased', output_attention=True, foo=False)
            assert config.output_attention == True
            config, unused_kwargs = BertConfig.from_pretrained('bert-base-uncased', output_attention=True,
                                                               foo=False, return_unused_kwargs=True)
            assert config.output_attention == True
            assert unused_kwargs == {'foo': False}

        """
        cache_dir = kwargs.pop('cache_dir', None)
        #you need to specify cache_dir if you want to force this module to not cache the downloaded config file into the default cache path but a different
        #path. The default cache path is /Users/msanatkar/.cache/torch/transformers
        
        force_download = kwargs.pop('force_download', False)
        #it forces to redownload the weights and configuration file of a model even if they already exists in cache
        
        resume_download = kwargs.pop('resume_download', False)
        
        proxies = kwargs.pop('proxies', None)
        #if you need proxies to get connected to the Internet
        
        return_unused_kwargs = kwargs.pop('return_unused_kwargs', False)

        #in the case of bert-based-uncased, pretrained_model_name_or_path is equal to bert-base-uncased. 
        if pretrained_model_name_or_path in cls.pretrained_config_archive_map:
            #cls.pretrained_config_archive_map is specified by the derived classes and it is a python dictionary that maps the existing model names like
            #bert-base-uncased to the url path of their corresponding config files. So, therefore, here, we want to check if the passed string to
            #this method as the pretrained_model_name_or_path is actually a model name like bert-base-uncased or it is a path to a potentially a local
            #json config file
            
            config_file = cls.pretrained_config_archive_map[pretrained_model_name_or_path]
            #here, the above config_file will be a url path to a file on Amazon S3
            
        elif os.path.isdir(pretrained_model_name_or_path):
            #here, we check if passed string is in fact, a path to a directory that contains the local config file
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
            #in file_utils.py, CONFIG_NAME is set to be config.json
            
        else:
            config_file = pretrained_model_name_or_path
            #here, the assumption is that the pretrained_model_name_or_path is actually the path to file config file

            
        # redirect to the cache, if necessary
        try:
            resolved_config_file = cached_path(config_file, cache_dir=cache_dir, force_download=force_download,
                                               proxies=proxies, resume_download=resume_download)
            #in above, we ususally want to use the default cache path. Therefore, we don't specify the cache_dir
            #the method cahced_path is defined at file_utils.py
            #for most of the pretrained encoder models, the config_file is a url path to a json config file in Amazon s3 that contains the architecture
            #information of the pretrained model like BERT

            #the method cache_path will download the json config file if it doen't exist already in cache folder or if explicitly ask force_download
            #at the end this method return the path to the downloaded file inside the cache folder
        except EnvironmentError:
            if pretrained_model_name_or_path in cls.pretrained_config_archive_map:
                msg = "Couldn't reach server at '{}' to download pretrained model configuration file.".format(
                        config_file)
            else:
                msg = "Model name '{}' was not found in model name list ({}). " \
                      "We assumed '{}' was a path or url to a configuration file named {} or " \
                      "a directory containing such a file but couldn't find any such file at this path or url.".format(
                        pretrained_model_name_or_path,
                        ', '.join(cls.pretrained_config_archive_map.keys()),
                        config_file, CONFIG_NAME)
            raise EnvironmentError(msg)

        if resolved_config_file == config_file:
            logger.info("loading configuration file {}".format(config_file))
        else:
            #config_file will have a different name compared to resolved_config_file
            logger.info("loading configuration file {} from cache at {}".format(
                config_file, resolved_config_file))

        # Load config
        config = cls.from_json_file(resolved_config_file)
        #the above class method will create an object of PreTrainedConfig from the json file. For bert-base-uncased, the config object is the following
        #{
        #"attention_probs_dropout_prob": 0.1,
        #"finetuning_task": null,
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

        if hasattr(config, 'pruned_heads'):
            config.pruned_heads = dict((int(key), value) for key, value in config.pruned_heads.items())
        # Update config with kwargs if needed
        to_remove = []
        #here, we have the option to override the configuaration attributes of the config object using the kwargs passed to from_pretrained method.
        for key, value in kwargs.items():
            setattr(config, key, value)
            #overriding happens here
            to_remove.append(key)

        #in above in the case of bert-base-uncased being used for mrpc task, the only attribute of the config that is overriden is finetunning_taks that
        #is changed from null to mrpc

        for key in to_remove:
            kwargs.pop(key, None)

        logger.info("Model config %s", str(config))
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `Config` from a Python dictionary of parameters."""
        config = cls(vocab_size_or_config_json_file=-1)
        #the above cls will create an object of this class => PretrainedConfig
        
        for key, value in json_object.items():
            setattr(config, key, value)
            #setattr will set the attributes of the config object where key are strings representing the names of attributes and value
            #will be their values
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())
