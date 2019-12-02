"""
Utilities for working with the local dataset cache.
This file is adapted from the AllenNLP library at https://github.com/allenai/allennlp
Copyright by the AllenNLP authors.
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

import sys
import json
import logging
import os
import six
import shutil
import tempfile
import fnmatch
from functools import wraps
from hashlib import sha256
from io import open

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import requests
from tqdm import tqdm
from contextlib import contextmanager

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

try:
    import tensorflow as tf
    assert hasattr(tf, '__version__') and int(tf.__version__[0]) >= 2
    _tf_available = True  # pylint: disable=invalid-name
    logger.info("TensorFlow version {} available.".format(tf.__version__))
except (ImportError, AssertionError):
    _tf_available = False  # pylint: disable=invalid-name

try:
    import torch
    _torch_available = True  # pylint: disable=invalid-name
    logger.info("PyTorch version {} available.".format(torch.__version__))
except ImportError:
    _torch_available = False  # pylint: disable=invalid-name


try:
    from torch.hub import _get_torch_home
    torch_cache_home = _get_torch_home()
    # torche_cache_home is equal to /Users/msanatkar/.cache/torch
except ImportError:
    torch_cache_home = os.path.expanduser(
        os.getenv('TORCH_HOME', os.path.join(
            os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')))
default_cache_path = os.path.join(torch_cache_home, 'transformers')
#default_cache_path is equal to /Users/msanatkar/.cache/torch/transformers

try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse

try:
    from pathlib import Path
    PYTORCH_PRETRAINED_BERT_CACHE = Path(
        os.getenv('PYTORCH_TRANSFORMERS_CACHE', os.getenv('PYTORCH_PRETRAINED_BERT_CACHE', default_cache_path)))
    #default_cache_path is /Users/msanatkar/.cache/torch/transformers which will be PYTORCH_PRETRAINED_BERT_CACHE as well since the
    #environemnt variables PYTORCH_TRANSFORMERS_CACHE and PYTORCH_PRETRAINED_BERT_CACH do not exist

except (AttributeError, ImportError):
    PYTORCH_PRETRAINED_BERT_CACHE = os.getenv('PYTORCH_TRANSFORMERS_CACHE',
                                              os.getenv('PYTORCH_PRETRAINED_BERT_CACHE',
                                                        default_cache_path))


PYTORCH_TRANSFORMERS_CACHE = PYTORCH_PRETRAINED_BERT_CACHE  # Kept for backward compatibility
TRANSFORMERS_CACHE = PYTORCH_PRETRAINED_BERT_CACHE  # Kept for backward compatibility

WEIGHTS_NAME = "pytorch_model.bin"
TF2_WEIGHTS_NAME = 'tf_model.h5'
TF_WEIGHTS_NAME = 'model.ckpt'
CONFIG_NAME = "config.json"

def is_torch_available():
    return _torch_available

def is_tf_available():
    return _tf_available

if not six.PY2:
    def add_start_docstrings(*docstr):
        def docstring_decorator(fn):
            fn.__doc__ = ''.join(docstr) + fn.__doc__
            return fn
        return docstring_decorator

    def add_end_docstrings(*docstr):
        def docstring_decorator(fn):
            fn.__doc__ = fn.__doc__ + ''.join(docstr)
            return fn
        return docstring_decorator
else:
    # Not possible to update class docstrings on python2
    def add_start_docstrings(*docstr):
        def docstring_decorator(fn):
            return fn
        return docstring_decorator

    def add_end_docstrings(*docstr):
        def docstring_decorator(fn):
            return fn
        return docstring_decorator

def url_to_filename(url, etag=None):
    #url for bert-base-uncased is https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json
    #and its etag that represents the version of this resource is the following:
    #74d4f96fdabdd865cbdbe905cd46c1f1
    
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.
    If the url ends with .h5 (Keras HDF5 weights) ands '.h5' to the name
    so that TF 2.0 can identify it as a HDF5 file
    (see https://github.com/tensorflow/tensorflow/blob/00fad90125b18b80fe054de1055770cfb8fe4ba3/tensorflow/python/keras/engine/network.py#L1380)
    """
    #url is a string
    url_bytes = url.encode('utf-8')
    #.encode('utf-8') ecnode the url string into utf-8 encoding where the type of url_bytes is bytes and not str anymore
    url_hash = sha256(url_bytes)
    #url_hash will be a hash object and .hexdigest() returns a str that represent the str of the hashcode in hexadecimal
    filename = url_hash.hexdigest()

    if etag:
        #etag which is the veriosn of this model json config file from s2 Amazon for bert-base-uncased is the following:
        #74d4f96fdabdd865cbdbe905cd46c1f1
        etag_bytes = etag.encode('utf-8')
        etag_hash = sha256(etag_bytes)
        filename += '.' + etag_hash.hexdigest()
        #wow, interesting. filename will be the hashcode corresponding to the url path of the json config file appended by the hashcode of the etag version
        #of the json config file

    if url.endswith('.h5'):
        filename += '.h5'
        #for bert-base-uncased, it doesn't

    return filename


def filename_to_url(filename, cache_dir=None):
    """
    Return the url and etag (which may be ``None``) stored for `filename`.
    Raise ``EnvironmentError`` if `filename` or its stored metadata do not exist.
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    cache_path = os.path.join(cache_dir, filename)
    if not os.path.exists(cache_path):
        raise EnvironmentError("file {} not found".format(cache_path))

    meta_path = cache_path + '.json'
    if not os.path.exists(meta_path):
        raise EnvironmentError("file {} not found".format(meta_path))

    with open(meta_path, encoding="utf-8") as meta_file:
        metadata = json.load(meta_file)
    url = metadata['url']
    etag = metadata['etag']

    return url, etag


def cached_path(url_or_filename, cache_dir=None, force_download=False, proxies=None, resume_download=False):
    #for most of the pretrained model like BERT url_or_filename is a url path to the json config file of the pretrained model that contains the
    #architectural information about the model. cache_dir is not provided so that this method will back off to the default cache path which is equal to
    #/Users/msanatkar/.cache/torch/transformers.
    #this method will download the json config file if it doesn't exist or if we enable force_download and finally it will return the path to the
    #downloaded file inside the cache folder
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    Args:
        cache_dir: specify a cache directory to save the file to (overwrite the default cache dir).
        force_download: if True, re-dowload the file even if it's already cached in the cache dir.
        resume_download: if True, resume the download if incompletly recieved file is found.
    """
    #for most casess of pretrained models, cache_dir is None
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
        #cache_dir is equal to /Users/msanatkar/.cache/torch/transformers
    if sys.version_info[0] == 3 and isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
        #this one is not satisfied since url_or_filename for existing pretrained models is a url path and not a Path
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        #sys.version_info returns the Python version
        cache_dir = str(cache_dir)

    #for bert-base-uncased, url_or_filename = https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json
    parsed = urlparse(url_or_filename)
    #the parsed verion of the above url path is the following:
    #ParseResult(scheme='https', netloc='s3.amazonaws.com', path='/models.huggingface.co/bert/bert-base-uncased-config.json', params='', query='',
    #fragment='')

    if parsed.scheme in ('http', 'https', 's3'):
        # URL, so get it from the cache (downloading if necessary)

        #for most of the pretrained encoders, url_or_filename is a url path and they will hit this if condition
        #get_from_cache download the json config file if it doen't exist or force_download option in enabled. After downloading the file, this method returns
        #the path to the file inside cache folder
        return get_from_cache(url_or_filename, cache_dir=cache_dir,
            force_download=force_download, proxies=proxies,
            resume_download=resume_download)
    
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        return url_or_filename
    elif parsed.scheme == '':
        # File, but it doesn't exist.
        raise EnvironmentError("file {} not found".format(url_or_filename))
    else:
        # Something unknown
        raise ValueError("unable to parse {} as a URL or as a local path".format(url_or_filename))


def split_s3_path(url):
    """Split a full s3 path into the bucket name and path."""
    parsed = urlparse(url)
    if not parsed.netloc or not parsed.path:
        raise ValueError("bad s3 path {}".format(url))
    bucket_name = parsed.netloc
    s3_path = parsed.path
    # Remove '/' at beginning of path.
    if s3_path.startswith("/"):
        s3_path = s3_path[1:]
    return bucket_name, s3_path


def s3_request(func):
    """
    Wrapper function for s3 requests in order to create more helpful error
    messages.
    """

    @wraps(func)
    def wrapper(url, *args, **kwargs):
        try:
            return func(url, *args, **kwargs)
        except ClientError as exc:
            if int(exc.response["Error"]["Code"]) == 404:
                raise EnvironmentError("file {} not found".format(url))
            else:
                raise

    return wrapper


@s3_request
def s3_etag(url, proxies=None):
    """Check ETag on S3 object."""
    s3_resource = boto3.resource("s3", config=Config(proxies=proxies))
    bucket_name, s3_path = split_s3_path(url)
    s3_object = s3_resource.Object(bucket_name, s3_path)
    return s3_object.e_tag


@s3_request
def s3_get(url, temp_file, proxies=None):
    """Pull a file directly from S3."""
    s3_resource = boto3.resource("s3", config=Config(proxies=proxies))
    bucket_name, s3_path = split_s3_path(url)
    s3_resource.Bucket(bucket_name).download_fileobj(s3_path, temp_file)


def http_get(url, temp_file, proxies=None, resume_size=0):
    headers={'Range':'bytes=%d-'%(resume_size,)} if resume_size > 0 else None
    response = requests.get(url, stream=True, proxies=proxies, headers=headers)
    if response.status_code == 416:  # Range not satisfiable
        return
    content_length = response.headers.get('Content-Length')
    total = resume_size + int(content_length) if content_length is not None else None

    #tqdm is a progress bar
    progress = tqdm(unit="B", total=total, initial=resume_size)
    for chunk in response.iter_content(chunk_size=1024):
        if chunk: # filter out keep-alive new chunks
            progress.update(len(chunk))
            temp_file.write(chunk)
    progress.close()


def get_from_cache(url, cache_dir=None, force_download=False, proxies=None, etag_timeout=10, resume_download=False):
    #for bert-based-uncased, url is https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json
    #also, cache_dir is the following: /Users/msanatkar/.cache/torch/transformers
    
    """
    Given a URL, look for the corresponding dataset in the local cache.
    If it's not there, download it. Then return the path to the cached file.
    """
    if cache_dir is None:
        cache_dir = TRANSFORMERS_CACHE
        #TRANSFORMERS_CACHE is equal to /Users/msanatkar/.cache/torch/transformers
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)
        #sys.version_info[0] returns the Python version
    if sys.version_info[0] == 2 and not isinstance(cache_dir, str):
        cache_dir = str(cache_dir)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    #cache_dir is equal to /Users/msanatkar/.cache/torch/transformers

    # Get eTag to add to filename, if it exists.
    if url.startswith("s3://"):
        #url for BERT starts with https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json so it doesn't satisfy this of condition
        etag = s3_etag(url, proxies=proxies)
    else:
        try:
            response = requests.head(url, allow_redirects=True, proxies=proxies, timeout=etag_timeout)
            #head method make a head request to a webpage and returns the HTTP header
            if response.status_code != 200:
                #response code 200 refers to an OK response and no error
                etag = None
            else:
                etag = response.headers.get("ETag")
                #The ETag HTTP response header is an identifier for a specific version of a resource. It lets caches be more efficient and save bandwidth,
                #as a web server does not need to resend a full response if the content has not changed
                #ETage for bert-base-uncased is 74d4f96fdabdd865cbdbe905cd46c1f1
        except (EnvironmentError, requests.exceptions.Timeout):
            etag = None

    if sys.version_info[0] == 2 and etag is not None:
        etag = etag.decode('utf-8')
        
    filename = url_to_filename(url, etag)
    #etag for bert-base-uncased is 74d4f96fdabdd865cbdbe905cd46c1f1 and url is the following:
    #https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json

    #filaname will be a str that is concatenation of the hashcode of the urlpath and the hashcode of the etag str

    # get cache path to put the file
    cache_path = os.path.join(cache_dir, filename)
    #cache_dir is equal to /Users/msanatkar/.cache/torch/transformers

    #cache_path for bert-base-uncased is the following:
    #/Users/msanatkar/.cache/torch/transformers/4dad0251492946e18ac39290fcfe91b89d370fee250efe9521476438fe8ca185.bf3b9ea126d8c0001ee8a1e8b92229871d06d36d8808208cc2449280da87785c

    # If we don't have a connection (etag is None) and can't identify the file
    # try to get the last downloaded one
    if not os.path.exists(cache_path) and etag is None:
        #here, in this if condition, we are saying if cache_path doesn't exist which means that we never downloaded this json config file in .cache before
        #and if we do not have access to the internet wihch is confirmed by etag being None, then, we try to find to get the latest downloaded one
        
        matching_files = fnmatch.filter(os.listdir(cache_dir), filename + '.*')
        #os.listdir will return all the files and directories in cache_dir which is /Users/msanatkar/.cache/torch/transormers
        #in above, fnmatch returns a sublist of files returned by listdir that matches the hash-based filename corresponding to this config json file
        
        matching_files = list(filter(lambda s: not s.endswith('.json'), matching_files))
        #in above, we only choose those files that do not end with ".json". It seems that for every encoder model, there exist two files in .cache
        #one of them is a josn file which will be the json config file describing the architecture of that model and the other one does not end with
        #json that must contain the weigths of the network
        
        if matching_files:
            cache_path = os.path.join(cache_dir, matching_files[-1])
            

    if resume_download:
        #resume_download is for those cases that for some reason the downloading process of the files was interupted before and here we want to resume the
        #download instead of starting from scratch
        incomplete_path = cache_path + '.incomplete'
        @contextmanager
        def _resumable_file_manager():
            with open(incomplete_path,'a+b') as f:
                yield f
            os.remove(incomplete_path)
        temp_file_manager = _resumable_file_manager
        if os.path.exists(incomplete_path):
            resume_size = os.stat(incomplete_path).st_size
        else:
            resume_size = 0
    else:
        temp_file_manager = tempfile.NamedTemporaryFile
        #here, temp_file_manager will be a temporary file that later on when the download is complete can be moved to the actual cache folder
        resume_size = 0

    #in below, we download the config file either if we didn't downlaod it before or the option force_download is True. Note: we never enable
    #force_download because we are not crazy!
    if not os.path.exists(cache_path) or force_download:
        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with temp_file_manager() as temp_file:
            logger.info("%s not found in cache or force_download set to True, downloading to %s", url, temp_file.name)

            # GET file object
            if url.startswith("s3://"):
                #for huggingface files, they don't start with s3
                if resume_download:
                    logger.warn('Warning: resumable downloads are not implemented for "s3://" urls')
                s3_get(url, temp_file, proxies=proxies)
            else:
                #http_get downloads the file and writes its content into temp_file
                http_get(url, temp_file, proxies=proxies, resume_size=resume_size)#resume_size will be zero if we didn't enable resume option
                #here, url refer to a json config file .json

            # we are copying the file before closing it, so flush to avoid truncation
            temp_file.flush()
            #flush method ensures that all the buffered data, are written into file
            
            # shutil.copyfileobj() starts at the current position, so go to the start
            temp_file.seek(0)

            logger.info("copying %s to cache at %s", temp_file.name, cache_path)
            with open(cache_path, 'wb') as cache_file:
                shutil.copyfileobj(temp_file, cache_file)
                #I believe cache_path here doesn't end with .json. In particular, if you look into .cache/torch/transformers, then there are bunch
                #of different resources which all of them have similar names hash(model_name).hash(url) with no .json suffix. Some of these files are
                #simply json config files of models and the other could be other resources like the weigths files. The json files inside the cache folder
                #reperesent the url path of the resource as well as the etag version. below, you can find how this json meta file is created!
                
            logger.info("creating metadata file for %s", cache_path)
            meta = {'url': url, 'etag': etag}
            meta_path = cache_path + '.json'
            with open(meta_path, 'w') as meta_file:
                output_string = json.dumps(meta)
                if sys.version_info[0] == 2 and isinstance(output_string, str):
                    output_string = unicode(output_string, 'utf-8')  # The beauty of python 2
                meta_file.write(output_string)

            logger.info("removing temp file %s", temp_file.name)

    return cache_path
