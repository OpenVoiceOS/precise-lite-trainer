# Copyright 2019 Mycroft AI Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass
from os.path import isfile
from typing import *

from keras.layers import Dense, GRU
from keras.models import Sequential, load_model
from precise_lite_runner.params import params

from precise_trainer.functions import false_pos, false_neg, \
    weighted_log_loss, set_loss_bias


@dataclass
class ModelParams:
    """
    Attributes:
        recurrent_units:
        dropout:
        extra_metrics: Whether to include false positive and false negative metrics
        skip_acc: Whether to skip accuracy calculation while training
    """
    recurrent_units: int = 20
    dropout: float = 0.2
    extra_metrics: bool = False
    skip_acc: bool = False
    loss_bias: float = 0.7
    freeze_till: int = 0


def get_model(model_name: Optional[str], model_params: ModelParams) -> 'Sequential':
    """
    Load or create a precise_lite model

    Args:
        model_name: Name of model
        model_params: Parameters used to create the model

    Returns:
        model: Loaded Keras model
    """
    if model_name and isfile(model_name):
        print('Loading from ' + model_name + '...')
        model = load_model(model_name)
    else:
        model = create_precise_model(model_params.recurrent_units,
                                     model_params.dropout)

    metrics = ['accuracy'] + model_params.extra_metrics * [false_pos, false_neg]
    set_loss_bias(model_params.loss_bias)
    for i in model.layers[:model_params.freeze_till]:
        i.trainable = False
    model.compile('rmsprop', weighted_log_loss,
                  metrics=(not model_params.skip_acc) * metrics)
    return model


def create_precise_model(recurrent_units, dropout):
    model = Sequential()
    model.add(GRU(
        recurrent_units, activation='linear',
        input_shape=(params.n_features, params.feature_size),
        dropout=dropout, name='net'
    ))
    model.add(Dense(1, activation='sigmoid'))
    return model
