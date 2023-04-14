#!/usr/bin/env python3
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
import os
import random
from math import exp
from os.path import splitext, isfile
from pprint import pprint
from typing import Tuple
import datetime

import numpy as np
import tensorflow as tf  # Using tensorflow v2.2
from keras.callbacks import LambdaCallback, ModelCheckpoint, TensorBoard
from keras.models import load_model
from precise_lite_runner.runner import TFLiteRunner

from precise_trainer.functions import weighted_log_loss
from precise_trainer.model import get_model, ModelParams
from precise_trainer.stats import Stats
from precise_trainer.train_data import TrainData


class PreciseTrainer:
    """
        Train a new model on a dataset

        :model str
            Keras model file (.net) to load from and save to

        :folder str
            Folder to load wav files from

        :-e --epochs int 10
            Number of epochs to train model for

        :-b --batch-size int 5000
            Batch size for training

        :-sb --save-best
            Only save the model each epoch if its stats improve

        :-nv --no-validation
            Disable accuracy and validation calculation
            to improve speed during training

        :-mm --metric-monitor str loss
            Metric used to determine when to save

    """

    def __init__(self, model, folder, model_params=None, epochs=50, batch_size=512,
                 save_best=True, no_validation=False, metric_monitor="loss",
                 log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                 ):

        model_params = model_params or ModelParams(skip_acc=no_validation, extra_metrics=False,
                                                   loss_bias=0.8, freeze_till=0)

        self.train_epochs = epochs
        self.batch_size = batch_size
        self.path = model
        self.model = get_model(model, model_params)
        self.no_validation = no_validation
        self.extra_metrics = extra_metrics
        self.train_data, self.test_data = self.load_data(folder, no_validation)

        checkpoint = ModelCheckpoint(model, monitor=metric_monitor, save_best_only=save_best)

        self.epoch = 0
        if isfile(splitext(model)[0] + '.epoch'):
            with open(splitext(model)[0] + '.epoch') as f:
                self.epoch = int(f.read())

        def on_epoch_end(_a, _b):
            self.epoch += 1
            with open(splitext(model)[0] + '.epoch', "w") as f:
                f.write(str(self.epoch))

        self.model_base = splitext(model)[0]

        self.samples = set()
        self.hash_to_ind = {}

        self.callbacks = [
            checkpoint,
            LambdaCallback(on_epoch_end=on_epoch_end),
            TensorBoard(log_dir=log_dir, histogram_freq=1)
        ]

    @staticmethod
    def load_data(folder, no_validation=False) -> Tuple[tuple, tuple]:
        data = TrainData.from_folder(folder)
        print('Data:', data)
        train, test = data.load(True, not no_validation)

        print('Inputs shape:', train[0].shape)
        print('Outputs shape:', train[1].shape)

        if test:
            print('Test inputs shape:', test[0].shape)
            print('Test outputs shape:', test[1].shape)

        if 0 in train[0].shape or 0 in train[1].shape:
            print('Not enough data to train')
            exit(1)

        return train, test

    @property
    def sampled_data(self):
        """Returns (train_inputs, train_outputs)"""
        return self.train_data[0], self.train_data[1]

    def train(self, convert=True):
        self.model.summary()
        train_inputs, train_outputs = self.sampled_data
        self.model.fit(
            train_inputs, train_outputs, self.batch_size,
            self.epoch + self.train_epochs, validation_data=self.test_data,
            initial_epoch=self.epoch, callbacks=self.callbacks,
            use_multiprocessing=True, validation_freq=5,
            verbose=1
        )
        if convert:
            return self.convert(self.path, f"{self.path}/model.tflite")
        else:
            self.model.save(self.path + ".h5")
            return self.path + ".h5"

    def _replace(self, porportion=0.4, balanced=True):
        pos_samples = []
        neg_samples = []
        for idx, x in enumerate(self.train_data[0]):
            y = self.train_data[1][idx]
            if y[0]:
                pos_samples.append((x, y))
            else:
                neg_samples.append((x, y))

        X = []
        y = []

        if balanced:
            s1 = s2 = int((len(neg_samples + pos_samples) * porportion)/2)
        else:
            s1 = int(len(pos_samples) * porportion)
            s2 = int(len(neg_samples) * porportion)

        # randomly sample data points
        while len(X) < s1:
            ns = random.choice(pos_samples)
            X.append(ns[0])
            y.append(ns[1])

        while len(X) < s1 + s2:
            ns = random.choice(neg_samples)
            X.append(ns[0])
            y.append(ns[1])

        return np.array(X), np.array(y)

    def train_with_replacement(self, mini_epochs=5, porportion=0.4, balanced=True, convert=True):
        self.model.summary()

        for i in range(self.train_epochs):
            train_inputs, train_outputs = self._replace(porportion, balanced)
            self.model.fit(
                train_inputs, train_outputs, self.batch_size,
                self.epoch + mini_epochs, validation_data=self.test_data,
                initial_epoch=self.epoch, callbacks=self.callbacks,
                use_multiprocessing=True, validation_freq=5,
                verbose=1
            )
            self.epoch += mini_epochs
        if convert:
            return self.convert(self.path, f"{self.path}/model.tflite")
        else:
            self.model.save(self.path + ".h5")
            return self.path + ".h5"

    def train_optimized(self, trials_name=".cache/trials", cycles=50, loss_bias=0.8, convert=True, backend="mixture"):
        from bbopt import BlackBoxOptimizer
        bb = BlackBoxOptimizer(file=trials_name)
        print('Writing to:', trials_name + '.bbopt.json')
        for i in range(cycles):
            if backend == "mixture":
                bb.run_backend("mixture", [
                    #    ("tree_structured_parzen_estimator", 1),
                   # ("annealing", 1),
                    ("gaussian_process", 1),
                   #  ("random_forest", 1),
                    #   ("extra_trees", 1),
                    ("gradient_boosted_regression_trees", 1),
                ])
            else:
                bb.run(backend)

            print("\n= %d = (example #%d)" % (i + 1, len(bb.get_data()["examples"]) + 1))

            params = ModelParams(
                recurrent_units=bb.randint("units", 1, 70, guess=50),
                dropout=bb.uniform("dropout", 0.1, 0.9, guess=0.6),
                extra_metrics=self.extra_metrics,
                skip_acc=self.no_validation,
                loss_bias=loss_bias
            )
            print('Testing with:', params)
            model = get_model(self.path, params)
            model.fit(
                *self.sampled_data, batch_size=self.batch_size,
                epochs=self.epoch + self.train_epochs,
                validation_data=self.test_data * (not self.no_validation),
                callbacks=self.callbacks, initial_epoch=self.epoch
            )
            resp = model.evaluate(*self.test_data, batch_size=self.batch_size)
            if not isinstance(resp, (list, tuple)):
                resp = [resp, None]
            test_loss, test_acc = resp
            predictions = model.predict(self.test_data[0], batch_size=self.batch_size)

            num_false_positive = np.sum(predictions * (1 - self.test_data[1]) > 0.5)
            num_false_negative = np.sum((1 - predictions) * self.test_data[1] > 0.5)
            false_positives = num_false_positive / np.sum(self.test_data[1] < 0.5)
            false_negatives = num_false_negative / np.sum(self.test_data[1] > 0.5)

            param_score = 1.0 / (1.0 + exp((model.count_params() - 11000) / 2000))
            fitness = param_score * (1.0 - 0.2 * false_negatives - 0.8 * false_positives)

            bb.remember({
                "test loss": test_loss,
                "test accuracy": test_acc,
                "false positive%": false_positives,
                "false negative%": false_negatives,
                "fitness": fitness
            })

            print("False positive: ", false_positives * 100, "%")

            bb.maximize(fitness)
            pprint(bb.get_current_run())

        best = bb.get_optimal_run()
        print("\n= BEST = (example #%d)" % bb.get_data()["examples"].index(best))
        pprint(best)
        if convert:
            return self.convert(self.path, f"{self.path}/model.tflite")
        else:
            self.model.save(self.path + ".h5")
            return self.path + ".h5"

    @staticmethod
    def test(model, folder, use_train=False, threshold=0.5, no_filenames=False):
        print(model)
        data = TrainData.from_folder(folder)
        train, test = data.load(use_train, not use_train, shuffle=False)
        inputs, targets = train if use_train else test

        filenames = sum(data.train_files if use_train else data.test_files, [])
        predictions = TFLiteRunner(model).predict(inputs)
        stats = Stats(predictions, targets, filenames)

        print('Data:', data)

        if not no_filenames:
            fp_files = stats.calc_filenames(False, True, threshold)
            fn_files = stats.calc_filenames(False, False, threshold)
            print('=== False Positives ===')
            print('\n'.join(fp_files))
            print()
            print('=== False Negatives ===')
            print('\n'.join(fn_files))
            print()
        print(stats.counts_str(threshold))
        print()
        print(stats.summary_str(threshold))

    @staticmethod
    def convert(model_path: str, out_file: str):
        """
        Converts an HD5F file from Keras to a .tflite for use with TensorFlow Runtime

        Args:
            model_path: location of Keras model
            out_file: location to write TFLite model
        """
        print('Converting', model_path, 'to', out_file, '...')

        out_dir, filename = os.path.split(out_file)
        out_dir = out_dir or '.'
        os.makedirs(out_dir, exist_ok=True)

        # Load custom loss function with model
        model = load_model(model_path, custom_objects={'weighted_log_loss': weighted_log_loss})
        model.summary()

        # Support for freezing Keras models to .pb has been removed in TF 2.0.

        # Converting instead to TFLite model
        print('Starting TFLite conversion.')
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                tf.lite.OpsSet.SELECT_TF_OPS]
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.experimental_new_converter = True
        converter.allow_custom_ops = True
        converter._experimental_default_to_single_batch_in_tensor_list_ops = True
        tflite_model = converter.convert()
        open(out_file, "wb").write(tflite_model)
        print('Wrote to ' + out_file)
        return out_file


if __name__ == "__main__":

    """
    :-s --sensitivity float 0.2
        Weighted loss bias. Higher values decrease increase positives

    :-em --extra-metrics
        Add extra metrics during training

    :-f --freeze-till int 0
        Freeze all weights up to this index (non-inclusive).
        Can be negative to wrap from end
    """

    # PreciseTrainer.convert(intput, output)

    extra_metrics = False
    no_validation = False
    freeze_till = 0
    sensitivity = 0.2

    params = ModelParams(skip_acc=no_validation, extra_metrics=extra_metrics,
                         loss_bias=1.0 - sensitivity, freeze_till=freeze_till)
    model_name = "hey_chatterbox"
    folder = f"/tmp/{model_name}"
    model_path = f"/home/miro/PycharmProjects/ovos-audio-classifiers/trained/{model_name}"
    log_dir = f"logs/fit/{model_name}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    trainer = PreciseTrainer(model_path, folder, epochs=200, log_dir=log_dir)
    model_file = trainer.train_with_replacement()
    # look for best hyperparams during a few cycles
    # model_file = trainer.train_optimized(cycles=20)
    # train the best model for more epochs
    #trainer.train_epochs = 5000
    #model_file = trainer.train()
    trainer.test(model_file, folder)
