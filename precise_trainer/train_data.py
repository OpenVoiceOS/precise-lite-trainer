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
import fnmatch
import hashlib
import json
import os
import shutil
from glob import glob
from hashlib import md5
from os import makedirs
from os import name as os_name
from os.path import join, getmtime, isfile
from typing import *
from typing import Callable, Union

import numpy as np
from precise_lite_runner.params import ListenerParams
from precise_lite_runner.vectorization import vectorize_delta, vectorize


def load_audio(file: Any) -> np.ndarray:
    """
    Args:
        file: Audio filename or file object
    Returns:
        samples: Sample rate and audio samples from 0..1
    """
    import wavio
    import wave

    try:
        wav = wavio.read(file)
    except (EOFError, wave.Error):
        wav = wavio.Wav(np.array([[]], dtype=np.int16), 16000, 2)
    if wav.data.dtype != np.int16:
        raise ValueError('Unsupported data type: ' + str(wav.data.dtype))
    if wav.rate != ListenerParams().sample_rate:
        raise ValueError('Unsupported sample rate: ' + str(wav.rate))

    data = np.squeeze(wav.data)
    return data.astype(np.float32) / float(np.iinfo(data.dtype).max)


def glob_all(folder: str, filt: str) -> List[str]:
    """Recursive glob"""
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in fnmatch.filter(filenames, filt):
            matches.append(os.path.join(root, filename))
    return matches


def find_wavs(folder: str) -> Tuple[List[str], List[str]]:
    """Finds wake-word and not-wake-word wavs in folder"""
    return (glob_all(join(folder, 'wake-word'), '*.wav'),
            glob_all(join(folder, 'not-wake-word'), '*.wav'))


def vectorization_md5_hash(pr):
    """Hash all the fields related to audio vectorization"""
    keys = sorted(pr.__dict__)
    keys.remove('threshold_config')
    keys.remove('threshold_center')
    return hashlib.md5(
        str([pr.__dict__[i] for i in keys]).encode()
    ).hexdigest()


class Pyache:
    """
    A class for storing and interacting with a folder numpy cache
    Args:
        folder: Location of cache
        loader: Function that receives a filename and returns a numpy array of consistent size
        loader_id: Unique string associated with loader
        max_loaders: Maximum number of loader caches before others are deleted
    """

    @property
    def file_delimiter(self) -> str:
        """
        Returns:
            an OS dependent string used as a delimiter in file names to improve readability.
        """
        if os_name == "nt":
            return "_"
        else:
            return "::"  # Keeping backwards compatibility (invalid file name character on windows)

    def __init__(self, folder: str, loader: Callable, loader_id: str, max_loaders=3):
        self.folder = folder
        self.loader = loader
        self.loader_id = loader_id
        self.max_loaders = max_loaders
        self.loader_folder = join(folder, 'loader{}{}'.format(self.file_delimiter, loader_id))
        self.data_folder = join(self.loader_folder, 'data')
        makedirs(self.data_folder, exist_ok=True)

    def load_file(self, filename, on_gen=lambda x: None) -> Union[np.ndarray, None]:
        """
        Loads a single file
        Args:
            filename: File to load
            on_gen: Function to call on a cache miss. Receives filename as only parameter
        Returns:
            np.ndarray: The loaded array or None if there was a ValueError while loading
        """
        cache_file = join(self.data_folder, md5(filename.encode()).hexdigest() + '.npy')
        if isfile(cache_file):
            return np.load(cache_file)
        try:
            data = self.loader(filename)
        except ValueError:
            return None
        np.save(cache_file, data)
        on_gen(filename)
        return data

    def load(self, filenames: list, on_gen=lambda x: None, on_loop=lambda: None) -> np.ndarray:
        """
        Load a chunk of filenames as a single numpy array, possibly from cache
        Args:
            filenames: List of filenames to load
            on_gen: Function receiving a filename that is run on all cache misses
            on_loop: Function called each loop where a new file was loaded
        Returns:
            np.ndarray: All data from filenames as a single numpy array
        """
        my_suffix = 'cacheblock' + self.file_delimiter + md5(''.join(sorted(filenames)).encode()).hexdigest() + '.npy'
        self._remove_old_resource(self.folder, self.max_loaders, 'loader' + self.file_delimiter,
                                  'loader{}{}/'.format(self.file_delimiter, self.loader_id))

        self._remove_old_resource(self.loader_folder, self.max_loaders, 'cacheblock' + self.file_delimiter, my_suffix)
        cache_file = join(self.loader_folder, my_suffix)
        if isfile(cache_file):
            try:
                return np.load(cache_file)
            except OSError:
                pass
        data = []
        for filename in filenames:
            res = self.load_file(filename, on_gen)
            if res is not None:
                data.append(res)
            on_loop()
        np.save(cache_file, data)
        return np.array(data)

    def cleanup(self):
        self._remove_old_resource(self.folder, self.max_loaders, 'loader' + self.file_delimiter)
        self._remove_old_resource(self.loader_folder, self.max_loaders, 'cacheblock' + self.file_delimiter)

    @staticmethod
    def _remove_old_resource(folder, max_count, prefix, my_suffix=None):
        remaining = max_count - 1  # Doesn't include ourself
        other_loaders = [i for i in glob(join(folder, prefix + '*/')) if not my_suffix or not i.endswith(my_suffix)]
        if len(other_loaders) > remaining:
            to_delete = sorted(other_loaders, key=getmtime)[:len(other_loaders) - remaining]
            for i in to_delete:
                shutil.rmtree(i)


class TrainData:
    """Class to handle loading of wave data from categorized folders and tagged text files
        :folder str
            Folder to load wav files from
    """

    def __init__(self, train_files: Tuple[List[str], List[str]],
                 test_files: Tuple[List[str], List[str]]):
        self.train_files, self.test_files = train_files, test_files

    @classmethod
    def from_folder(cls, folder: str) -> 'TrainData':
        """
        Load a set of data from a structured folder in the following format:
        {prefix}/
            wake-word/
                *.wav
            not-wake-word/
                *.wav
            test/
                wake-word/
                    *.wav
                not-wake-word/
                    *.wav
        """
        return cls(find_wavs(folder), find_wavs(join(folder, 'test')))

    @classmethod
    def from_tags(cls, tags_file: str, tags_folder: str) -> 'TrainData':
        """
        Load a set of data from a text file with tags in the following format:
            <file_id>  (tab)  <tag>
            <file_id>  (tab)  <tag>

            file_id: identifier of file such that the following
                     file exists: {tags_folder}/{data_id}.wav
            tag: "wake-word" or "not-wake-word"
        """
        if not tags_file:
            num_ignored_wavs = len(glob(join(tags_folder, '*.wav')))
            if num_ignored_wavs > 10:
                print('WARNING: Found {} wavs but no tags file specified!'.format(num_ignored_wavs))
            return cls(([], []), ([], []))
        if not isfile(tags_file):
            raise RuntimeError('Database file does not exist: ' + tags_file)

        train_groups = {}
        train_group_file = join(tags_file.replace('.txt', '') + '.groups.json')
        if isfile(train_group_file):
            try:
                with open(train_group_file) as f:
                    train_groups = json.load(f)
            except ValueError:
                pass

        tags_files = {
            'wake-word': [],
            'not-wake-word': []
        }
        with open(tags_file) as f:
            for line in f.read().split('\n'):
                if not line:
                    continue
                file, tag = line.split('\t')
                tags_files[tag.strip()].append(join(tags_folder, file.strip() + '.wav'))

        train_files, test_files = ([], []), ([], [])
        for label, rows in enumerate([tags_files['wake-word'], tags_files['not-wake-word']]):
            for fn in rows:
                if not isfile(fn):
                    print('Missing file:', fn)
                    continue
                if fn not in train_groups:
                    train_groups[fn] = (
                        'test' if md5(fn.encode('utf8')).hexdigest() > 'c' * 32
                        else 'train'
                    )
                {
                    'train': train_files,
                    'test': test_files
                }[train_groups[fn]][label].append(fn)

        with open(train_group_file, 'w') as f:
            json.dump(train_groups, f)

        return cls(train_files, test_files)

    @classmethod
    def from_both(cls, tags_file: str, tags_folder: str, folder: str) -> 'TrainData':
        """Load data from both a database and a structured folder"""
        return cls.from_tags(tags_file, tags_folder) + cls.from_folder(folder)

    def load(self, train=True, test=True, shuffle=True) -> tuple:
        """
        Load the vectorized representations of the stored data files
        Args:
            train: Whether to load train data
            test: Whether to load test data
        """
        return self.__load(self.__load_files, train, test, shuffle=shuffle)

    @staticmethod
    def merge(data_a: tuple, data_b: tuple) -> tuple:
        return np.concatenate((data_a[0], data_b[0])), np.concatenate((data_a[1], data_b[1]))

    def __repr__(self) -> str:
        string = '<TrainData wake_words={kws} not_wake_words={nkws}' \
                 ' test_wake_words={test_kws} test_not_wake_words={test_nkws}>'
        return string.format(
            kws=len(self.train_files[0]), nkws=len(self.train_files[1]),
            test_kws=len(self.test_files[0]), test_nkws=len(self.test_files[1])
        )

    def __add__(self, other: 'TrainData') -> 'TrainData':
        if not isinstance(other, TrainData):
            raise TypeError('Can only add TrainData to TrainData')
        return TrainData((self.train_files[0] + other.train_files[0],
                          self.train_files[1] + other.train_files[1]),
                         (self.test_files[0] + other.test_files[0],
                          self.test_files[1] + other.test_files[1]))

    def __load(self, loader: Callable, train: bool, test: bool, **kwargs) -> tuple:
        return tuple([
            loader(*files, **kwargs) if files else None
            for files in (train and self.train_files,
                          test and self.test_files)
        ])

    @staticmethod
    def __load_files(kw_files: list, nkw_files: list, vectorizer: Callable = None, shuffle=True) -> tuple:
        params = ListenerParams()

        input_parts = []
        output_parts = []

        vectorizer = vectorizer or (vectorize_delta if params.use_delta else vectorize)
        cache = Pyache('.cache', lambda x: vectorizer(load_audio(x)),
                       vectorization_md5_hash(params))

        def add(filenames, output):
            def on_loop():
                on_loop.i += 1
                print('\r{0:.2%}  '.format(on_loop.i / len(filenames)), end='', flush=True)

            on_loop.i = 0

            new_inputs = cache.load(filenames, on_loop=on_loop)
            new_outputs = np.array([[output] for _ in range(len(new_inputs))])
            if new_inputs.size == 0:
                new_inputs = np.empty((0, params.n_features, params.feature_size))
            if new_outputs.size == 0:
                new_outputs = np.empty((0, 1))
            input_parts.append(new_inputs)
            output_parts.append(new_outputs)
            print('\r       \r', end='', flush=True)

        print('Loading wake-word...')
        add(kw_files, 1.0)

        print('Loading not-wake-word...')
        add(nkw_files, 0.0)

        inputs = np.concatenate(input_parts) if input_parts \
            else np.empty((0, params.n_features, params.feature_size))
        outputs = np.concatenate(output_parts) if output_parts \
            else np.empty((0, 1))

        shuffle_ids = np.arange(len(inputs))
        if shuffle:
            np.random.shuffle(shuffle_ids)
        return inputs[shuffle_ids], outputs[shuffle_ids]
