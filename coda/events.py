import numpy as np
import pandas as pd
import re
from glob import glob
from six import string_types
from functools import partial
import json
from bids.grabbids import BIDSLayout

__all__ = ['BIDSEventReader', 'FSLEventReader', 'EventTransformer']


class Transformations(object):

    @staticmethod
    def standardize(data, demean=True, rescale=True):
        if demean:
            data -= data.mean(0)
        if rescale:
            data /= data.std(0)
        return data

    @staticmethod
    def orthogonalize(data, other):
        ''' Orthogonalize each of the variables in cols with respect to all
        of the variables in x_cols.
        '''
        y = data.values
        X = other.values
        _aX = np.c_[np.ones(len(y)), X]
        coefs, resids, rank, s = np.linalg.lstsq(_aX, y)
        return y - X.dot(coefs[1:])

    @staticmethod
    def binarize(data, threshold=0.0):
        above = data > threshold
        data[above] = 1
        data[~above] = 0
        return data


def alias(target, append=False):
    def decorator(func):
        def wrapper(self, cols, groupby=None, output=None, *args, **kwargs):

            cols = self._select_cols(cols)

            if 'other' in kwargs:
                kwargs['other'] = self.data[kwargs['other']]

            # groupby can be either a single column, in which case it's
            # interpreted as a categorical variable to groupby directly, or a
            # list of column output, in which case it's interpreted as a set of
            # dummy columns to reconstruct a categorical from.
            if groupby is not None:
                groupby = self._select_cols(groupby)
                if len(groupby) > 1:
                    group_results = []
                    output = ['%s_%s' % (cn, gn)for cn in cols for gn in groupby]
                    for i, col in enumerate(groupby):
                        _result = target(self.data[cols], *args, **kwargs)
                        group_results.extend(_result.T)
                    result = np.c_[group_results].squeeze().T
                    result = pd.DataFrame(result, columns=output)
                else:
                    result = self.data.groupby(groupby)[cols].apply(
                        target, *args, **kwargs)
            else:
                result = target(self.data[cols], *args, **kwargs)

            if append:
                output = result.columns

            if output is not None:
                result = pd.DataFrame(result, columns=output)
                self.data = self.data.join(result)
            else:
                self.data[cols] = result
        return wrapper
    return decorator


class EventTransformer(object):

    def __init__(self, events, orig_hz=1, target_hz=1000):
        self.events = events
        self.orig_hz = orig_hz
        self.target_hz = target_hz
        self._to_dense()

    # Aliased functions #
    @alias(np.log)
    def log(): pass

    @alias(partial(np.any, axis=1))
    def or_(): pass

    @alias(partial(np.all, axis=1))
    def and_(): pass

    @alias(pd.get_dummies, append=True)
    def factor(): pass

    @alias(Transformations.standardize)
    def standardize(): pass

    @alias(Transformations.binarize)
    def binarize(): pass

    @alias(Transformations.orthogonalize)
    def orthogonalize(): pass

    # Standard instance methods #
    def select(self, cols):
        # Always retain onsets
        if 'onset' not in cols:
            cols.insert(0, 'onset')
        self.data = self.data[self._select_cols(cols)]

    def formula(self, f, target=None, replace=False, *args, **kwargs):
        from patsy import dmatrix
        result = dmatrix(f, self.data,
                         return_type='dataframe', *args, **kwargs)
        if target is not None:
            self.data[target] = result
        elif replace:
            self.data[result.columns] = result
        else:
            raise ValueError("Either a target column must be passed or replace"
                             " must be True.")

    def multiply(self, cols, x_cols):
        x_cols = self._select_cols(x_cols)
        result = self.data[x_cols].apply(
            lambda x: np.multiply(x, self.data[cols]))
        output = ['%s_%s' % (cols, x) for x in x_cols]
        self.data[output] = result

    def rename(self, cols, output):
        rename = dict(zip(cols, output))
        self.data = self.data.rename(columns=rename)

    def query(self, q, *args, **kwargs):
        self.data = self.data.query(filter)

    def apply(self, func, *args, **kwargs):
        if isinstance(func, string_types):
            func = getattr(self, func)
        func(*args, **kwargs)

    def apply_from_json(self, json_file):
        tf = json.load(open(json_file, 'rU'))
        for t in tf['transformations']:
                    name = t.pop('name')
                    cols = t.pop('input', None)
                    self.apply(name, cols, **t)

    def _select_cols(self, cols):
        if isinstance(cols, string_types) and '*' in cols:
            patt = re.compile(cols.replace('*', '.*'))
            cols = [l for l in self.data.columns.tolist()
                    for m in [patt.search(l)] if m]
        return cols

    def _to_dense(self):
        """ Convert the sparse [onset, duration, amplitude] representation
        typical of event files to a dense matrix where each row represents
        a fixed unit of time. """
        end = int((self.events['onset'] + self.events['duration']).max())

        targ_hz, orig_hz = self.target_hz, self.orig_hz
        len_ts = end * targ_hz
        conditions = self.events['condition'].unique().tolist()
        n_conditions = len(conditions)
        ts = np.zeros((len_ts, n_conditions))

        _events = self.events.copy().reset_index()
        _events[['onset', 'duration']] = \
            _events[['onset', 'duration']] * targ_hz / orig_hz

        cond_index = [conditions.index(c) for c in _events['condition']]
        ev_end = np.round(_events['onset'] + _events['duration']).astype(int)
        onsets = np.round(_events['onset']).astype(int)

        for i, row in _events.iterrows():
            ts[onsets[i]:ev_end[i], cond_index[i]] = row['amplitude']

        self.data = pd.DataFrame(ts, columns=conditions)
        onsets = np.arange(len(ts)) / self.target_hz
        self.data.insert(0, 'onset', onsets)

    def resample(self, sampling_rate):
        """
        Resample the design matrix to the specified sampling rate (in seconds).
        Primarily useful for downsampling to match the TR, so as to export the
        design as a n(TR) x n(conds) matrix.
        """
        sampling_rate = np.round(sampling_rate * 1000)
        self.data.index = pd.to_datetime(self.data['onset'], unit='s')
        self.data = self.data.resample('%dL' % sampling_rate).mean()
        self.data['onset'] = self.data.index.astype(np.int64) / int(1e9)
        self.data = self.data.reset_index(drop=True)

    """
    Maybe need to remove "grouping" columns from here.
    How to differentiate between grouping and "data" columns.
    """
    def get_data(self, by_run=True):
        return self.data


class EventReader(object):
    def _add_patterns(self, data, event):
        """
        For each group pattern, tries to find matching pattern in a given file
        name, and then adds the matching pattern to data frame using
        the given pattern name.

        Next, adds additional event attributes as columns in data, replacing
        file patterns if there is a conflict.
        """

        for name, pattern in self.group_patterns.iteritems():
            m = re.search(pattern, event['filename'])
            if m is None:
                raise ValueError(
                    "Pattern '{}' failed to match any part of "
                    "filename '{}'.".format(name, file))
            data[name] = m.group(1)

        for name, value in event.iteritems():
            if name not in ['ext', 'filename']:
                data[name] = value

        return data

    def _validate_columns(self, data, f):
        """ Checks for necessary columns in event files """
        cols = data.columns

        if 'onset' not in cols:
            raise ValueError(
                'Event file "%s" is missing \'onset\' column.' % f)
        if 'duration' not in cols:
            if self.default_duration is None:
                raise ValueError(
                    'Event file "%s" is missing \'duration\''
                    ' column, and no default_duration was provided.' % f)
            else:
                data['duration'] = self.default_duration

        return data


class FSLEventReader(EventReader):
    """ Reads in FSL-style event files into long format pandas dataframe """
    def __init__(self, columns=None, header='infer', sep=None,
                 default_duration=0., default_amplitude=1.,
                 group_patterns={'condition': '(.*)\.[a-zA-Z0-9]{3,4}'}):
        '''
        Args:
            columns (list): Optional list of column output to use. If passed,
                number of elements must match number of columns in the text
                files to be read. If omitted, column output are inferred by
                pandas (depending on value of header).
            header (str): passed to pandas; see pd.read_table docs for details.
            sep (str): column separator; see pd.read_table docs for details.
            default_duration (float): Optional default duration to set for all
                events. Will be ignored if a column named 'duration' is found.
            default_amplitude (float): Optional default amplitude to set for
                all events. Will be ignored if an amplitude column is found.
            group_patterns (dict): pairs of pattern names and regex with which
                to capture groups from the input text file fileoutput.
                Only the first captured group will be used for each.
                Defaults to setting condition to file base name.
        '''

        self.columns = columns
        self.header = header
        self.sep = sep
        self.default_duration = default_duration
        self.default_amplitude = default_amplitude
        if group_patterns is None:
            group_patterns = {}
        self.group_patterns = group_patterns

    def read(self, path, rename=None, **kwargs):
        dfs = []
        if isinstance(path, string_types):
            path = glob(path)

        for f in path:
            _data = pd.read_table(f, names=self.columns, header=self.header,
                                  sep=self.sep)

            if rename is not None:
                _data = _data.rename(rename)

            _data = self._validate_columns(_data, f)
            kwargs.update({'filename': f})
            _data = self._add_patterns(_data, kwargs)

            if 'amplitude' not in _data.columns:
                _data['amplitude'] = self.default_amplitude

            dfs.append(_data)

        return pd.concat(dfs, axis=0)


class BIDSEventReader(EventReader):
    """ Reads in BIDS event tsv files into long format pandas dataframe """
    def __init__(self, default_duration=0., default_amplitude=1.,
                 amplitude_column=None, condition_column='trial_type',
                 sep='\t', base_dir=None, group_patterns=None):
        self.default_duration = default_duration
        self.default_amplitude = default_amplitude
        self.condition_column = condition_column
        self.amplitude_column = amplitude_column
        self.sep = sep
        self.base_dir = base_dir
        if group_patterns is None:
            group_patterns = {}
        self.group_patterns = group_patterns

    def read(self, path=None, **kwargs):
        """ Read in events.tsv file, either by specifying file name, or
        by passing in argument with which to query BIDS directory.if
        e.g. subject, run, etc... """

        if self.base_dir is None:
            if isinstance(path, str):
                path = glob(path)
            events = [{'filename': f} for f in path]

        else:
            events = BIDSLayout(self.base_dir).get(type='events', **kwargs)
            events = [dict(e.__dict__) for e in events]

        if not events:
            raise Exception("BIDS event file(s) could not be found"
                            "or were not provided.")

        dfs = []
        for event in events:
            f = event['filename']
            _data = pd.read_table(f, sep=self.sep)
            _data = self._validate_columns(_data, f)

            # If condition column is provided, either extract amplitudes
            # from given amplitude column, or to default value
            if self.condition_column is not None:
                if self.condition_column not in _data.columns:
                    raise ValueError(
                        "Event file is missing the specified"
                        "condition column, {}".format(self.condition_column))
                else:
                    if self.amplitude_column is not None:
                        if self.amplitude_column not in _data.columns:
                            raise ValueError(
                                "Event file is missing the specified"
                                "amplitude column, {}".format(
                                    self.amplitude_column))
                        else:
                            amplitude = _data[self.amplitude_column]
                    else:
                        amplitude = self.default_amplitude

                    _data['amplitude'] = amplitude
                    _data['condition'] = _data['trial_type']

            else:
                # If no condition specified, get amplitudes from all columns,
                # except 'trial_type'
                if 'trial_type' in _data.columns:
                    _data.drop('trial_type', axis=1, inplace=True)

                _data = pd.melt(_data, id_vars=['onset', 'duration'],
                                value_name='amplitude', var_name='condition')

            # Drop non-coda columns
            _data = _data.drop(
                [c for c in _data.columns if c not in
                 ['onset', 'condition', 'amplitude', 'duration']], axis=1)

            _data = self._add_patterns(_data, event)
            dfs.append(_data)
        return pd.concat(dfs)
