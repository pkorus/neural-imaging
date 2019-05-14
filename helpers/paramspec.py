#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import regex
import types

from helpers import utils


class ParamSpec(object):

    def __init__(self, specs):

        # Validate the specs
        for key, spec in specs.items():

            if type(spec) is not tuple and len(spec) != 3:
                raise ValueError('Invalid parameter specification for key {} - expected tuple of length 3'.format(key))

            if spec[2] is None:
                continue

            if spec[1] is str and not any(type(spec[2]) is s for s in [str, set, types.FunctionType]):
                raise ValueError('String data types can be validated by a regex (string), enum (set) or custom function')

            if utils.is_numeric_type(spec[1]) and not any(type(spec[2]) is s for s in [tuple, set]):
                raise ValueError('Numeric data types can be validated by a range (2-elem tuple), or enum (set)')

        self.__dict__['_specs'] = specs
        self.__dict__['_values'] = {}

    def __getattr__(self, name):
        if name in self._values:
            return self._values[name]
        elif name in self._specs:
            return self._specs[name][0]
        else:
            raise KeyError(name)

    def __setattr__(self, key, value):
        raise ValueError('Values cannot be set directly. Use the `update` method.')

    def get_dtype(self, name):
        return self._specs[name][1]
    
    def get_default(self, name):
        return self._specs[name][0]
    
    def get_value(self, name):
        return self.__getattr__(name)
    
    def get_min(self, name):
        validation = self._specs[name][2]
        if type(validation) is tuple and len(validation) == 2:
            return validation[0]
        else:
            return None
    
    def get_max(self, name):
        validation = self._specs[name][2]
        if type(validation) is tuple and len(validation) == 2:
            return validation[1]
        else:
            return None
    
    def get_enum(self, name):
        validation = self._specs[name][2]
        if type(validation) is set:
            return set(validation)
        else:
            return None
    
    def get_regex(self, name):
        validation = self._specs[name][2]
        if type(validation) is str:
            return validation
        else:
            return None
    
    def __repr__(self):
        return 'ParamSpec()'

    def to_dict(self):
        params = {key: spec[0] for key, spec in self._specs.items()}
        params.update(self._values)
        return params

    def to_json(self):
        params = self.to_dict()
        params = {k: v if utils.is_number(v) else str(v) for k, v in params.items()}
        return params

    def __contains__(self, item):
        return item in self._specs
    
    def update(self, **params):

        # Iterate over submitted values
        for key, value in params.items():

            if key in self._specs:
                # Get specification for the current parameter
                _, dtype, validation = self._specs[key]

                # Accept the new value if it:
                #   is not None
                #   is not np.nan (for numerical types)
                #   passes validation checks
                if value is not None:

                    if utils.is_number(value) and np.isnan(value):
                        raise ValueError('Invalid value {} for attribute {}'.format(value, key))

                    candidate = value if dtype is None else dtype(value)

                    # Validation checks
                    if validation is not None:

                        # 1. if tuple - treat as min and max values
                        if type(validation) == tuple and len(validation) == 2:
                            if validation[0] is not None and candidate < validation[0]:
                                raise ValueError('{}: {} fails minimum validation check >= {}!'.format(key, candidate, validation[0]))
                            if validation[1] is not None and candidate > validation[1]:
                                raise ValueError('{}: {} fails maximum validation check (<= {})!'.format(key, candidate, validation[1]))

                        # 2. if set - treat as a set of valid values
                        if type(validation) == set:
                            if candidate not in validation:
                                raise ValueError('{}: {} is not an allowed value ({})!'.format(key, candidate, validation))

                        # 3. if both string - treat as a regular expression match
                        if type(validation) == str and dtype == str:
                            if not regex.match(validation, candidate):
                                raise ValueError('{}: {} does not match regex ({})!'.format(key, candidate, validation))

                        # 4. if function - run custom validation code
                        if callable(validation):
                            if not validation(candidate):
                                raise ValueError('{}: {} failed custom validation check!'.format(key, candidate))

                    self._values[key] = candidate

            else:
                raise ValueError('Unexpected parameter: {}!'.format(key))
