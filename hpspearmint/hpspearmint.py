##
# Copyright (C) 2012 Jasper Snoek, Hugo Larochelle and Ryan P. Adams
#                                                                   
# This code is written for research and educational purposes only to
# supplement the paper entitled "Practical Bayesian Optimization of
# Machine Learning Algorithms" by Snoek, Larochelle and Adams Advances
# in Neural Information Processing Systems, 2012
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see
# <http://www.gnu.org/licenses/>.

"""
This file is derived from the file:
    spearmint/spearmint-lite/spearmint-lite.py
in the Spearmint project by Jasper Snoek.

"""

import numpy as np

import hyperopt
from hyperopt.pyll_utils import expr_to_config
from hyperopt.base import miscs_update_idxs_vals

if 0:
    import logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    info = logger.info
else:
    def info(msg):
        print msg


try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

from spearmint.ExperimentGrid  import GridMap

#
# There are two things going on here.  There are "experiments", which are
# large-scale things that live in a directory and in this case correspond
# to the task of minimizing a complicated function.  These experiments
# contain "jobs" which are individual function evaluations.  The set of
# all possible jobs, regardless of whether they have been run or not, is
# the "grid".  This grid is managed by an instance of the class
# ExperimentGrid.

def variables_from_domain(domain):
    rval = OrderedDict()
    hps = OrderedDict()
    expr_to_config(domain.expr, (True,), hps)
    for key, val in hps.items():
        if val['conditions'] != set([(True,)]):
            raise NotImplementedError('Conditional parameter: %s' % key,
                                     val['conditions'])

        if val['node'].name == 'uniform':
            low = val['node'].arg['low'].obj
            high = val['node'].arg['high'].obj
            rval[key] = {
                'name': key,
                'type': 'float',
                'size': 1,
                'min': low,
                'max': high,
                # -- following just used in this file
                'to_unit': lambda x: (x - low) / (high - low),
                'from_unit': lambda y: y * (high - low) + low,
                        }
        elif val['node'].name == 'loguniform':
            raise NotImplementedError()
        elif val['node'].name == 'randint':
            raise NotImplementedError()
        elif val['node'].name == 'normal':
            raise NotImplementedError()
        elif val['node'].name == 'lognormal':
            raise NotImplementedError()
        else:
            raise NotImplementedError('Unsupported hyperparamter type: %s' %
                                     val['node'].name)
    return rval


def trial_duration(trial, default_time=None):
    try:
        return float(trial['result']['duration'])
    except KeyError:
        try:
            datetime_diff = trial['refresh_time'] - trial['book_time']
        except TypeError:
            if default_time is None:
                raise
            else:
                return default_time
        return datetime_diff.total_seconds()


def unit_assignment(trial, variables):
    return [var['to_unit'](trial['misc']['vals'][key][0])
            for key, var in variables.items()]


def unit_to_list(candidate, variables):
    assert len(candidate) == len(variables)
    return [var['from_unit'](c) for var, c in zip(variables.values(), candidate)]


def suggest(new_ids, domain, trials,
            chooser,
            grid_size, grid_seed,
            expt_dir,  # -- a state object will be maintained here
            verbose=0,
           ):

    variables = variables_from_domain(domain)
    gmap = GridMap(variables.values(), grid_size)

    values = []
    complete = []
    pending =  []
    durations = []

    for trial in trials.trials:
        # Each line in this file represents an experiment
        # It is whitespace separated and of the form either
        # <Value> <time taken> <space separated list of parameters>
        # incating a completed experiment or
        # P P <space separated list of parameters>
        # indicating a pending experiment

        state = trial['state']
        status = trial['result']['status']
        val = trial['result'].get('loss')
        dur = trial_duration(trial)
        unit_vals = unit_assignment(trial, variables)

        if state in (hyperopt.JOB_STATE_NEW, hyperopt.JOB_STATE_RUNNING):
            pending.append(unit_vals)
        elif state in (hyperopt.JOB_STATE_DONE,):
            if status in hyperopt.STATUS_OK:
                complete.append(unit_vals)
                durations.append(dur)
                values.append(val)

    # Some stats
    info("#Complete: %d #Pending: %d" % (len(complete), len(pending)))

    # Let's print out the best value so far
    if len(values):
        best_val = np.min(values)
        best_job = np.argmin(values)
        info("Current best: %f (job %d)" % (best_val, best_job))

    # Now lets get the next job to run
    # First throw out a set of candidates on the unit hypercube
    # Increment by the number of observed so we don't take the
    # same values twice
    seed_increment = len(pending) + len(complete)
    candidates = gmap.hypercube_grid(grid_size, grid_seed + seed_increment)

    # Ask the chooser to actually pick one.
    # First mash the data into a format that matches that of the other
    # spearmint drivers to pass to the chooser modules.        

    grid = np.asarray(complete + list(candidates) + pending)
    grid_idx = np.hstack((np.zeros(len(complete)),
                          np.ones(len(candidates)),
                          1.0 + np.ones(len(pending))))
    chosen = chooser.next(grid, np.asarray(values), np.asarray(durations),
                          np.nonzero(grid_idx == 1)[0],
                          np.nonzero(grid_idx == 2)[0],
                          np.nonzero(grid_idx == 0)[0])

    # If the chosen is a tuple, then the chooser picked a new job not from
    # the candidate list
    if isinstance(chosen, tuple):
        (chosen, candidate) = chosen
    else:
        candidate = grid[chosen]

    info("Selected job %d from the grid." % (chosen,))

    params = unit_to_list(candidate, variables)

    if len(new_ids) > 1:
        raise NotImplementedError('TODO: recurse for multiple jobs')

    rval = []
    for new_id in new_ids:
        idxs = dict([(v, [new_id]) for v in variables])
        vals = dict([(v, [p]) for v, p in zip(variables, params)])
        new_result = domain.new_result()
        new_misc = dict(tid=new_id, cmd=domain.cmd, workdir=domain.workdir)
        miscs_update_idxs_vals([new_misc], idxs, vals)
        rval.extend(trials.new_trial_docs([new_id],
                [None], [new_result], [new_misc]))

    return rval

    
