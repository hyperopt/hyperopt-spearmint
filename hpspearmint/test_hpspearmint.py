from functools import partial
import os
import hpspearmint
from spearmint.RandomChooser import RandomChooser
from spearmint.GPEIChooser import GPEIChooser

from hyperopt import fmin, hp, Trials

def test_quadratic1_tpe():
    trials = Trials()

    suggest = partial(hpspearmint.suggest,
                      chooser=RandomChooser(),
                      grid_size=10,
                      grid_seed=0,
                      expt_dir=None,
                      verbose=1)

    argmin = fmin(
            fn=lambda x: (x - 3) ** 2,
            space=hp.uniform('x', -5, 5),
            algo=suggest,
            max_evals=50,
            trials=trials)

    print argmin
    assert len(trials) == 50, len(trials)
    assert abs(argmin['x'] - 3.0) < .25, argmin


def test_quadratic5_tpe():
    trials = Trials()

    # XXX: I wish I could delete this file after it has been written
    #      rather than here
    try:
        os.remove('spearmint.GPEIChooser.pkl')
    except IOError:
        pass

    chooser=GPEIChooser(expt_dir=os.getcwd())

    suggest = partial(hpspearmint.suggest,
                      chooser=chooser,
                      grid_size=1000,
                      grid_seed=0,
                      expt_dir=None,
                      verbose=1)

    argmin = fmin(
            fn=lambda xs: sum((x - 3) ** 2 for x in xs),
            space=[
                hp.uniform('x0', -5, 5),
                hp.uniform('x1', -5, 5),
                hp.uniform('x2', -5, 5),
                hp.uniform('x3', -5, 5),
                hp.uniform('x4', -5, 5),
                  ],
            algo=suggest,
            max_evals=50,
            trials=trials)

    print argmin
    assert len(trials) == 50, len(trials)
    assert abs(argmin['x0'] - 3.0) < .25, argmin

