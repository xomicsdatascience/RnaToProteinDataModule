'''
import logging
import sys
import io
import time
import subprocess

class StreamToLogger(io.TextIOBase):
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

def patch_logger_adapter(logger_adapter, logger, level=logging.INFO):
    original_info = logger_adapter.info
    def patched_info(message, *args, **kwargs):
        logger.log(level, message)
        original_info(message, *args, **kwargs)
    logger_adapter.info = patched_info

# Configure the logging module
logging.basicConfig(filename=f'logs/overall_output_{time.strftime("%Y%m%d-%H%M%S")}.log', level=logging.INFO)

# Redirect stdout to the logger
stdout_logger = logging.getLogger('STDOUT')
sys.stdout = StreamToLogger(stdout_logger, logging.INFO)

# Redirect stderr to the logger
stderr_logger = logging.getLogger('STDERR')
sys.stderr = StreamToLogger(stderr_logger, logging.ERROR)
#'''

from pathlib import Path

import torchx

from torchx import specs
from torchx.components import utils
import time
import os

curDir = os.getcwd()

def trainer(trial_idx: int = -1, *args, **kwargs) -> specs.AppDef:
    # define the log path so we can pass it to the TorchX ``AppDef``
    if trial_idx >= 0:
        kwargs['log_path'] = Path(kwargs['log_path']).joinpath(str(trial_idx)).absolute().as_posix()
    values = []
    for key, value in kwargs.items():
        if key in ['root']: continue
        if key in ['block1_exists', 'block2_exists', 'addMRNA', 'onlyCodingTranscripts', "removeCoad", "useUnsharedTranscripts"]:
            if value: values += [f'--{key}']
            continue
        values += [f'--{key}', str(value)]

    output = utils.python(
        *values,
        # command line arguments to the training script
        # other config options
        name="trainer",
        script=os.path.join(curDir, "nas_model_running_script.py"),
        image=torchx.version.TORCHX_IMAGE,
    )
    #logging.info(output)
    outputStr = ['python']+output.roles[0].args
    print(' '.join(outputStr))
    #subprocess.run(['python']+output.roles[0].args)
    return output

import tempfile
from ax.runners.torchx import TorchXRunner

# Make a temporary dir to log our results into
log_dir = tempfile.mkdtemp(prefix='logs', dir=curDir)

ax_runner = TorchXRunner(
    tracker_base="/tmp/",
    component=trainer,
    # NOTE: To launch this job on a cluster instead of locally you can
    # specify a different scheduler and adjust arguments appropriately.
    scheduler="local_cwd",
    component_const_params={"log_path": log_dir},
    cfg={},
)

from ax.core import (
    ChoiceParameter,
    ParameterType,
    RangeParameter,
    FixedParameter,
)
from ax.core.search_space import HierarchicalSearchSpace

parameters = [
    # NOTE: In a real-world setting, hidden_size_1 and hidden_size_2
    # should probably be powers of 2, but in our simple example this
    # would mean that ``num_params`` can't take on that many values, which
    # in turn makes the Pareto frontier look pretty weird.

    FixedParameter(
        name="root",
        value='true',
        parameter_type=ParameterType.STRING,
        dependents={'true':[
            "block1_exists",
            "block2_exists",
            "block3_type",
            "activation3",
            "dropout3",
            "addMRNA",
            "learning_rate",
            "batch_size",
            "onlyCodingTranscripts",
            "removeCoad",
        ]}
    ),
    ChoiceParameter(
        name="block1_exists",
        values=[True, False],
        parameter_type=ParameterType.BOOL,
        dependents={
            True:[
                "block1_type",
                "hidden_size1",
                "activation1",
                "dropout1",
            ],
            False:[],
        }
    ),
    ChoiceParameter(
        name="block1_type",
        values=["fully_connect", "resnet"],
        parameter_type=ParameterType.STRING,
        dependents={
            "fully_connect":[
                "fc1",
            ],
            "resnet":[
                "resNetType1",
            ]
        }
    ),
    RangeParameter(
        name="fc1",
        lower=1,
        upper=3,
        parameter_type=ParameterType.INT,
    ),
    ChoiceParameter(
        name="resNetType1",
        values=["simple", "complex"],
        parameter_type=ParameterType.STRING,
        dependents={
            "simple":[],
            "complex":[
                "resNetComplexConnections1",
            ],
        }
    ),
    RangeParameter(
        name="resNetComplexConnections1",
        lower=1,
        upper=31,
        parameter_type=ParameterType.INT,
    ),
    RangeParameter(
        name="hidden_size1",
        lower=200,
        upper=1200,
        parameter_type=ParameterType.INT,
        log_scale=True,
    ),
    ChoiceParameter(
        name="activation1",
        values=["leaky_relu_steep", "leaky_relu_slight", "sigmoid", "tanh", "selu"],
        parameter_type=ParameterType.STRING,
        is_ordered=False,
        sort_values=False,
    ),
    RangeParameter(
        name="dropout1",
        lower=0.0,
        upper=0.9,
        parameter_type=ParameterType.FLOAT,
    ),
    ChoiceParameter(
        name="block2_exists",
        values=[True, False],
        parameter_type=ParameterType.BOOL,
        dependents={
            True:[
                "block2_type",
                "hidden_size2",
                "activation2",
                "dropout2",
            ],
            False:[],
        }
    ),
    ChoiceParameter(
        name="block2_type",
        values=["fully_connect", "resnet"],
        parameter_type=ParameterType.STRING,
        dependents={
            "fully_connect":[
                "fc2",
            ],
            "resnet":[
                "resNetType2",
            ]
        }
    ),
    RangeParameter(
        name="fc2",
        lower=1,
        upper=3,
        parameter_type=ParameterType.INT,
    ),
    ChoiceParameter(
        name="resNetType2",
        values=["simple", "complex"],
        parameter_type=ParameterType.STRING,
        dependents={
            "simple":[],
            "complex":[
                "resNetComplexConnections2",
            ],
        }
    ),
    RangeParameter(
        name="resNetComplexConnections2",
        lower=1,
        upper=31,
        parameter_type=ParameterType.INT,
    ),
    RangeParameter(
        name="hidden_size2",
        lower=200,
        upper=900,
        parameter_type=ParameterType.INT,
        log_scale=True,
    ),
    ChoiceParameter(
        name="activation2",
        values=["leaky_relu_steep", "leaky_relu_slight", "sigmoid", "tanh", "selu"],
        parameter_type=ParameterType.STRING,
        is_ordered=False,
        sort_values=False,
    ),
    RangeParameter(
        name="dropout2",
        lower=0.0,
        upper=0.9,
        parameter_type=ParameterType.FLOAT,
    ),
    ChoiceParameter(
        name="block3_type",
        values=["fully_connect", "resnet"],
        parameter_type=ParameterType.STRING,
        is_ordered=False,
        sort_values=False,
        dependents={
            "fully_connect": [
                "fc3",
            ],
            "resnet": [
            ]
        }
    ),
    RangeParameter(
        name="fc3",
        lower=1,
        upper=3,
        parameter_type=ParameterType.INT,
    ),
    ChoiceParameter(
        name="activation3",
        values=["leaky_relu_steep", "leaky_relu_slight", "sigmoid", "tanh", "selu"],
        parameter_type=ParameterType.STRING,
        is_ordered=False,
    ),
    RangeParameter(
        name="dropout3",
        lower=0.0,
        upper=0.9,
        parameter_type=ParameterType.FLOAT,
    ),
    ChoiceParameter(
        name="addMRNA",
        values=[True, False],
        parameter_type=ParameterType.BOOL,
    ),
    ChoiceParameter(
        name="onlyCodingTranscripts",
        values=[True, False],
        parameter_type=ParameterType.BOOL,
        dependents={
            True: [],
            False: ['useUnsharedTranscripts'],
        }
    ),
    ChoiceParameter(
        name="useUnsharedTranscripts",
        values=[True, False],
        parameter_type=ParameterType.BOOL,
    ),
    ChoiceParameter(
        name="removeCoad",
        values=[True, False],
        parameter_type=ParameterType.BOOL,
    ),
    RangeParameter(
        name="learning_rate",
        lower=1e-4,
        upper=1e-2,
        parameter_type=ParameterType.FLOAT,
        log_scale=True,
    ),
    ChoiceParameter(
        name="batch_size",
        values=[32, 64, 128],
        parameter_type=ParameterType.INT,
        is_ordered=True,
        sort_values=True,
    ),
]

search_space = HierarchicalSearchSpace(
    parameters=parameters,
    parameter_constraints=[],
)

from ax.metrics.tensorboard import TensorboardCurveMetric

class MyTensorboardMetric(TensorboardCurveMetric):

    # NOTE: We need to tell the new TensorBoard metric how to get the id /
    # file handle for the TensorBoard logs from a trial. In this case
    # our convention is to just save a separate file per trial in
    # the prespecified log dir.
    @classmethod
    def get_ids_from_trials(cls, trials):
        return {
            trial.index: Path(log_dir).joinpath(str(trial.index)).as_posix()
            for trial in trials
        }

    # This indicates whether the metric is queryable while the trial is
    # still running. We don't use this in the current tutorial, but Ax
    # utilizes this to implement trial-level early-stopping functionality.
    @classmethod
    def is_available_while_running(cls):
        return False

val_loss = MyTensorboardMetric(
    name="val_loss",
    curve_name="val_loss",
    lower_is_better=True,
)

from ax.core import Objective
from ax.core.optimization_config import OptimizationConfig

opt_config = OptimizationConfig(
    objective=Objective(metric=val_loss, minimize=True)
)

from ax.core import Experiment

experiment = Experiment(
    name="torchx_cptac",
    search_space=search_space,
    optimization_config=opt_config,
    runner=ax_runner,
)

total_trials = 30  # total evaluation budget

from ax.modelbridge.dispatch_utils import choose_generation_strategy

gs = choose_generation_strategy(
    search_space=experiment.search_space,
    optimization_config=experiment.optimization_config,
    num_trials=total_trials,
  )

from ax.service.scheduler import Scheduler, SchedulerOptions

scheduler = Scheduler(
    experiment=experiment,
    generation_strategy=gs,
    options=SchedulerOptions(
        total_trials=total_trials, max_pending_trials=8
    ),
)

scheduler.run_all_trials()

from ax.service.utils.report_utils import exp_to_df

df = exp_to_df(experiment).sort_values('val_loss', ascending=True)
df.to_csv(os.path.join(curDir, f'sample_dataset_nas_output_{time.strftime("%Y%m%d-%H%M%S")}.csv'), index=False)
