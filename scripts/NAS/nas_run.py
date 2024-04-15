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
import tempfile
from ax.runners.torchx import TorchXRunner
from ax.core import (
    ChoiceParameter,
    ParameterType,
    RangeParameter,
    FixedParameter,
)
from ax.core.search_space import HierarchicalSearchSpace
from ax.metrics.tensorboard import TensorboardCurveMetric
from ax.core import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.core import Experiment
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.service.scheduler import Scheduler, SchedulerOptions
from ax.service.utils.report_utils import exp_to_df

curDir = os.getcwd()

def trainer(trial_idx: int = -1, *args, **kwargs) -> specs.AppDef:
    # define the log path so we can pass it to the TorchX ``AppDef``
    if trial_idx >= 0:
        kwargs['log_path'] = Path(kwargs['log_path']).joinpath(str(trial_idx)).absolute().as_posix()
    values = []
    for key, value in kwargs.items():
        if key in ['root']: continue
        if key == 'log_path':
            values += [f'--{key}', f'{value}']
            continue
        if value: values += [f'--{key}']

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


# Make a temporary dir to log our results into
log_dir = tempfile.mkdtemp(prefix='logs_zDELETE', dir=curDir)

ax_runner = TorchXRunner(
    tracker_base="/tmp/",
    component=trainer,
    # NOTE: To launch this job on a cluster instead of locally you can
    # specify a different scheduler and adjust arguments appropriately.
    scheduler="local_cwd",
    component_const_params={"log_path": log_dir},
    cfg={},
)

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
            "insertResidualAfterL1",
            "insertResidualAfterL2",
            "addMRNA_during",
            "addMRNA_after",
        ]}
    ),
    ChoiceParameter(
        name="insertResidualAfterL1",
        values=[True, False],
        parameter_type=ParameterType.BOOL,
        dependents={
            True:[
                "l1i_category0",
                "l1i_category1",
                "l1i_category2",
                "l1i_category3",
            ],
            False:[],
        }
    ),
    ChoiceParameter(
        name="l1i_category0",
        values=[True, False],
        parameter_type=ParameterType.BOOL,
    ),
    ChoiceParameter(
        name="l1i_category1",
        values=[True, False],
        parameter_type=ParameterType.BOOL,
    ),
    ChoiceParameter(
        name="l1i_category2",
        values=[True, False],
        parameter_type=ParameterType.BOOL,
    ),
    ChoiceParameter(
        name="l1i_category3",
        values=[True, False],
        parameter_type=ParameterType.BOOL,
    ),
    ChoiceParameter(
        name="insertResidualAfterL2",
        values=[True, False],
        parameter_type=ParameterType.BOOL,
        dependents={
            True: [
                "l2i_category0",
                "l2i_category1",
                "l2i_category2",
                "l2i_category3",
            ],
            False: [],
        }
    ),
    ChoiceParameter(
        name="l2i_category0",
        values=[True, False],
        parameter_type=ParameterType.BOOL,
    ),
    ChoiceParameter(
        name="l2i_category1",
        values=[True, False],
        parameter_type=ParameterType.BOOL,
    ),
    ChoiceParameter(
        name="l2i_category2",
        values=[True, False],
        parameter_type=ParameterType.BOOL,
    ),
    ChoiceParameter(
        name="l2i_category3",
        values=[True, False],
        parameter_type=ParameterType.BOOL,
    ),
    ChoiceParameter(
        name="addMRNA_during",
        values=[True, False],
        parameter_type=ParameterType.BOOL,
    ),
    ChoiceParameter(
        name="addMRNA_after",
        values=[True, False],
        parameter_type=ParameterType.BOOL,
    ),
]

search_space = HierarchicalSearchSpace(
    parameters=parameters,
    parameter_constraints=[],
)


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



opt_config = OptimizationConfig(
    objective=Objective(metric=val_loss, minimize=True)
)

experiment = Experiment(
    name="torchx_cptac",
    search_space=search_space,
    optimization_config=opt_config,
    runner=ax_runner,
)

total_trials = 20  # total evaluation budget


gs = choose_generation_strategy(
    search_space=experiment.search_space,
    optimization_config=experiment.optimization_config,
    num_trials=total_trials,
  )


scheduler = Scheduler(
    experiment=experiment,
    generation_strategy=gs,
    options=SchedulerOptions(
        total_trials=total_trials, max_pending_trials=10
    ),
)

scheduler.run_all_trials()

df = exp_to_df(experiment).sort_values('val_loss', ascending=True)
df.to_csv(os.path.join(curDir, f'sample_dataset_nas_output_input_residual{time.strftime("%Y%m%d-%H%M%S")}.csv'), index=False)
