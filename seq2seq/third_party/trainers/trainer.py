from typing import Dict, List, Optional
import numpy as np 
import time
import torch
from packaging import version
from torch.utils.data.dataset import Dataset

from transformers import Trainer
from transformers import logging
from transformers.trainer_utils import speed_metrics
from transformers.file_utils import is_torch_tpu_available


if version.parse(torch.__version__) >= version.parse("1.6"):
    from torch.cuda.amp import autocast

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


logger = logging.get_logger(__name__)

class BaseTrainer(Trainer):
    def __init__(self, evaluation_metrics=[], *args, **kwargs):
        """When doing evaluation, it computes average of list of metrics 
        given in evaluation_metrics and adds it to the dictionary of results.
        Trainer class then use this average metric to save the best model."""
        super().__init__(*args, **kwargs)
        self.evaluation_metrics = evaluation_metrics 

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.
        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init :obj:`compute_metrics` argument).
        You can also subclass and override this method to inject custom behavior.
        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement the
                :obj:`__len__` method.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (:obj:`str`, `optional`, defaults to :obj:`"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)
        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        output.metrics.update(speed_metrics(metric_key_prefix, start_time, output.num_samples))

        # TODO: this for now only supports one dataset.
        if len(self.evaluation_metrics) != 0:
           selected_metrics = [output.metrics[metric_key_prefix+"_"+k] for k in self.evaluation_metrics if metric_key_prefix+"_"+k in output.metrics]
           assert len(selected_metrics) >= 1, "at least one metric should be selected to compute the average_metrics."
           output.metrics.update({metric_key_prefix+'_average_metrics': np.mean(selected_metrics)})         
    
        self.log(output.metrics)

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics
