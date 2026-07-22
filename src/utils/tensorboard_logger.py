import logging
import torch
from torch.utils.tensorboard import SummaryWriter

class TrafficTensorBoardLogger:
    """
    A fail-safe observer for TensorBoard metrics.
    Ensures that logging errors do not interrupt the main training pipeline.
    """
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.writer = None
        self.logger = logging.getLogger(__name__)
        
        try:
            self.writer = SummaryWriter(log_dir=self.log_dir)
            self.logger.info(f"TensorBoard initialized at: {self.log_dir}")
        except Exception as e:
            self.logger.error(f"Failed to initialize TensorBoard SummaryWriter: {e}")

    def log_scalar(self, tag: str, value: float, step: int):
        if self.writer is None: return
        try:
            self.writer.add_scalar(tag, float(value), step)
        except Exception as e:
            self.logger.debug(f"Failed to log scalar {tag}: {e}")

    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        if self.writer is None or values is None: return
        try:
            self.writer.add_histogram(tag, values.detach().cpu(), step)
        except Exception as e:
            self.logger.debug(f"Failed to log histogram {tag}: {e}")

    def log_image(self, tag: str, img_tensor: torch.Tensor, step: int, dataformats: str = 'CHW'):
        if self.writer is None or img_tensor is None: return
        try:
            self.writer.add_image(tag, img_tensor.detach().cpu(), step, dataformats=dataformats)
        except Exception as e:
            self.logger.debug(f"Failed to log image {tag}: {e}")

    def log_graph(self, model: torch.nn.Module, dummy_input: tuple):
        """
        Attempts to trace the PyTorch computational graph.
        Custom autograd functions (like IMDEquilibriumFunction) or dynamic control flows 
        often break JIT tracing. This wrapper catches those exceptions safely.
        """
        if self.writer is None: return
        try:
            # Suppress tracer warnings specifically for the graph generation
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=torch.jit.TracerWarning)
                self.writer.add_graph(model, dummy_input)
            self.logger.info("Successfully registered computational graph in TensorBoard.")
        except Exception as e:
            self.logger.warning(f"Could not trace model graph for TensorBoard (expected with custom autograd/sparse operations). Error: {e}")

    def close(self):
        if self.writer is not None:
            self.writer.close()