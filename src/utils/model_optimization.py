"""
Model Optimization and Quantization.
Provides tools for optimizing models for production deployment.
"""

import torch
from typing import Optional, Literal, Dict
from pathlib import Path
import time

from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from .logger import get_logger
from .config_loader import get_config

logger = get_logger(__name__)


class ModelOptimizer:
    """
    Optimize models for production deployment.
    """

    def __init__(self):
        """Initialize model optimizer."""
        self.config = get_config()
        logger.info("Model optimizer initialized")

    def quantize_model(
        self,
        model: torch.nn.Module,
        quantization_type: Literal['dynamic', 'static', '8bit', '4bit'] = 'dynamic'
    ) -> torch.nn.Module:
        """
        Quantize model for faster inference.

        Args:
            model: Model to quantize
            quantization_type: Type of quantization

        Returns:
            Quantized model
        """
        logger.info(f"Quantizing model with {quantization_type} quantization...")

        try:
            if quantization_type == 'dynamic':
                # Dynamic quantization (easiest, good for CPU)
                quantized_model = torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )

            elif quantization_type == 'static':
                # Static quantization (requires calibration)
                model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                torch.quantization.prepare(model, inplace=True)
                # Would need calibration data here
                torch.quantization.convert(model, inplace=True)
                quantized_model = model

            elif quantization_type == '8bit':
                # 8-bit quantization using bitsandbytes
                # Model must be loaded with load_in_8bit=True
                logger.info("8-bit quantization requires model to be loaded with appropriate config")
                quantized_model = model

            elif quantization_type == '4bit':
                # 4-bit quantization
                logger.info("4-bit quantization requires model to be loaded with appropriate config")
                quantized_model = model

            else:
                logger.warning(f"Unknown quantization type: {quantization_type}")
                return model

            logger.info("Model quantization complete")
            return quantized_model

        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            return model

    def get_quantization_config(
        self,
        bits: Literal[4, 8] = 8,
        compute_dtype: str = 'float16'
    ) -> BitsAndBytesConfig:
        """
        Get configuration for quantized model loading.

        Args:
            bits: Number of bits (4 or 8)
            compute_dtype: Compute dtype

        Returns:
            BitsAndBytes configuration
        """
        dtype_map = {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'float32': torch.float32
        }

        dtype = dtype_map.get(compute_dtype, torch.float16)

        if bits == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif bits == 8:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True
            )
        else:
            logger.warning(f"Unsupported bits: {bits}")
            return None

    def measure_inference_speed(
        self,
        model: torch.nn.Module,
        input_data: torch.Tensor,
        num_runs: int = 100
    ) -> Dict:
        """
        Measure model inference speed.

        Args:
            model: Model to measure
            input_data: Sample input
            num_runs: Number of runs for averaging

        Returns:
            Performance metrics
        """
        model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(input_data)

        # Measure
        times = []

        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                _ = model(input_data)
                end = time.time()
                times.append(end - start)

        avg_time = sum(times) / len(times)
        throughput = 1.0 / avg_time

        return {
            'avg_inference_time_ms': avg_time * 1000,
            'throughput_samples_per_sec': throughput,
            'min_time_ms': min(times) * 1000,
            'max_time_ms': max(times) * 1000
        }

    def optimize_for_inference(
        self,
        model: torch.nn.Module,
        device: str = 'cpu'
    ) -> torch.nn.Module:
        """
        Apply various optimizations for inference.

        Args:
            model: Model to optimize
            device: Target device

        Returns:
            Optimized model
        """
        logger.info("Optimizing model for inference...")

        # Set to eval mode
        model.eval()

        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False

        # JIT compile if possible
        try:
            # Note: Not all models support JIT compilation
            # model = torch.jit.script(model)
            # logger.info("Model JIT compiled")
            pass
        except Exception as e:
            logger.warning(f"JIT compilation failed: {e}")

        # Move to device
        model = model.to(device)

        # Use inference mode context
        # This is typically done at inference time, not model preparation

        logger.info("Model optimization complete")
        return model

    def get_model_size(self, model: torch.nn.Module) -> Dict:
        """
        Get model size information.

        Args:
            model: Model to analyze

        Returns:
            Size information
        """
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        total_size = param_size + buffer_size
        size_mb = total_size / (1024 ** 2)

        num_params = sum(p.numel() for p in model.parameters())

        return {
            'total_size_mb': size_mb,
            'param_size_mb': param_size / (1024 ** 2),
            'buffer_size_mb': buffer_size / (1024 ** 2),
            'num_parameters': num_params,
            'num_parameters_millions': num_params / 1e6
        }


class ModelCache:
    """
    Manage model caching for faster loading.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize model cache.

        Args:
            cache_dir: Directory for cached models
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "emotion_ai" / "models"

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Model cache initialized at: {self.cache_dir}")

    def get_cache_path(self, model_name: str) -> Path:
        """
        Get cache path for a model.

        Args:
            model_name: Model identifier

        Returns:
            Cache path
        """
        # Sanitize model name for filesystem
        safe_name = model_name.replace("/", "_").replace("\\", "_")
        return self.cache_dir / safe_name

    def is_cached(self, model_name: str) -> bool:
        """
        Check if model is cached.

        Args:
            model_name: Model identifier

        Returns:
            True if cached
        """
        cache_path = self.get_cache_path(model_name)
        return cache_path.exists() and any(cache_path.iterdir())

    def clear_cache(self, model_name: Optional[str] = None):
        """
        Clear model cache.

        Args:
            model_name: Specific model to clear (None for all)
        """
        if model_name:
            cache_path = self.get_cache_path(model_name)
            if cache_path.exists():
                import shutil
                shutil.rmtree(cache_path)
                logger.info(f"Cleared cache for: {model_name}")
        else:
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                logger.info("Cleared all model cache")


class InferenceOptimizer:
    """
    Optimize inference pipeline.
    """

    def __init__(self):
        """Initialize inference optimizer."""
        self.config = get_config()
        logger.info("Inference optimizer initialized")

    def batch_inputs(
        self,
        inputs: list,
        batch_size: int = 8
    ) -> list:
        """
        Batch inputs for efficient processing.

        Args:
            inputs: List of inputs
            batch_size: Batch size

        Returns:
            List of batches
        """
        batches = []

        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            batches.append(batch)

        return batches

    def optimize_tokenization(
        self,
        tokenizer,
        use_fast: bool = True
    ):
        """
        Optimize tokenizer settings.

        Args:
            tokenizer: Tokenizer instance
            use_fast: Use fast tokenizer if available
        """
        # Enable padding and truncation
        tokenizer.padding_side = 'right'
        tokenizer.truncation_side = 'right'

        # Use fast tokenizer if available
        if use_fast and hasattr(tokenizer, 'is_fast'):
            if not tokenizer.is_fast:
                logger.warning("Fast tokenizer not available")

        return tokenizer

    def enable_torch_optimizations(self):
        """Enable PyTorch optimizations."""
        # Enable cudnn benchmarking for faster convolutions
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            logger.info("Enabled cudnn benchmark mode")

        # Enable TF32 on Ampere GPUs
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("Enabled TF32 for Ampere+ GPUs")

        # Set number of threads for CPU
        if not torch.cuda.is_available():
            torch.set_num_threads(torch.get_num_threads())
            logger.info(f"Using {torch.get_num_threads()} CPU threads")


def benchmark_model(
    model_name: str,
    sample_input: str = "This is a test input for benchmarking."
) -> Dict:
    """
    Benchmark a model's performance.

    Args:
        model_name: Model to benchmark
        sample_input: Sample input for testing

    Returns:
        Benchmark results
    """
    logger.info(f"Benchmarking model: {model_name}")

    try:
        from transformers import pipeline

        # Load model
        start_load = time.time()
        nlp = pipeline("text-classification", model=model_name)
        load_time = time.time() - start_load

        # Warmup
        for _ in range(5):
            _ = nlp(sample_input)

        # Benchmark inference
        times = []
        for _ in range(50):
            start = time.time()
            _ = nlp(sample_input)
            times.append(time.time() - start)

        avg_time = sum(times) / len(times)

        results = {
            'model_name': model_name,
            'load_time_seconds': load_time,
            'avg_inference_time_ms': avg_time * 1000,
            'min_inference_time_ms': min(times) * 1000,
            'max_inference_time_ms': max(times) * 1000,
            'throughput_samples_per_sec': 1.0 / avg_time
        }

        logger.info(f"Benchmark complete: {avg_time * 1000:.2f}ms avg")
        return results

    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return {'error': str(e)}
