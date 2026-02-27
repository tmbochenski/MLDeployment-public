from pathlib import Path

from optimum.onnxruntime import ORTModelForSequenceClassification, ORTOptimizer
from optimum.onnxruntime.configuration import OptimizationConfig

# --- Configuration ---
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
OUTPUT_DIR = Path("models")


def _safe_model_id(model_name: str) -> str:
    return model_name.replace("/", "_")


def main() -> None:
    safe_name = _safe_model_id(MODEL_NAME)

    fp32_model_dir = OUTPUT_DIR / f"{safe_name}_fp32"
    fp16_model_dir = OUTPUT_DIR / f"{safe_name}_fp16"

    print(f"--- Starting ONNX Conversion for '{MODEL_NAME}' ---")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")

    # Step 1: Export FP32 ONNX model
    print("\n--- Step 1: Exporting FP32 ONNX model ---")
    fp32_model = ORTModelForSequenceClassification.from_pretrained(
        MODEL_NAME, export=True
    )
    fp32_model.save_pretrained(fp32_model_dir)
    print(f"FP32 ONNX model saved to '{fp32_model_dir.as_posix()}'")

    # Step 2: Optimize to FP16
    print("\n--- Step 2: Optimizing model to FP16 ---")
    optimizer = ORTOptimizer.from_pretrained(fp32_model_dir)
    optimization_config = OptimizationConfig(
        optimization_level=1,
        fp16=True,
        disable_attention_fusion=True,
        disable_gelu_fusion=True,
        disable_skip_layer_norm_fusion=True,
    )
    optimizer.optimize(save_dir=fp16_model_dir, optimization_config=optimization_config)
    print(f"FP16 optimized model saved to '{fp16_model_dir.as_posix()}'")
    print("\n--- ONNX Conversion Complete ---")


if __name__ == "__main__":
    main()
