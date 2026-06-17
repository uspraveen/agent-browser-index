"""Modal deployment for MiMo-VL-7B-RL vLLM server.

Deploy:  modal deploy deploy_modal.py --profile uspraveenraj
URL:     https://uspraveenraj--mimo-vl-7b-rl-serve.modal.run
"""
import subprocess
import modal

app = modal.App("mimo-vl-7b-rl")

MODEL_ID = "XiaomiMiMo/MiMo-VL-7B-RL"
HF_CACHE = "/root/.cache/huggingface"

image = (
    modal.Image.from_registry("vllm/vllm-openai:latest", add_python="3.12")
    # vllm/vllm-openai sets ENTRYPOINT ["vllm"], which breaks Modal's Python runner.
    # Clear it so Modal can start its own `python -m modal._container_entrypoint`.
    .dockerfile_commands(["ENTRYPOINT []"])
    .run_commands("pip install hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

hf_vol = modal.Volume.from_name("hf-model-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu="L40S:1",
    region="us-east",
    timeout=7200,
    scaledown_window=300,
    volumes={HF_CACHE: hf_vol},
)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=8000, startup_timeout=600)
def serve():
    subprocess.Popen([
        "vllm", "serve", MODEL_ID,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--gpu-memory-utilization", "0.70",
        "--max-model-len", "8192",
        "--limit-mm-per-prompt", '{"image":1,"video":0}',
    ])
