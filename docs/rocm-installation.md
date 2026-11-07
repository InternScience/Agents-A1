# Run Sage-A1.5 on AMD ROCm

This guide deploys the full multimodal Sage-A1.5 model on AMD GPUs using
vLLM or llama.cpp. A compatible AMD GPU driver and Docker are required. See the
[ROCm compatibility matrix](https://rocm.docs.amd.com/en/docs-10.0.0/compatibility/compatibility-matrix.html)
for supported driver and ROCm combinations.

Find your GPU target name before you start; the build steps below need it:

```bash
rocminfo | grep gfx
```

## Choosing a configuration

The table below provides reference configurations and benchmark results
for Sage-A1.5 to help you choose a setup that fits your hardware and available
memory.

Reference text-only throughput is reported in tok/s.

| Configuration | Model and weight size | AMD Radeon Pro W7900D (48 GB, gfx1100) | AMD Radeon AI PRO R9700 (32 GB, gfx1201) | AMD Ryzen AI Max+ 395 (128 GB unified memory, gfx1151) | AMD Instinct MI355X (288 GB, gfx950) |
|---|---|---|---|---|---|
| llama.cpp Q4_K_M + 262K context | [InternScience/Sage-A1.5-Q4_K_M-GGUF](https://huggingface.co/InternScience/Sage-A1.5-Q4_K_M-GGUF) — 21.71 GB + 0.90 GB projector | 103.94 | 100.36 | 66.12 | 180.44 |
| llama.cpp Q8_0 + 262K context | [InternScience/Sage-A1.5-Q8_0-GGUF](https://huggingface.co/InternScience/Sage-A1.5-Q8_0-GGUF) — 37.80 GB + 0.90 GB projector | 82.23 | 83.04† | 48.44 | 173.97 |
| vLLM BF16 + 262K context | [InternScience/Sage-A1.5](https://huggingface.co/InternScience/Sage-A1.5) — 71.91 GB | 194.25* | 193.95* | 60.20 | 582.73 |
| vLLM FP8 + 262K context | [InternScience/Sage-A1.5-FP8](https://huggingface.co/InternScience/Sage-A1.5-FP8) — 39.37 GB | FP8 backend unavailable | 137.93* | FP8 backend unavailable | 552.87‡ |

\* Uses four GPUs; † uses two GPUs; all other results use one GPU.

‡ For MI355X FP8 with AITER enabled (`VLLM_ROCM_USE_AITER=1`), add `--compilation-config '{"pass_config":{"fuse_norm_quant":false}}'` to the `vllm serve` command as a workaround for a potential numerical accuracy issue with this image.

<details>
<summary>Benchmark commands</summary>

Run these benchmarks in the respective containers, using Sage-A1.5 in the
server commands below. The examples use BF16 and Q8_0; change the model
identifiers for other configurations.

**vLLM:** Start the corresponding model with the hardware-specific command
below, then run in another shell. Use `Output token throughput (tok/s)`.

```bash
vllm bench serve \
  --model InternScience/Sage-A1.5 \
  --host 127.0.0.1 --port 8000 \
  --dataset-name random \
  --random-input-len 128 \
  --random-output-len 512 \
  --num-prompts 32 \
  --max-concurrency 4 \
  --num-warmups 3
```

**llama.cpp:** If `llama-server` is running on the same GPU(s), stop it before
benchmarking. Use the `tg512` row's `t/s` value.

```bash
HIP_VISIBLE_DEVICES=0 \
llama-bench \
  --hf-repo InternScience/Sage-A1.5-Q8_0-GGUF:Q8_0 \
  -p 128 -n 512 \
  -r 5
```

For R9700 Q8_0, use `HIP_VISIBLE_DEVICES=0,1` and add `--split-mode layer`.

</details>

> [!NOTE]
> **AMD Ryzen AI Max (RDNA 3.5) memory configuration**
>
> On AMD Ryzen AI Max (RDNA 3.5) systems, ensure sufficient GPU-accessible
> memory for the model and context size. See AMD's
> [system optimization guide](https://rocm.docs.amd.com/en/latest/reference/system-optimization/rdna3-5.html)
> for kernel requirements and memory configuration instructions.
>
> Choose an option and adjust the memory allocation to suit your GPU memory
> requirements:
>
> - Reserve sufficient VRAM in the BIOS, for example 96 GB.
> - Use a small BIOS VRAM reservation, for example 0.5 GB, and increase the
>   GTT limit sufficiently, for example to 100 GB.
>
> Neither configuration requires setting llama.cpp's
> [`GGML_CUDA_ENABLE_UNIFIED_MEMORY`](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#unified-memory-2)
> environment variable.

## vLLM

### Pull the image

```bash
docker pull rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0
```

> [!NOTE]
> **Docker Hub mirror**
>
> If Docker Hub is inaccessible, pull through the
> [DaoCloud mirror](https://github.com/DaoCloud/public-image-mirror)
> and retag the image:
>
> ```bash
> docker pull m.daocloud.io/docker.io/rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0
> docker tag m.daocloud.io/docker.io/rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0 \
>   rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0
> ```

### Start and enter the container

Start the container:

```bash
docker run --detach --name sage-a15-vllm \
  --device /dev/kfd \
  --device /dev/dri \
  --shm-size 64g \
  --publish 127.0.0.1:8000:8000 \
  --security-opt seccomp=unconfined \
  --entrypoint sleep \
  rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0 \
  infinity
```

Enter the container:

```bash
docker exec -it sage-a15-vllm bash
```

### Pre-download the model

Use the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
to pre-download the model. See the linked documentation for installation and
usage.

```bash
hf download InternScience/Sage-A1.5
```

> [!NOTE]
> **Hugging Face mirror for restricted networks**
>
> To use [HF-Mirror](https://hf-mirror.com/), set the endpoint before
> running `hf download` above and `vllm serve` below, in the same shell:
>
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```
>
> If `hf download` fails, you can try
> [`export HF_HUB_DISABLE_XET=1`](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhubdisablexet)
> before retrying.

> [!NOTE]
> **ModelScope for networks with limited Hugging Face access**
>
> Use ModelScope 1.37.1 with this image when setting `VLLM_USE_MODELSCOPE=True`.
> See [PR #52431](https://github.com/vllm-project/vllm/pull/52431) and
> [PR #55300](https://github.com/vllm-project/vllm/pull/55300)
> for compatibility issues with newer ModelScope SDK versions.
> Pre-download with the
> [ModelScope CLI](https://www.modelscope.cn/docs/models/download):
>
> ```bash
> python3 -m pip install modelscope==1.37.1
> modelscope download --model InternScience/Sage-A1.5
> ```
>
> Before running `vllm serve`, enable vLLM's
> [ModelScope integration](https://www.modelscope.cn/docs/ecosystem-integrations/model-download/vllm)
> in the same shell:
>
> ```bash
> export VLLM_USE_MODELSCOPE=True
> ```

### Start the model server

Choose one of the following commands for your hardware.
If you are unsure which category applies to your GPU, see the
[AMD GPU hardware specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html).

#### AMD Radeon GPUs

This example uses four GPUs to accommodate the large BF16 model.
Enable Triton FlashAttention for the multimodal vision encoder:

```bash
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
vllm serve InternScience/Sage-A1.5 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

You can use `HIP_VISIBLE_DEVICES` to select the four GPUs for this command, for
example `HIP_VISIBLE_DEVICES=0,1,2,3`.

#### AMD Ryzen APUs

Expandable memory segments and a lower sequence limit help avoid out-of-memory
errors. Enable Triton FlashAttention for the multimodal vision encoder:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
vllm serve InternScience/Sage-A1.5 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --max-num-seqs 8 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

#### AMD Instinct MI GPUs

Enable [AITER](https://github.com/ROCm/aiter)-optimized ROCm kernels:

```bash
VLLM_ROCM_USE_AITER=1 \
vllm serve InternScience/Sage-A1.5 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

### Test the OpenAI-compatible API

From another terminal, either on the host or inside the container, send a text
request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data '{
    "model": "InternScience/Sage-A1.5",
    "messages": [{"role": "user", "content": "Introduce yourself."}]
  }'
```

Send a multimodal request using a public image from Hugging Face:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data '{
    "model": "InternScience/Sage-A1.5",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"}}
      ]
    }]
  }'
```

> [!NOTE]
> **Image URL for restricted networks**
>
> If the image URL is inaccessible, replace it with this
> [HF-Mirror](https://hf-mirror.com/) URL:
>
> ```text
> https://hf-mirror.com/datasets/huggingface/documentation-images/resolve/main/bee.jpg
> ```

## llama.cpp

### Pull the image

```bash
docker pull rocm/dev-ubuntu-24.04:10.0.0-full
```

> [!NOTE]
> **Docker Hub mirror**
>
> If Docker Hub is inaccessible, pull through the
> [DaoCloud mirror](https://github.com/DaoCloud/public-image-mirror)
> and retag the image:
>
> ```bash
> docker pull m.daocloud.io/docker.io/rocm/dev-ubuntu-24.04:10.0.0-full
> docker tag m.daocloud.io/docker.io/rocm/dev-ubuntu-24.04:10.0.0-full \
>   rocm/dev-ubuntu-24.04:10.0.0-full
> ```

### Start and enter the container

Start the container:

```bash
docker run --detach --name sage-a15-llama \
  --device /dev/kfd \
  --device /dev/dri \
  --publish 127.0.0.1:8080:8080 \
  --security-opt seccomp=unconfined \
  --entrypoint sleep \
  rocm/dev-ubuntu-24.04:10.0.0-full \
  infinity
```

Enter the container:

```bash
docker exec -it sage-a15-llama bash
```

### Install llama.cpp

Install the build dependencies inside the container:

```bash
apt-get update
apt-get install -y --no-install-recommends build-essential ca-certificates cmake git libcurl4-openssl-dev libssl-dev
```

Download the source:

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
```

> [!NOTE]
> **GitHub proxy for restricted networks**
>
> If GitHub is inaccessible, use the third-party
> [GH-Proxy](https://gh-proxy.com/)
> instead of the clone command above:
>
> ```bash
> git clone --depth 1 https://gh-proxy.org/https://github.com/ggml-org/llama.cpp.git
> cd llama.cpp
> ```

Build and install llama.cpp with HIP support:

Set `GPU_TARGETS` using the LLVM target name from the
[AMD GPU hardware specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-specs.html)
or from `rocminfo | grep gfx`. Specify only the targets you need;
separate multiple targets with semicolons. The example below builds for
`gfx950`, `gfx1100`, `gfx1151`, and `gfx1201`.

```bash
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
cmake -S . -B build \
  -DGGML_HIP=ON \
  -DGPU_TARGETS="gfx950;gfx1100;gfx1151;gfx1201" \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --parallel
cmake --install build
```

After installation, register the ROCm library directory with the system
dynamic linker:

```bash
echo /opt/rocm/lib > /etc/ld.so.conf.d/rocm.conf && ldconfig
```

### Pre-download the model

Use the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
to pre-download the model. See the linked documentation for installation and
usage.

```bash
hf download InternScience/Sage-A1.5-Q8_0-GGUF
```

> [!NOTE]
> **Hugging Face mirror for restricted networks**
>
> To use [HF-Mirror](https://hf-mirror.com/), set the endpoint before
> running `hf download` above and `llama-server` below, in the same shell:
>
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```
>
> If `hf download` fails, you can try
> [`export HF_HUB_DISABLE_XET=1`](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables#hfhubdisablexet)
> before retrying.

> [!NOTE]
> **ModelScope for networks with limited Hugging Face access**
>
> Use the [ModelScope CLI](https://www.modelscope.cn/docs/models/download)
> to download the files into a local directory. See the linked documentation
> for installation and usage:
>
> ```bash
> modelscope download --model InternScience/Sage-A1.5-Q8_0-GGUF \
>   --local-dir /models/Sage-A1.5-Q8_0-GGUF
> ```
>
> In the server commands below, replace `--hf-repo` with local model and
> projector paths, for example:
>
> ```bash
> llama-server \
>   --model /models/Sage-A1.5-Q8_0-GGUF/Sage-A1.5-Q8_0.gguf \
>   --mmproj /models/Sage-A1.5-Q8_0-GGUF/Sage-A1.5-mmproj.gguf \
>   --alias Sage-A1.5-Q8_0
> ```
>
> For integration details and updates, see the
> [ModelScope guide](https://www.modelscope.cn/docs/ecosystem-integrations/model-download/llama-cpp),
> [issue #1658](https://github.com/modelscope/modelscope/issues/1658), and
> [PR #24716](https://github.com/ggml-org/llama.cpp/pull/24716).
> Native `--ms-repo` support is not yet merged.

### Start the model server

Start the server inside the container. `HIP_VISIBLE_DEVICES=0` selects GPU 0
for this single-GPU example:

```bash
HIP_VISIBLE_DEVICES=0 \
llama-server \
  --hf-repo InternScience/Sage-A1.5-Q8_0-GGUF:Q8_0 \
  --alias Sage-A1.5-Q8_0 \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 262144 \
  --n-gpu-layers all
```

On Radeon AI PRO R9700, Q8_0 may not fit on one GPU. Use two GPUs with
layer splitting:

```bash
HIP_VISIBLE_DEVICES=0,1 \
llama-server \
  --hf-repo InternScience/Sage-A1.5-Q8_0-GGUF:Q8_0 \
  --alias Sage-A1.5-Q8_0 \
  --host 0.0.0.0 \
  --port 8080 \
  --ctx-size 262144 \
  --n-gpu-layers all \
  --split-mode layer
```

### Test the OpenAI-compatible API

From another terminal, either on the host or inside the container, send a
text request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data '{
    "model": "Sage-A1.5-Q8_0",
    "messages": [{"role": "user", "content": "Introduce yourself."}]
  }'
```

Send a multimodal request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  --data '{
    "model": "Sage-A1.5-Q8_0",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"}}
      ]
    }]
  }'
```

> [!NOTE]
> **Image URL for restricted networks**
>
> If the image URL is inaccessible, replace it with this
> [HF-Mirror](https://hf-mirror.com/) URL:
>
> ```text
> https://hf-mirror.com/datasets/huggingface/documentation-images/resolve/main/bee.jpg
> ```
