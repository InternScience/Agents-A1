# Agents-A1

[![MODELS](https://img.shields.io/badge/Models-5EDDD2?style=for-the-badge&logo=huggingface&logoColor=ffffff&labelColor)](https://huggingface.co/InternScience/Agents-A1)
[![GITHUB](https://img.shields.io/badge/Github-24292F?style=for-the-badge&logo=github&logoColor=white)](https://github.com/InternScience/Agents-A1)
[![Blog](https://img.shields.io/badge/Blog-4285F4?style=for-the-badge&logo=google-chrome&logoColor=white)](https://xxx/)
[![Paper](https://img.shields.io/badge/Paper-red?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/xxx)


**Agents‑A1** is a 35B Mixture‑of‑Experts agentic model designed to scale heterogeneous agentic abilities across multiple domains including **Long‑horizon Search, Engineering, Scientific Research, Instruction Following, and Tool-calling**. Rather than relying solely on flat instruction–response corpora, **Agents‑A1** is trained with the assistance of a domain-grounded knowledge-action graph infrastructure that jointly constructs actions, observations, and verifier outcomes, turning the agent's process into a trainable target. Based on this, **Agents‑A1** presents a three-stage training paradigm for building scalable general-purpose agentic model. First, we perform full-domain supervised fine-tuning to align the base model with broad agentic behaviors. Second, we train domain-level teacher models to capture specialized expertise in each domain. Third, we propose multi-teacher multi-domain on-policy distillation with heterogeneity-aware optimization to improve knowledge transfer efficiency across different domains.

![Agents-A1 Benchmark Overview](./assets/a1_benchmarks_altair_grid.svg)

## Highlights

- **Agentic Reasoning**: Agents-A1 excels at decomposing complex tasks into executable sub-steps, planning ahead, and adapting its strategy based on intermediate results.
- **Tool Use**: Natively supports function calling and tool integration, enabling seamless interaction with APIs, code interpreters, search engines, and other external tools.
- **Long-Context Understanding**: Handles extended conversations and documents with strong coherence and recall.
- **Instruction Following**: Precisely follows detailed, multi-constraint instructions across diverse domains.

We welcome developers and enterprises to integrate and try Agents-A1 and share their feedback.

## Performance

We evaluate Agents-A1 in real-world agentic and research-oriented workflows across six directions — long-horizon search, engineering tasks, scientific research, instruction following, general agentic tasks, and scientific agentic tasks. Despite operating in the ~35B model class, Agents-A1 delivers highly competitive performance against frontier-scale systems such as GPT-5.5, DeepSeek-V4-pro, and Kimi-K2.6. It achieves overall SOTA results on several challenging benchmarks, including Seal-0 (56.4), HiPhO (46.4), FrontierScience-Olympiad (79.0), FrontierScience-Research (40.00), IFBench (80.6), and IFEval (94.8), while also ranking as the best among comparable models on a broad range of tasks such as BrowseComp (75.5), XBench-DS-2510 (86.0), GAIA (96.0), SciCode (44.3), HLE with tools (47.6), and MolBench-bind (56.8). These results show that Agents-A1 combines strong long-horizon search ability, robust scientific reasoning, and reliable instruction following, establishing it as a highly capable and efficient agentic model that narrows the gap with much larger frontier models.

<p>
🥇 Overall SOTA &nbsp;&nbsp;
🟢 Best Among Comparable Models (~35B)
</p>

<table>
<thead>
<tr>
<th rowspan="2" align="left">Benchmark</th>
<th colspan="3" align="center" style="text-align:center;">
    📏 Comparable (~35B)
</th>

<th colspan="4" align="center" style="text-align:center;">
    🚀 Frontier (&gt;1T)
</th>

<th colspan="2" align="center" style="text-align:center;">
    ⭐ Ours
</th>
</tr>

<tr>
<th align="center">Qwen3.5-35B-A3B</th>
<th align="center">Qwen3.6-35B-A3B</th>
<th align="center">Nex-N2-mini</th>
  
<th align="center">Step-3.5-Flash</th>
<th align="center">Kimi-K2.6</th>
<th align="center">DeepSeek-V4-pro(Max)</th>
<th align="center">GPT-5.5(xhigh)</th>

<th align="center">Agents-A1</th>
</tr>
</thead>

<tbody>

<tr>
    <td colspan="9" align="left"><b>🔍 Long-horizon Search</b></td>
</tr>

<tr>
<td align="left">BrowseComp</td>
<td align="center">61.0</td>
<td align="center">67.93</td>
<td align="center">74.1</td>
<td align="center">69.0</td>
<td align="center">83.2</td>
<td align="center">83.4</td>
<td align="center">🥇 84.4</td>
<td align="center">🟢 75.51</td>
</tr>

<tr>
<td align="left">XBench-DS-2510</td>
<td align="center">77.0</td>
<td align="center">71.0</td>
<td align="center">82.0</td>
<td align="center">56.3</td>
<td align="center">🥇 90.0</td>
<td align="center">🥇 90.0</td>
<td align="center">84.0</td>
<td align="center">🟢 86.0</td>
</tr>

<tr>
<td align="left">Seal0</td>
<td align="center">41.4</td>
<td align="center">38.74</td>
<td align="center">49.55</td>
<td align="center">36.94</td>
<td align="center">50.45</td>
<td align="center">54.95</td>
<td align="center">42.34</td>
<td align="center">🥇 56.36</td>
</tr>

<tr>
<td align="left">GAIA</td>
<td align="center">59.8</td>
<td align="center">78.64</td>
<td align="center">82.52</td>
<td align="center">84.5</td>
<td align="center">80.58</td>
<td align="center">🥇 98.06</td>
<td align="center">87.38</td>
<td align="center">🟢 96.04</td>
</tr>

<tr>
    <td colspan="9" align="left"><b>⚙️ Engineering Tasks</b></td>
</tr>

<tr>
<td align="left">SciCode</td>
<td align="center">37.7</td>
<td align="center">35.8</td>
<td align="center">29.9</td>
<td align="center">40.4</td>
<td align="center">53.5</td>
<td align="center">50.0</td>
<td align="center">🥇 56.1</td>
<td align="center">🟢 44.33</td>
</tr>

<tr>
<td align="left">MLE-Lite</td>
<td align="center">24.24</td>
<td align="center">34.85</td>
<td align="center">34.85</td>
<td align="center">54.55</td>
<td align="center">62.12</td>
<td align="center">63.64</td>
<td align="center">🥇 72.73</td>
<td align="center">🟢 43.94</td>
</tr>

<tr>
    <td colspan="9" align="left"><b>🧪 Scientific Research</b></td>
</tr>

<tr>
<td align="left">HLE w/ tools</td>
<td align="center">47.4</td>
<td align="center">36.2</td>
<td align="center">32.0</td>
<td align="center">23.1</td>
<td align="center">🥇 54.0</td>
<td align="center">48.2</td>
<td align="center">52.2</td>
<td align="center">🟢 47.6</td>
</tr>

<tr>
<td align="left">HiPhO</td>
<td align="center">37.0</td>
<td align="center">37.7</td>
<td align="center">38.5</td>
<td align="center">38.3</td>
<td align="center">41.1</td>
<td align="center">38.7</td>
<td align="center">43.3</td>
<td align="center">🥇 46.4</td>
</tr>

<tr>
<td align="left">FrontierScience-Olympiad</td>
<td align="center">64.5</td>
<td align="center">60.3</td>
<td align="center">52.0</td>
<td align="center">61.0</td>
<td align="center">73.0</td>
<td align="center">76.0</td>
<td align="center">78.0</td>
<td align="center">🥇 79.0</td>
</tr>

<tr>
<td align="left">FrontierScience-Research</td>
<td align="center">2.5</td>
<td align="center">2.9</td>
<td align="center">5.0</td>
<td align="center">6.7</td>
<td align="center">17.9</td>
<td align="center">13.3</td>
<td align="center">26.7</td>
<td align="center">🥇 40.0</td>
</tr>

<tr>
    <td colspan="9" align="left"><b>📋 Instruction Following</b></td>
</tr>

<tr>
<td align="left">IFBench</td>
<td align="center">70.2</td>
<td align="center">64.4</td>
<td align="center">54.08</td>
<td align="center">64.6</td>
<td align="center">71.77</td>
<td align="center">73.47</td>
<td align="center">75.9</td>
<td align="center">🥇 80.61</td>
</tr>

<tr>
<td align="left">LongBench-v2</td>
<td align="center">59.0</td>
<td align="center">57.7</td>
<td align="center">59.6</td>
<td align="center">57.5</td>
<td align="center">62.0</td>
<td align="center">🥇 64.3</td>
<td align="center">-</td>
<td align="center">🟢 60.2</td>
</tr>

<tr>
<td align="left">IFEval</td>
<td align="center">91.9</td>
<td align="center">91.3</td>
<td align="center">88.4</td>
<td align="center">93.53</td>
<td align="center">94.45</td>
<td align="center">93.35</td>
<td align="center">93.35</td>
<td align="center">🥇 94.82</td>
</tr>

<tr>
    <td colspan="9" align="left"><b>🤖 General Agentic Tasks</b></td>
</tr>

<tr>
<td align="left">τ<sup>2</sup>-Bench</td>
<td align="center">🟢 81.2</td>
<td align="center">79.0</td>
<td align="center">74.53</td>
<td align="center">75.77</td>
<td align="center">81.93</td>
<td align="center">🥇 82.2</td>
<td align="center">81.63</td>
<td align="center">79.81</td>
</tr>

<tr>
<td align="left">VitaBench</td>
<td align="center">31.9</td>
<td align="center">35.6</td>
<td align="center">23.0</td>
<td align="center">30.0</td>
<td align="center">35.63</td>
<td align="center">🥇 49.04</td>
<td align="center">45.0</td>
<td align="center">🟢 38.75</td>
</tr>

<tr>
    <td colspan="9" align="left"><b>🔬 Scientific Agentic Tasks</b></td>
</tr>

<tr>
<td align="left">MatTools</td>
<td align="center">21.0</td>
<td align="center">15.9</td>
<td align="center">34.1</td>
<td align="center">44.93</td>
<td align="center">63.8</td>
<td align="center">47.1</td>
<td align="center">🥇 68.8</td>
<td align="center">🟢 47.1</td>
</tr>

<tr>
<td align="left">MolBench-bind</td>
<td align="center">46.0</td>
<td align="center">48.7</td>
<td align="center">51.4</td>
<td align="center">45.95</td>
<td align="center">21.6</td>
<td align="center">37.8</td>
<td align="center">🥇 62.2</td>
<td align="center">🟢 56.8</td>
</tr>

</tbody>
</table>


## Usage

### SGLang

[SGLang](https://github.com/sgl-project/sglang) is a fast serving framework for large language models and vision language models.

Install SGLang with uv:

```shell
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate

uv pip install sglang
```

See [its documentation](https://docs.sglang.ai/get_started/install.html) for more details.

The following commands create API endpoints at `http://localhost:8000/v1`:

- **Standard Version** (1 GPUs, 262K context):

  ```shell
  python -m sglang.launch_server \
    --model-path InternScience/Agents-A1 \
    --port 8000 \
    --tp-size 1 \
    --mem-fraction-static 0.8 \
    --context-length 262144 \
    --reasoning-parser qwen3
  ```
- **Tool Use**:

  ```shell
  python -m sglang.launch_server \
    --model-path InternScience/Agents-A1 \
    --port 8000 \
    --tp-size 1 \
    --mem-fraction-static 0.8 \
    --context-length 262144 \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_coder
  ```

### vLLM

[vLLM](https://github.com/vllm-project/vllm) is a high-throughput and memory-efficient inference and serving engine for LLMs.

Install vLLM from the main branch via uv:

```shell
uv venv --python 3.12 --seed --managed-python
source .venv/bin/activate

uv pip install vllm --torch-backend=auto
```

See [its documentation](https://docs.vllm.ai/en/stable/getting_started/installation/index.html) for more details.

The following commands create API endpoints at `http://localhost:8000/v1`:

- **Standard Version** (1 GPUs, 262K context):

  ```shell
  vllm serve InternScience/Agents-A1 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 262144 \
    --reasoning-parser qwen3
  ```
- **Tool Call**:

  ```shell
  vllm serve InternScience/Agents-A1 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 262144 \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
  ```
- **Text-Only** (skips vision encoder to free KV cache memory):

  ```shell
  vllm serve InternScience/Agents-A1 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 262144 \
    --reasoning-parser qwen3 \
    --language-model-only
  ```

  For multi-GPU deployment, e.g., serving the model on 4 GPUs, please refer to the example launch script in the [`scripts/`](./scripts) directory.

### Recommended Sampling Parameters

For the best generation quality, we recommend the following sampling parameters:

- `temperature`: 0.85
- `top_p`: 0.95
- `top_k`: 20
- `min_p`: 0.0
- `presence_penalty`: 1.1
- `repetition_penalty`: 1.0


## Agent Capability Evaluation

To provide the community with a unified agent evaluation codebase for fair comparison, we have also open-sourced an evaluation framework for assessing agentic models across core capabilities, including tool use and multi-step reasoning. The evaluation code is included in the [`evaluation/`](./evaluation) of this repository.

We use this framework to evaluate the released model under a standardized and reproducible setting. 
Specifically, the model is tested on a set of agent-oriented tasks that require it to understand user goals, decompose complex instructions, interact with tools or environments when necessary, and produce final results. The evaluation results reported in [Model Card](https://huggingface.co/InternScience/Agents-A1) are generated using the open-source framework above, so that users can reproduce the experiments, compare other models under the same protocol, and further extend the benchmark for new agent scenarios. (**Note that:** To ensure a fair comparison, we report the benchmark results from their original technical reports. If a model does not report the corresponding benchmark results, we evaluate it using the same evaluation protocol as our model.)

For detailed evaluation scripts, task definitions, metrics, and reproduction instructions, please refer to the evaluation codebase.

## Citation

If you find our work helpful, feel free to give us a cite.

```

```
