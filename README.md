## Benchmark NVFP4 Training Performance on Blackwell 8xB200 with NeMo

*24/Oct/2025*

NVIDIA’s latest [**Transformer Engine 2.8**](https://github.com/NVIDIA/TransformerEngine/releases/tag/v2.8) introduces support for NVFP4 training (both forward and backward passes). While official upstream support in Megatron-LM and NeMo has not yet been released, this repo is an early local integration to see the performance uplift.

The repository provides:

* A custom Docker build that enables NVFP4 within the NeMo framework.

* Adapts NeMo's performance scripts to run locally on a **single-node 8×B200 system** (original scripts require Slurm).

* Reproducible steps and benchmarks of Llama3-8B pretraining, supervised fine-tuning and Llama3-70B LORA.

*Key takeaway*: **NVFP4 pretraining of Llama3-8B, 8K length on 8xB200 achieves **1.65× speedup** over BF16, and is **1.25× faster** than MXFP8.** 

Main references: [Paper](https://arxiv.org/abs/2509.25149), [Blog](https://developer.nvidia.com/blog/nvfp4-trains-with-precision-of-16-bit-and-speed-and-efficiency-of-4-bit/), [PR](https://github.com/NVIDIA/TransformerEngine/pull/2177). Great work by NVIDIA team!

Do check out the paper to learn about the key ingredients for NVFP4 convergence, i.e. stochastic rounding, randomized Hadamard transform during backpropagation.

### Benchmarks
1. FP8 and MXFP8 performance are within 5% of the [official numbers](https://docs.nvidia.com/nemo-framework/user-guide/25.07/performance/performance-summary.html) of NeMo 25.07, verifying our setup and benchmarking.

-------------------
> Config: 8xB200, GBS=128, MBS=2, Seq.Len=8192, TP=1, PP=1, CP=1, VP=1, EP=1, GA=8

| Workload                | Precision        | TPS/GPU  | Speedup over BF16 |
|-------------------------| -----------------|---------:|-----:|
| **Pretrain llama3-8b**  | BF16             | 21,618   | 1.00×|
|                         | FP8 (per-tensor) | 29,567   | 1.37×|
|                         | MXFP8            | 28,432   | 1.32×|
|                         | NVFP4            | 35,578   | 1.65×|

NVFP4 is 1.25× higher than MXFP8 here.

-------------------
> Config: 8xB200, GBS=*8*, MBS=*1*, Seq.Len=*16384*, TP=1, PP=1, CP=1, VP=1, EP=1, GA=*1*

| Workload         | Precision        | TPS / GPU | Speedup over BF16 |
|------------------|------------------|----------:|------------------:|
|**SFT llama3-8b** | BF16             | 24,690    | 1.00×             |
|                  | FP8 (per-tensor) | 33,362    | 1.35×             |
|                  | MXFP8            | 31,556    | 1.28×             |
|                  | NVFP4            | 35,578    | 1.44×             |

NVFP4 is 1.12× higher than MXFP8. Lower due to longer sequence length (16k) which has larger portion of the load and is not targeted for optimization, NVFP4 only applies to linear layers.

-------------------
> Config: 8xB200, GBS=*32*, MBS=*1*, Seq.Len=*4096*, TP=1, PP=*4*, CP=1, VP=*20*, EP=1, GA=*16*

| Workload         | Precision        | TPS / GPU | Speedup over BF16 |
|------------------|------------------|----------:|------------------:|
| LORA Llama3-70b  | BF16             | 3,772     | 1.00×             |
|                  | FP8 (per-tensor) | 5,919     | 1.57×             |
|                  | MXFP8            | 5,675     | 1.50×             |
|                  | NVFP4            | OOM       | –                 |

OOM for NVFP4, suspect due to larger memory footprint given block size of 16 as opposed to 32 for MXFP8. Further investigation needed. Potentially changing the bench config to reduce memory usage.

-------------------
### Build: `docker build -t nemo-2509-nvfp4 .`
or pre-built image: `docker pull vuiseng9:nemo-2509-nvfp4`

### Steps:
1. **Pretrain llama3_8b**  
    ```bash
    export NEMORUN_HOME=/path/to/run/output/dir
    NSTEP=25
    # ----------------------------------------------------------------------------------
    # Pretraining llama3_8b Benchmarks
    # pretrain.1. 8xB200, BF16 
    python -m scripts.performance.llm.pretrain_llama3_8b -g b200 --max_steps $NSTEP

    # pretrain.2. 8xB200, FP8 (delayed scaling)
    python -m scripts.performance.llm.pretrain_llama3_8b -g b200 --max_steps $NSTEP \
        --compute_dtype fp8 --fp8_recipe ds

    # pretrain.3. 8xB200, MXFP8
    python -m scripts.performance.llm.pretrain_llama3_8b -g b200 --max_steps $NSTEP \
        --compute_dtype fp8 --fp8_recipe mxfp8

    # pretrain.4. 8xB200, NVFP4
    python -m scripts.performance.llm.pretrain_llama3_8b -g b200 --max_steps $NSTEP \
        --compute_dtype fp8 --fp8_recipe nvfp4
    ```

1. **SFT llama3_8b**
    ```bash
    export NEMO_HOME=/path/to/nemo/model/root
    export NEMORUN_HOME=/path/to/run/output/dir
    STEP=25
    export hftoken=<hf token for model download>
    # ----------------------------------------------------------------------------------
    # SFT llama3_8b Benchmarks
    # sft.1. 8xB200, BF16
    python -m scripts.performance.llm.finetune_llama3_8b -g b200 --max_steps $NSTEP \
        --finetuning sft -hf $hftoken

    # sft.2. 8xB200, FP8 (delayed scaling) 
    python -m scripts.performance.llm.finetune_llama3_8b -g b200 --max_steps $NSTEP \
        --compute_dtype fp8 --fp8_recipe ds --finetuning sft -hf $hftoken

    # sft.3. 8xB200, MXFP8 
    python -m scripts.performance.llm.finetune_llama3_8b -g b200 --max_steps $NSTEP \
        --compute_dtype fp8 --fp8_recipe mxfp8 --finetuning sft -hf $hftoken

    # sft.4. 8xB200, NVFP4
    python -m scripts.performance.llm.finetune_llama3_8b -g b200 --max_steps $NSTEP \
        --compute_dtype fp8 --fp8_recipe nvfp4 --finetuning sft -hf $hftoken
    ```

1. **LORA llama3_70b**
    ```bash
    export NEMO_HOME=/path/to/nemo/model/root
    export NEMORUN_HOME=/path/to/run/output/dir
    STEP=25
    export hftoken=<hf token for model download>
    # ----------------------------------------------------------------------------------
    # LORA llama3_70b Benchmarks
    # lora.1. 8xB200, BF16
    python -m scripts.performance.llm.finetune_llama3_70b -g b200 --max_steps $NSTEP \
        --finetuning lora -hf $hftoken

    # lora.2. 8xB200, FP8
    python -m scripts.performance.llm.finetune_llama3_70b -g b200 --max_steps $NSTEP \
        --compute_dtype fp8 --fp8_recipe ds --finetuning lora -hf $hftoken

    # lora.3. 8xB200, MXFP8
    python -m scripts.performance.llm.finetune_llama3_70b -g b200 --max_steps $NSTEP \
        --compute_dtype fp8 --fp8_recipe mxfp8 --finetuning lora -hf $hftoken

    # lora.4. 8xB200, NVFP4
    python -m scripts.performance.llm.finetune_llama3_70b -g b200 --max_steps $NSTEP \
        --compute_dtype fp8 --fp8_recipe nvfp4 --finetuning lora -hf $hftoken
    ```

### Training Throughput (Tokens/second/gpu):
```
TPS = (GBS × L) / train_step_timing
TPS/gpu = TPS/n_gpus

L: sequence length; GBS: global batch size; n_gpus: number of gpus used

e.g. given log:
# mbs_128gbs/0 Training epoch 0, iteration 10/14 | lr: 1.649e-06 | global_batch_size: 128
| global_step: 10 | max_memory_reserved: 140129599488 | max_memory_allocated: 107693703168
| reduced_train_loss: 11.97 | train_step_timing in s: 3.684 | TFLOPS_per_GPU: 1.831e+03 | consumed_samples: 1408

# TPS/gpu = (128*8192)/3.684/8 = 35578 tps/gpu
```

### Helpful Notes:
1. NeMo bench script `scripts/performance/llm/pretrain_llama3_8b.py` can look up perf config `recommended_model_configs/*.csv` which are ones used in official benchmark. Therefore, setting minimal set of args is sufficient, e.g. gpu type, precision etc.

1. The default run output is at `~/.nemo_run`, use `NEMORUN_HOME` to override. Overide `NEMO_HOME` to point where models live (needed for fine-tuning bench). Do ensure enough disk space is available.

1. Where to look for the benchmark logs? 
    1. Nemo CLIs that will shown at end of run. Take note of the id at the end that indicates the step of a given experiment run, example for a sft bench:
        
        `nemo experiment logs sft_finetune_llama3_8b_fp8_1nodes_tp1_pp1_cp1_vp1_1mbs_8gbs_1760722087 1` 
    2. In  the output directory, use `tree`, find `stdout.log`. 
