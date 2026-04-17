Experiments on qauntization.


Model                         Perplexity    PPL delta  Sizefloat32 
baseline                      4.3070         —          41MB 
PTQ int8 (w-out fine-tunning) 4.3187        +0.27%      0.5 MB
QAT int8 (1000 steps)         4.3161        +0.21%      10 MB

Experiment A: QAT steps vs quality

Method                             Val loss   Perplexity    Size MB    PPL delta
--------------------------------------------------------------------------------
float32_baseline                     1.4597       4.3047       41.0     baseline
qat_0_steps                          1.4614       4.3122       41.0      +0.175%
qat_500_steps                        1.4620       4.3144       41.0      +0.226%
qat_1000_steps                       1.4558       4.2877       41.0      -0.393%
qat_3000_steps                       1.4633       4.3203       41.0      +0.364%
qat_5000_steps                       1.4697       4.3479       41.0      +1.004%



Experiment B: quantization sensitivity by model size
QAT iters: 1000

Model                Method                 Val loss        PPL    PPL delta
--------------------------------------------------------------------------------
small (nl=6)         float32_baseline         1.4603     4.3073     baseline
small (nl=6)         ptq_int8                 1.4638     4.3222       0.348%
small (nl=6)         qat_int8                 1.4612     4.3113       0.093%

large (nl=12)        float32_baseline         1.4706     4.3518     baseline
large (nl=12)        ptq_int8                 1.4676     4.3386      -0.302%
large (nl=12)        qat_int8                 1.4790     4.3886       0.847%



As we can see perplexity delta is very small in the comparison with the memory we spend on the model. 


To recreate this experiment you need to download nanoGPT model https://github.com/karpathy/nanoGPT (or any other model that can be quantized) and use a wrapper qat.py and qat_ab.py that will train model.
As the result you will get perplexity which we now can compare.
WARNING you need to chage paths to the model in files.
