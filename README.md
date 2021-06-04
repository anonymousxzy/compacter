# Compacter: Efficient Low-Rank Hypercomplex Adapter Layers

## Python requirements
This code is tested on:
- Python 3.7
- transformers  4.6.0
- pytorch 1.8.1

## Parameters in the code:
* `task_reduction_factor`: defines the adapter bottleneck size.
* `unfreeze_lm_head`: If set trains the final output layer, and if set to False, does not tune this layer.
* `train_task_adapters`: To train adapters.
* `task_adapter_layers_encoder`: A list of layers to have adapters for the encoder
* `task_adapter_layers_decoder`: A list of layers to have adapters for the decoder
* `prefix_tuning`: If set, trains prompt tuning method.
* `prefix_dim`: Specifies the number of tokens in the prompt to be added to the input.
* `init_prefix_from_vocab`: If set, intialize the prompt tokens from the pretrained vocabulary of T5-base model.
* `intrinsic_model`: If set, trains  intrinsic based methods (DID/SAID), see https://arxiv.org/abs/2012.13255 for more info.
* `intrinsic_said`: If set, trains the intrinsic said method. If this is not set, trains the DID method.
* `intrinsic_dim`: Specifies the intrinsic dimension.
* `bitfit`: If set, trains the bitfit model.
* `freeze_bitfit_lm_head`: If set, freezes the output layer for bitfit model but only sets the biases to be trainable.
* `hypercomplex_adapters`: Set to train phm-adapters.
* `hypercomplex_division`: Defines the number of kronecker products.
* `factorized_phm`: If set factorizes B matrices in phm layers.
* `shared_phm_rule`: If set, shares the A matrices between all layers of the transformer.


## Setup 
1. install pytorch 
```
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
```
2. run the installation
```
python setup.py develop 
```
3. install the CUDA implementations 
```
cd seq2seq/projections/fwh_cuda
python setup.py develop  

cd seq2seq/projections/hadamard_cuda
python setup.py develop
```


## Usage 
We provide the sample config files to run each model in ```seq2seq/sample_configs```.

* To train T5 model 
```
python run_seq2seq.py sample_configs/t5.json 
```

* To train Adapter model
```
python run_seq2seq.py sample_configs/adapter.json
```


* To train Pfeiffer Adapter
```
python run_seq2seq.py sample_configs/pfeiffer_adapter.json 
```


* To train AdapterDrop 
```
python run_seq2seq.py sample_configs/adapter_drop.json 
```

* To train prompt tuning with intializing tokens randomly
```
python run_seq2seq.py sample_configs/prompt_tuning_random.json 
```

* To train prompt tuning with initializing tokens from the vocabulary
```
python run_seq2seq.py sample_configs/prompt_tuning_token.json
```

* To train the intrinsic said method
```
python run_seq2seq.py sample_configs/intrinsic_said.json
```

* To train bitfit 
```
python run_seq2seq.py sample_configs/bitfit.json 
```

* To train phm-adapters 
```
python run_seq2seq.py sample_configs/phm_adapter.json
```

* To train compacter 
```
python run_seq2seq.py sample_configs/compacter.json 
```

* To train compacter++ 
```
python run_seq2seq.py sample_configs/compacter++.json  
```







