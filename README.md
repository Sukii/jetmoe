# Python scripts for running JetMoE on CPU/GPU

Please pull `jetmoe-8b` model from `Huggingface` to `model/` folder and `jetmoe-8b-chat` model from `Huggingface` to `chat-model/` folder by running:
```
./install.sh
```
It should take about 15 minutes to pull both the models. 

Now you should be able to run:
```
python jet.py
```
and
```
python chat-jet.py
```
Please use higher version of `python` (`>=3.10`) and higher version of `transformers` (`>=3.11`).

You need atleast 64GB RAM and about 10 CPU cores to be able to run it. But if you have GPUs then it should work much more efficienty as it can process all those large arrays together through those thosands of parallel-gears of GPUs.

The bash script for pulling models should work directly in Linux/MacOSX and in Windows if you use install git-bash terminal (https://gitforwindows.org/).