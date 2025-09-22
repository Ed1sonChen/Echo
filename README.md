# Can Multimodal Large Language Models Be Guided to Improve Industrial Anoamly Detection?

This is the official implementation of paper "Can Multimodal Large Language Models Be Guided to Improve Industrial Anoamly Detection?".

> Zhiling Chen*, Hanning Chen, Mohsen Imani, Farhad Imani.

We propose an multi-experts framework named Echo, which comprises four expert modules working together to guide open source MLLMs to improve anomaly detection.

<p align="center" width="80%">
<a ><img src="Figures\model.png" alt="overview" style="width: 100%; min-width: 300px; display: block; margin: auto;"></a>
</p>



## Environment Setup ##

```bash
# Create environment
conda create -n echo python=3.12 -y
conda activate echo

# Install dependencies
pip install -r requirements.txt
```

## Build Memory Index ##

```bash
python /Memory/build_memory.py
```


## Run Evaluation ##

```bash
Python /Experiemnts/Qwen2-VL.py
```

## 👨‍💻 Todo
- [x] Release the evaluation code
- [x] Release the paper


## Acknowledgements ##
We would like to acknowledge the use of code snippets from various open-source libraries and contributions from the online coding community, which have been invaluable in the development of this project. Specifically, we would like to thank the authors and maintainers of the following resources:

[MMAD](https://github.com/jam-cc/MMAD)

[FAISS](https://github.com/facebookresearch/faiss)

[RAR](https://github.com/Liuziyu77/RAR)

[Qwen2-VL](https://github.com/QwenLM/Qwen2-VL)

[LLaVA-VL](https://github.com/LLaVA-VL)

[MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V)

[InternVL](https://github.com/OpenGVLab/InternVL)


## Citation
```
@article{chen2025can,
  title={Can multimodal large language models be guided to improve industrial anomaly detection?},
  author={Chen, Zhiling and Chen, Hanning and Imani, Mohsen and Imani, Farhad},
  journal={arXiv preprint arXiv:2501.15795},
  year={2025}
}
```
