<p align="center">
    <img src="./assets/ChartLLamma.jpg" width="250" style="margin-bottom: 0.2;"/>
<p>
<h2 align="center"> ViQwen-MoE-LLaVA: Intern VIT Mixture of Experts for Large Vision-Language Models</a></h2>
<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for latest update.  </h2>
<h5 align="center"> This Repo is fork from MoE-LLaVa!! </h2>

<h5 align="center">
    
## 😮 Highlights
This is my graduate dissertation at IUH base on MoE-LLaVA we will train it on chart dataset with small, complex feature that got extract feature by InternViT 300M then go through Small-Language Model (SLM) that is Qwen 0.5B 

This is our poster presentation:

<p align="center">
    <img src="./assets/poster.png" style="margin-bottom: 0.2;"/>


## 🤗 Demo

### Gradio Web UI  <a href='https://github.com/gradio-app/gradio'><img src='https://img.shields.io/github/stars/gradio-app/gradio'></a> 

Highly recommend trying out our web demo by the following command, which incorporates all features currently supported by MoE-LLaVA. We also provide [online demo](https://huggingface.co/spaces/LanguageBind/MoE-LLaVA) in Huggingface Spaces.
```bash
# use phi2
deepspeed --include localhost:0 moellava/serve/gradio_web_server.py --model-path "LanguageBind/MoE-LLaVA-Phi2-2.7B-4e" 
# use qwen
deepspeed --include localhost:0 moellava/serve/gradio_web_server.py --model-path "LanguageBind/MoE-LLaVA-Qwen-1.8B-4e" 
# use stablelm
deepspeed --include localhost:0 moellava/serve/gradio_web_server.py --model-path "LanguageBind/MoE-LLaVA-StableLM-1.6B-4e" 
```



https://github.com/PKU-YuanGroup/MoE-LLaVA/assets/62638829/8541aac6-9ef6-4fde-aa94-80d0375b9bdb



### CLI Inference

```bash
# use phi2
deepspeed --include localhost:0 moellava/serve/cli.py --model-path "LanguageBind/MoE-LLaVA-Phi2-2.7B-4e"  --image-file "image.jpg"
# use qwen
deepspeed --include localhost:0 moellava/serve/cli.py --model-path "LanguageBind/MoE-LLaVA-Qwen-1.8B-4e"  --image-file "image.jpg"
# use stablelm
deepspeed --include localhost:0 moellava/serve/cli.py --model-path "LanguageBind/MoE-LLaVA-StableLM-1.6B-4e"  --image-file "image.jpg"
```

<img src="assets/imagecli.gif" />

## 🐳 Model Zoo

| Model | Activated Param | Transformers(HF) | ModelScope(HF) | Avg | VQAv2 | GQA | VizWiz | SQA-IMG | T-VQA | POPE | MME | MM-Bench | MM-Vet |
|----------|-----------|-----------|---|---|---|---|---|---|---|---|---|---|---|
| MoE-LLaVA-1.6B×4-Top2 | 2.0B | [🤗LanguageBind/MoE-LLaVA-StableLM-1.6B-4e](https://huggingface.co/LanguageBind/MoE-LLaVA-StableLM-1.6B-4e) | [<img src="https://github.com/PKU-YuanGroup/MoE-LLaVA/raw/main/assets/modelscope_logo.png" width="20px" style="max-width: 100%;">PKU-YuanLab/MoE-LLaVA-StableLM-1.6B-4e](https://modelscope.cn/models/PKU-YuanLab/MoE-LLaVA-StableLM-1.6B-4e) | 57.3 | 76.7 | 60.3 | 36.2 | 62.6 | 50.1 | 85.7 | 1318.1 | 60.2 | 26.9 |
| MoE-LLaVA-1.8B×4-Top2 | 2.2B | [🤗LanguageBind/MoE-LLaVA-Qwen-1.8B-4e](https://huggingface.co/LanguageBind/MoE-LLaVA-Qwen-1.8B-4e) | [<img src="https://github.com/PKU-YuanGroup/MoE-LLaVA/raw/main/assets/modelscope_logo.png" width="20px" style="max-width: 100%;">PKU-YuanLab/MoE-LLaVA-Qwen-1.8B-4e](https://modelscope.cn/models/PKU-YuanLab/MoE-LLaVA-Qwen-1.8B-4e) | 56.7 | 76.2 | 61.5 | 32.6 | 63.1 | 48.0 | 87.0 | 1291.6 | 59.6 | 25.3 |
| MoE-LLaVA-2.7B×4-Top2 | 3.6B | [🤗LanguageBind/MoE-LLaVA-Phi2-2.7B-4e](https://huggingface.co/LanguageBind/MoE-LLaVA-Phi2-2.7B-4e) | [<img src="https://github.com/PKU-YuanGroup/MoE-LLaVA/raw/main/assets/modelscope_logo.png" width="20px" style="max-width: 100%;">PKU-YuanLab/MoE-LLaVA-Phi2-2.7B-4e](https://modelscope.cn/models/PKU-YuanLab/MoE-LLaVA-Phi2-2.7B-4e) | 61.1 | 77.6 | 61.4 | 43.9 | 68.5 | 51.4 | 86.3 | 1423.0 | 65.2 | 34.3 |
| MoE-LLaVA-1.6B×4-Top2-384 | 2.0B | [🤗LanguageBind/MoE-LLaVA-StableLM-1.6B-4e-384](https://huggingface.co/LanguageBind/MoE-LLaVA-StableLM-1.6B-4e-384) | [<img src="https://github.com/PKU-YuanGroup/MoE-LLaVA/raw/main/assets/modelscope_logo.png" width="20px" style="max-width: 100%;">PKU-YuanLab/MoE-LLaVA-StableLM-1.6B-4e-384](https://modelscope.cn/models/PKU-YuanLab/MoE-LLaVA-StableLM-1.6B-4e-384) | 60.0 | 78.6 | 61.5 | 40.5 | 63.9 | 54.3 | 85.9 | 1335.7 | 63.3 | 32.3 |
| MoE-LLaVA-2.7B×4-Top2-384 | 3.6B | [🤗LanguageBind/MoE-LLaVA-Phi2-2.7B-4e-384](https://huggingface.co/LanguageBind/MoE-LLaVA-Phi2-2.7B-4e-384) | [<img src="https://github.com/PKU-YuanGroup/MoE-LLaVA/raw/main/assets/modelscope_logo.png" width="20px" style="max-width: 100%;">PKU-YuanLab/MoE-LLaVA-Phi2-2.7B-4e-384](https://modelscope.cn/models/PKU-YuanLab/MoE-LLaVA-Phi2-2.7B-4e-384) | **62.9** | 79.9 | 62.6 | 43.7 | 70.3 | 57.0 | 85.7 | 1431.3 | 68.0 | 35.9 |
| LLaVA-1.5 | 7B | [🤗liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b) | - | 62.0 | 78.5 | 62.0 | 50.0 | 66.8 | 58.2 | 85.9 | 1510.7 | 64.3 | 30.5 |

<!--
| LLaVA-1.5 | 13B | [liuhaotian/llava-v1.5-13b](https://huggingface.co/liuhaotian/llava-v1.5-13b) | 64.9 | 80.0 | 63.3 | 53.6 | 71.6 | 61.3 | 85.9 | 1531.3 | 67.7 | 35.4 |
-->

<details>


🚨 **Please know https://github.com/PKU-YuanGroup/MoE-LLaVA/issues/27.**


<summary>Stage2 Model</summary>


    
| Model  | Checkpoint |
|----------|-----------|
| MoE-LLaVA-1.6B×4-Top2 | [LanguageBind/MoE-LLaVA-StableLM-Stage2](https://huggingface.co/LanguageBind/MoE-LLaVA-StableLM-Stage2) |
| MoE-LLaVA-1.6B×4-Top2-384 | [LanguageBind/MoE-LLaVA-StableLM-Stage2-384](https://huggingface.co/LanguageBind/MoE-LLaVA-StableLM-Stage2-384) |
| MoE-LLaVA-1.8B×4-Top2 | [LanguageBind/MoE-LLaVA-Qwen-Stage2](https://huggingface.co/LanguageBind/MoE-LLaVA-Qwen-Stage2) |
| MoE-LLaVA-2.7B×4-Top2 | [LanguageBind/MoE-LLaVA-Phi2-Stage2](https://huggingface.co/LanguageBind/MoE-LLaVA-Phi2-Stage2) |
| MoE-LLaVA-2.7B×4-Top2-384 | [LanguageBind/MoE-LLaVA-Phi2-Stage2-384](https://huggingface.co/LanguageBind/MoE-LLaVA-Phi2-Stage2-384) |

</details>

<details>
<summary>Pretrain Model</summary>

| Model  | Checkpoint |
|----------|-----------|
| MoE-LLaVA-1.6B×4-Top2 | [LanguageBind/MoE-LLaVA-StableLM-Pretrain](https://huggingface.co/LanguageBind/MoE-LLaVA-StableLM-Pretrain) |
| MoE-LLaVA-1.6B×4-Top2-384 | [LanguageBind/MoE-LLaVA-StableLM-384-Pretrain](https://huggingface.co/LanguageBind/MoE-LLaVA-StableLM-384-Pretrain) |
| MoE-LLaVA-1.8B×4-Top2 | [LanguageBind/MoE-LLaVA-Qwen-Pretrain](https://huggingface.co/LanguageBind/MoE-LLaVA-Qwen-Pretrain) |
| MoE-LLaVA-2.7B×4-Top2 | [LanguageBind/MoE-LLaVA-Phi2-Pretrain](https://huggingface.co/LanguageBind/MoE-LLaVA-Phi2-Pretrain) |
| MoE-LLaVA-2.7B×4-Top2-384 | [LanguageBind/MoE-LLaVA-Phi2-384-Pretrain](https://huggingface.co/LanguageBind/MoE-LLaVA-Phi2-384-Pretrain) |


</details>

## ⚙️ Requirements and Installation
We recommend the requirements as follows.
* Python == 3.10
* Pytorch == 2.0.1
* CUDA Version >= 11.7
* **Transformers == 4.37.0**
* **Tokenizers==0.15.1**
* Install required packages:
```bash
git clone https://github.com/PKU-YuanGroup/MoE-LLaVA
cd MoE-LLaVA
conda create -n moellava python=3.10 -y
conda activate moellava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install -e ".[train]"
pip install flash-attn --no-build-isolation

# Below are optional. For Qwen model.
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention && pip install .
# Below are optional. Installing them might be slow.
# pip install csrc/layer_norm
# If the version of flash-attn is higher than 2.1.1, the following is not needed.
# pip install csrc/rotary
```

> [!Warning]
> <div align="left">
> <b>
> 🚨 We find that using flash attention2 makes performance degradation.
> </b>
> </div>

## 🗝️ Training & Validating
The training & validating instruction is in [TRAIN.md](docs/TRAIN.md) & [EVAL.md](docs/EVAL.md).

## 💡 Customizing your MoE-LLaVA
The instruction is in [CUSTOM.md](docs/CUSTOM.md).

## 😍 Visualization
The instruction is in [VISUALIZATION.md](docs/VISUALIZATION.md).

## 🤖 API
**We open source all codes.** If you want to load the model (e.g. ```LanguageBind/MoE-LLaVA-Phi2-2.7B-4e```) on local, you can use the following code snippets.

**Using the following command to run the code.**

```bash
deepspeed --include localhost:0 predict.py
```

```python
import torch
from PIL import Image
from moellava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from moellava.conversation import conv_templates, SeparatorStyle
from moellava.model.builder import load_pretrained_model
from moellava.utils import disable_torch_init
from moellava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

def main():
    disable_torch_init()
    image = 'moellava/serve/examples/extreme_ironing.jpg'
    inp = 'What is unusual about this image?'
    model_path = 'LanguageBind/MoE-LLaVA-Phi2-2.7B-4e'  # LanguageBind/MoE-LLaVA-Qwen-1.8B-4e or LanguageBind/MoE-LLaVA-StableLM-1.6B-4e
    device = 'cuda'
    load_4bit, load_8bit = False, False  # FIXME: Deepspeed support 4bit or 8bit?
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name, load_8bit, load_4bit, device=device)
    image_processor = processor['image']
    conv_mode = "phi"  # qwen or stablelm
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    image_tensor = image_processor.preprocess(Image.open(image).convert('RGB'), return_tensors='pt')['pixel_values'].to(model.device, dtype=torch.float16)

    print(f"{roles[1]}: {inp}")
    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
    print(outputs)

if __name__ == '__main__':
    main()
```

## 🙌 Related Projects
* [Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA) This framework empowers the model to efficiently utilize the united visual tokens.
* [LanguageBind](https://github.com/PKU-YuanGroup/LanguageBind) An open source five modalities language-based retrieval framework.

## 👍 Acknowledgement
* [LLaVA](https://github.com/haotian-liu/LLaVA) The codebase we built upon and it is an efficient large language and vision assistant.

## 🔒 License
* The majority of this project is released under the Apache 2.0 license as found in the [LICENSE](https://github.com/PKU-YuanGroup/MoE-LLaVA/blob/main/LICENSE) file.
* The service is a research preview intended for non-commercial use only, subject to the model [License](https://github.com/facebookresearch/llama/blob/main/MODEL_CARD.md) of LLaMA, [Terms of Use](https://openai.com/policies/terms-of-use) of the data generated by OpenAI, and [Privacy Practices](https://chrome.google.com/webstore/detail/sharegpt-share-your-chatg/daiacboceoaocpibfodeljbdfacokfjb) of ShareGPT. Please contact us if you find any potential violation.



## ✏️ Citation
If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil:.

```BibTeX
@article{lin2024moe,
  title={MoE-LLaVA: Mixture of Experts for Large Vision-Language Models},
  author={Lin, Bin and Tang, Zhenyu and Ye, Yang and Cui, Jiaxi and Zhu, Bin and Jin, Peng and Zhang, Junwu and Ning, Munan and Yuan, Li},
  journal={arXiv preprint arXiv:2401.15947},
  year={2024}
}
```

```BibTeX
@article{lin2023video,
  title={Video-LLaVA: Learning United Visual Representation by Alignment Before Projection},
  author={Lin, Bin and Zhu, Bin and Ye, Yang and Ning, Munan and Jin, Peng and Yuan, Li},
  journal={arXiv preprint arXiv:2311.10122},
  year={2023}
}
```



## ✨ Star History
[![Star History](https://api.star-history.com/svg?repos=PKU-YuanGroup/MoE-LLaVA&type=Date)](https://star-history.com/#PKU-YuanGroup/MoE-LLaVA&Date)


## 🤝 Contributors

<a href="https://github.com/Bagumeow/ViQwen-_MoE_Llava_Training/graphs/contributors">
  <img src="https://avatars.githubusercontent.com/u/93193119?v=4&size=64" />
</a>
