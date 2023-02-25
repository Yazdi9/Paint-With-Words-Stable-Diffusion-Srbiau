
Subject:paint with word with Stable Diffiusion models 

final  project for computer vision SRBIAU Dr.Koochari by Saba Hesaraki 


Article : eDiff-I: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers     Link:https://arxiv.org/pdf/2211.01324.pdf


Website for read more: https://www.unite.ai/nvidias-ediffi-diffusion-model-allows-painting-with-words-and-more/


All of Results Exists on content/demo  and output_example  folder  


Unofficial [🤗 huggingface/diffusers](https://github.com/huggingface/diffusers)-based implementation of Paint-with-Words proposed by the paper *eDiff-I: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers*. 
This implementation is based on [cloneofsimo/paint-with-words-sd](https://github.com/cloneofsimo/paint-with-words-sd).

## Subtle Control of the Image Generation

<!-- #region -->
<p align="center">
<img  src="contents/rabbit_mage.jpg">
</p>
<!-- #endregion -->

> Notice how without PwW the cloud is missing.

<!-- #region -->
<p align="center">
<img  src="contents/road.jpg">
</p>
<!-- #endregion -->

> Notice how without PwW, abandoned city is missing, and road becomes purple as well.

## Shift the object : Same seed, just the segmentation map's positional difference

<!-- #region -->
<p align="center">
<img  src="contents/aurora_1_merged.jpg">
</p>
<!-- #endregion -->

<!-- #region -->
<p align="center">
<img  src="contents/aurora_2_merged.jpg">
</p>
<!-- #endregion -->

> "A digital painting of a half-frozen lake near mountains under a full moon and aurora. A boat is in the middle of the lake. Highly detailed."

> Notice how nearly all of the composition remains the same, other than the position of the moon.

---

Recently, researchers from NVIDIA proposed [eDiffi](https://arxiv.org/abs/2211.01324). In the paper, they suggested method that allows "painting with word". Basically, this is like make-a-scene, but with just using adjusted cross-attention score. You can see the results and detailed method in the paper.

Their paper and their method was not open-sourced. Yet, paint-with-words can be implemented with Stable Diffusion since they share common Cross Attention module. So, I implemented it with Stable Diffusion.

<!-- #region -->
<p align="center">
<img  src="contents/paint_with_words_figure.png">
</p>
<!-- #endregion -->

# Installation

```bash
pip install paint-with-words-pipeline
```

# Basic Usage

Prepare segmentation map, and map-color : tag label such as below. keys are (R, G, B) format, and values are tag label.

```python
{
    (0, 0, 0): "cat,1.0",
    (255, 255, 255): "dog,1.0",
    (13, 255, 0): "tree,1.5",
    (90, 206, 255): "sky,0.2",
    (74, 18, 1): "ground,0.2",
}
```

You neeed to have them so that they are in format "{label},{strength}", where strength is additional weight of the attention score you will give during generation, i.e., it will have more effect.

```python
import torch
from paint_with_words.pipelines import PaintWithWordsPipeline

settings = {
    "color_context": {
        (0, 0, 0): "cat,1.0",
        (255, 255, 255): "dog,1.0",
        (13, 255, 0): "tree,1.5",
        (90, 206, 255): "sky,0.2",
        (74, 18, 1): "ground,0.2",
    },
    "color_map_img_path": "contents/example_input.png",
    "input_prompt": "realistic photo of a dog, cat, tree, with beautiful sky, on sandy ground",
    "output_img_path": "contents/output_cat_dog.png",
}

color_map_image_path = settings["color_map_img_path"]
color_context = settings["color_context"]
input_prompt = settings["input_prompt"]

# load pre-trained weight with paint with words pipeline
pipe = PaintWithWordsPipeline.from_pretrained(
    model_name,
    revision="fp16",
    torch_dtype=torch.float16,
)
pipe.safety_checker = None  # disable the safety checker
pipe.to("cuda")

# load color map image
color_map_image = Image.open(color_map_image_path).convert("RGB")

with torch.autocast("cuda"):
    image = pipe(
        prompt=input_prompt,
        color_context=color_context,
        color_map_image=color_map_image,
        latents=latents,
        num_inference_steps=30,
    ).images[0]

img.save(settings["output_img_path"])
```

---

# Weight Scaling

In the paper, they used $w \log (1 + \sigma)  \max (Q^T K)$ to scale appropriate attention weight. However, this wasn't optimal after few tests, found by [CookiePPP](https://github.com/AUTOMATIC1111/stable-diffusion-webui/issues/4406). You can check out the effect of the functions below:

<!-- #region -->
<p align="center">
<img  src="contents/compare_std.jpg">
</p>
<!-- #endregion -->

> $w' = w \log (1 + \sigma)  std (Q^T K)$

<!-- #region -->
<p align="center">
<img  src="contents/compare_max.jpg">
</p>
<!-- #endregion -->

> $w' = w \log (1 + \sigma)  \max (Q^T K)$

<!-- #region -->
<p align="center">
<img  src="contents/compare_log2_std.jpg">
</p>
<!-- #endregion -->

> $w' = w \log (1 + \sigma^2)  std (Q^T K)$

You can define your own weight function and further tweak the configurations by defining `weight_function` argument in the `PaintWithWordsPipeline`.

Example:

```python
def weight_function(
    w: torch.Tensor, 
    sigma: torch.Tensor, 
    qk: torch.Tensor,
) -> torch.Tensor:
    return 0.4 * w * math.log(sigma ** 2 + 1) * qk.std()

with torch.autocast("cuda"):
    image = pipe(
        prompt=input_prompt,
        color_context=color_context,
        color_map_image=color_map_image,
        latents=latents,
        num_inference_steps=30,
        #
        # set the weight function here:
        weight_function=weight_function,
        #
    ).images[0]
```

## More on the weight function, (but higher)

<!-- #region -->
<p align="center">
<img  src="contents/compare_4_std.jpg">
</p>
<!-- #endregion -->

> $w' = w \log (1 + \sigma)  std (Q^T K)$

<!-- #region -->
<p align="center">
<img  src="contents/compare_4_max.jpg">
</p>
<!-- #endregion -->

> $w' = w \log (1 + \sigma)  \max (Q^T K)$

<!-- #region -->
<p align="center">
<img  src="contents/compare_4_log2_std.jpg">
</p>
<!-- #endregion -->

> $w' = w \log (1 + \sigma^2)  std (Q^T K)$




