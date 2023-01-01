# Paint-with-Words, Implemented with Stable diffusion

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
pip install git+https://github.com/cloneofsimo/paint-with-words-sd.git
```







