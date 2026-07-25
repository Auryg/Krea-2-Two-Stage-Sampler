# Krea 2 Two Stage Sampler



Right now this repo includes two nodes for ComfyUI:



A Sigma-locked two-stage sampler with separate inputs for models for each.  The general thinking was to run several steps of a base/raw model for better variation between seeds, and then finish it off with an extracted turbo lora on the second stage for both speed and possibly higher quality. It also has support for running two resolutions, so you can run the first stage at a lower, faster resolution.  The neat thing is that you can choose a percentage and any amount of steps for the first and second stage and it will change over with the original noise (if not upscaling) at just the right sigma values.



Because of that I've also included a dual resolution node—select the aspect ratio and the base and final megapixels. It includes random modes covering all ratios, vertical ratios, horizontal ratios, or a constrained set (1:1, 4:5, 5:4, 2:3, 3:2, 3:4, and 4:3). The included aspect ratios are specifically tailored for Krea 2.



The main knob you'll want to play with is `handoff_percent`, which sets the point in the denoising process where stage 1 hands off to stage 2. For example, at 25%, stage 1 handles the first 25% and stage 2 handles the remaining 75%. At 0%, stage 2 performs the full generation; at 100%, stage 1 performs the full generation. There's no single right answer for where it should be set.

Installation: Put in the custom_nodes folder or grab from ComfyUI manager. 



Here's an image with a sample workflow embedded:

![Krea 2 raw-to-turbo LoRA workflow](images/TwoStageKrea.png)

