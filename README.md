# Krea 2 Two Stage Sampler



Right now this repo includes two nodes for ComfyUI:



A Sigma-locked two-stage sampler with separate inputs for models for each.  The general thinking was to run several steps of a base/raw model for better variation between seeds, and then finish it off with an extracted turbo lora on the second stage for both speed and possibly higher quality. It also has support for running two resolutions, so you can run the first stage at a lower, faster resolution.



Because of that I've also includes a dual resolution node - select the aspect ratio and the base and final megapixels.  It also includes a random mode for the aspect ratio.  The included aspect ratios are specifically tailored for Krea 2.



The main know you'll want to play with is the handoff\_percent.  There's no right answer on what it should be.  

Installation: Put in the custom_nodes folder or grab from ComfyUI manager. 



Here's an image with a sample workflow:

!\[Krea 2 raw-to-turbo LoRA workflow](images/TwoStageKrea.png)

