# Roop-Unleashed-Runpod
Roop Unleashed Runpod

------------------------

I changed a few things in the "requirements.txt".

Changes:

I deleted these plugins, because they were interfering with the functionality:

- "torch==2.5.1; sys_platform == 'darwin'"
- "torchvision==0.20.1; sys_platform == 'darwin'"
- "onnxruntime-silicon==1.16.3; sys_platform == 'darwin' and platform_machine == 'arm64'"

- I changed "onnxruntime-gpu==1.20.1; sys_platform != 'darwin'" to "onnxruntime-gpu==1.19.0; sys_platform != 'darwin'", because it didn't work for me otherwise.

------------------------

I also changed a few things in "settings.py" to suit my taste.

Changes:

- "self.server_share = self.default_get(data, 'server_share', True)" - This must be done anyway, because otherwise you can't open the page. So now you don't have to keep dealing with rewriting it every time you use a new pod.
- "self.output_video_codec = self.default_get(data, 'output_video_codec', 'libx265')"
- "self.clear_output = self.default_get(data, 'clear_output', False)"
- "self.max_threads = self.default_get(data, 'max_threads', 4)"

------------------------

There is a runpod.txt file in the folder that needs to be followed, and it will work.

------------------------

It works with the "RunPod Pytorch 2.4.0
runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04" template.

------------------------

Credit goes to https://github.com/C0untFloyd/roop-unleashed

------------------------
