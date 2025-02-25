---
title: CountGD_Multi-Modal_Open-World_Counting
sdk: docker
app_port: 7860
---
# CountGD: Multi Modal Open World Counting Model

To Run Locally, the best method is to use docker.

Make sure you have installed docker, nvidia-driver and nvidia container toolkit for the your platform.

Then, you can run the app locally with the following command

```bash
docker run -it \
    --name countgd \
    -p 7860:7860 \
    --platform=linux/amd64 \
    --gpus all \
	registry.hf.space/nikigoli-countgd:latest \
    python app.py
```