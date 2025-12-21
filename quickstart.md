```bash
conda install -c conda-forge libstdcxx-ng
git clone https://github.com/sy75452381/ComfyUI-SeedVR2_VideoUpscaler.git
cd ComfyUI-SeedVR2_VideoUpscaler
pip install torch==2.8.0 torchvision torchaudio
pip install -r requirements.txt
pip install flashinfer-python
pip install flash-attn --no-build-isolation --no-cache-dir
pip install sageattention
pip install "numpy<2.0.0"
python webui.py --port 8000
```