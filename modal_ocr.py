import modal

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("git")
    .pip_install(
        "wheel",
        "packaging",
        "ninja",
        "fastapi[standard]",
        "torch==2.6.0",
        "torchvision==0.21.0",
        "transformers==4.46.3",
        "tokenizers==0.20.3",
        "einops",
        "addict",
        "easydict",
        "pillow",
        "accelerate",
    )
    .pip_install(
        "flash-attn==2.7.3",
        extra_options="--no-build-isolation",
    )
)

app = modal.App("deepseek-ocr")

@app.cls(
    image=image,
    gpu="A10G",
    scaledown_window=300,
    timeout=600,
)
class OCRModel:
    @modal.enter()
    def load(self):
        from transformers import AutoModel, AutoTokenizer
        import torch

        model_name = "deepseek-ai/DeepSeek-OCR-2"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            _attn_implementation="flash_attention_2",
            trust_remote_code=True,
            use_safetensors=True,
        )
        self.model = self.model.eval().cuda().to(torch.bfloat16)

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def ocr(self, payload: dict):
        import base64
        import os
        import tempfile
        import shutil

        image_b64 = payload["image_base64"]
        img_bytes = base64.b64decode(image_b64)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_in:
            tmp_in.write(img_bytes)
            input_path = tmp_in.name

        output_dir = tempfile.mkdtemp()
        prompt = "<image>\nFree OCR. "

        try:
            result = self.model.infer(
                self.tokenizer,
                prompt=prompt,
                image_file=input_path,
                output_path=output_dir,
                base_size=1024,
                image_size=768,
                crop_mode=True,
                save_results=True,
            )

            extracted_text = ""
            if isinstance(result, str) and result.strip():
                extracted_text = result
            else:
                for root, _dirs, files in os.walk(output_dir):
                    for fname in files:
                        if fname.endswith((".md", ".txt", ".mmd")):
                            with open(os.path.join(root, fname), "r", encoding="utf-8") as f:
                                extracted_text = f.read()
                            break
                    if extracted_text:
                        break

            debug_files = []
            for root, _dirs, files in os.walk(output_dir):
                for fname in files:
                    debug_files.append(os.path.join(root, fname))

            return {
                "text": extracted_text,
                "debug_output_files": debug_files,
                "debug_result_type": str(type(result)),
                "debug_result_preview": str(result)[:500] if result else None,
            }
        finally:
            try:
                os.unlink(input_path)
            except OSError:
                pass
            shutil.rmtree(output_dir, ignore_errors=True)