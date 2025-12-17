# -*- coding: utf-8 -*-
"""
推理引擎 (API适配版)
负责模型的加载、显存优化及图片生成。
返回结构化数据而非 UI 字符串。
"""
import torch
from diffusers import DiffusionPipeline # type: ignore
import gc
import time
from core.utils import detect_device, get_torch_dtype
from core.lora_manager import LoRAMerger
import config

class ZImageEngine:
    def __init__(self):
        self.pipe = None
        self.device = None
        self.dtype = None
        self.lora_merger = None
        self.current_lora_applied = False

    def is_loaded(self):
        return self.pipe is not None

    def load_model(self):
        """加载模型 (自动检测设备)"""
        self.device = detect_device()
        self.dtype = get_torch_dtype(self.device)
        
        print(f"🚀 [Engine] 正在加载模型... 设备: {self.device.upper()}, 精度: {self.dtype}")
        
        # 清理旧显存
        if self.pipe:
            del self.pipe
            self.pipe = None
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            if torch.backends.mps.is_available(): torch.mps.empty_cache()

        try:
            self.pipe = DiffusionPipeline.from_pretrained(
                config.MODEL_PATH,
                torch_dtype=self.dtype,
                trust_remote_code=True,
            )
            self.pipe.to(self.device)
            
            self.lora_merger = LoRAMerger(self.pipe)
            self.current_lora_applied = False
            
            self._apply_optimizations()
            
            print("✅ [Engine] 模型加载完毕。")
            return True, f"就绪 ({self.device.upper()})"
            
        except Exception as e:
            print(f"❌ [Engine] 加载失败: {e}")
            return False, str(e)

    def _apply_optimizations(self):
        """应用优化策略"""
        # VAE 强制 FP32
        if hasattr(self.pipe, "vae"):
            self.pipe.vae.to(dtype=torch.float32) # pyright: ignore[reportOptionalMemberAccess]
            self.pipe.vae.config.force_upcast = True # pyright: ignore[reportOptionalMemberAccess]

        # 硬件特定优化
        if self.device == "mps":
            # MPS 显存足够时关闭 Tiling 以获得最佳画质
            pass 
        elif self.device == "cuda":
            self.pipe.enable_model_cpu_offload() # pyright: ignore[reportOptionalMemberAccess]
            if hasattr(self.pipe, "enable_vae_tiling"):
                self.pipe.enable_vae_tiling() # pyright: ignore[reportOptionalMemberAccess]

    def update_lora(self, enable, scale):
        """更新 LoRA 状态"""
        if not self.is_loaded(): return
        
        # 简化逻辑：状态变更则重载模型
        if (not enable and self.current_lora_applied) or (enable and self.current_lora_applied):
            print("🔄 [Engine] LoRA 变更，重载模型...")
            self.load_model()
            if enable:
                self.lora_merger.load_lora_weights(config.LORA_PATH, scale) # pyright: ignore[reportOptionalMemberAccess]
                self.current_lora_applied = True
        elif enable and not self.current_lora_applied:
            self.lora_merger.load_lora_weights(config.LORA_PATH, scale) # pyright: ignore[reportOptionalMemberAccess]
            self.current_lora_applied = True

    def generate(self, prompt, neg_prompt, steps, cfg, width, height, seed, seed_mode):
        """
        生成图片
        Returns:
            dict: { "image": PIL_Image, "seed": int, "duration": float }
        """
        start_time = time.time()
        
        # 显存清理
        gc.collect()
        if self.device == "mps": torch.mps.empty_cache()
        if self.device == "cuda": torch.cuda.empty_cache()

        # 种子处理
        if seed_mode == "random" or seed == -1:
            actual_seed = torch.randint(0, 2**32 - 1, (1,)).item()
        else:
            actual_seed = int(seed)
            
        gen_device = "cpu" if self.device == "mps" else self.device
        generator = torch.Generator(gen_device).manual_seed(actual_seed) # pyright: ignore[reportArgumentType]

        print(f"🎨 [Generate] 尺寸: {width}x{height} | 步数: {steps} | 种子: {actual_seed}")

        try:
            image = self.pipe(prompt=prompt,negative_prompt=neg_prompt,num_inference_steps=steps,guidance_scale=cfg,width=width,height=height,generator=generator).images[0] # type: ignore
            
            duration = time.time() - start_time
            
            return {
                "success": True,
                "image": image,
                "seed": actual_seed,
                "duration": round(duration, 2)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }