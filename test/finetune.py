"""
Convert LoRA Adapter to GGUF for Ollama
Step-by-step conversion pipeline
"""

import torch
import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ============================================================================
# STEP 1: MERGE LORA WITH BASE MODEL
# ============================================================================

print("="*70)
print("STEP 1: MERGING LORA ADAPTER WITH BASE MODEL")
print("="*70)

# Paths
MODEL_ID = "meta-llama/Llama-3.2-1B"
ADAPTER_PATH = "./sql_agent_adapter_final"
MERGED_OUTPUT_PATH = "./sql_agent_merged"

print(f"\n📥 Loading base model: {MODEL_ID}")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="cpu",
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)

print(f"📥 Loading LoRA adapter: {ADAPTER_PATH}")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

print("\n🔀 Merging LoRA weights into base model...")
merged_model = model.merge_and_unload()  # This creates a standalone model

print(f"\n💾 Saving merged model to: {MERGED_OUTPUT_PATH}")
merged_model.save_pretrained(MERGED_OUTPUT_PATH)

# Also save tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.save_pretrained(MERGED_OUTPUT_PATH)

print("✓ Merge complete!")
print(f"   Merged model saved to: {MERGED_OUTPUT_PATH}")

# ============================================================================
# STEP 2: CONVERT TO GGUF USING LLAMA.CPP
# ============================================================================

print("\n" + "="*70)
print("STEP 2: CONVERT TO GGUF FORMAT")
print("="*70)

print("""
📋 Manual Conversion Steps (requires llama.cpp):

1. Clone llama.cpp repository:
   git clone https://github.com/ggerganov/llama.cpp.git
   cd llama.cpp

2. Install Python requirements:
   pip install -r requirements.txt

3. Convert HuggingFace model to GGUF:
   python convert_hf_to_gguf.py {MERGED_OUTPUT_PATH} --outfile sql_agent.gguf --outtype f16

4. (Optional) Quantize to 4-bit for smaller size:
   ./llama-quantize sql_agent.gguf sql_agent_q4_k_m.gguf Q4_K_M

5. Result:
   - sql_agent.gguf (FP16, ~2GB)
   - sql_agent_q4_k_m.gguf (4-bit, ~600MB) ← Recommended for Ollama

Quantization Options:
   Q4_K_M  : 4-bit, best quality/size balance (recommended)
   Q5_K_M  : 5-bit, higher quality, larger file
   Q8_0    : 8-bit, near-original quality, 2x size of Q4
""")

# ============================================================================
# AUTOMATED CONVERSION (if llama.cpp is available)
# ============================================================================

LLAMA_CPP_PATH = "llama.cpp"  # Update this to your llama.cpp directory

if os.path.exists(LLAMA_CPP_PATH):
    print("\n🔄 Attempting automated conversion...")
    
    import subprocess
    
    # Check if conversion script exists
    convert_script = os.path.join(LLAMA_CPP_PATH, "convert_hf_to_gguf.py")
    
    if os.path.exists(convert_script):
        try:
            # Convert to FP16 GGUF
            print("\n1️⃣ Converting to FP16 GGUF...")
            result = subprocess.run([
                sys.executable,
                convert_script,
                MERGED_OUTPUT_PATH,
                "--outfile", "sql_agent_fp16.gguf",
                "--outtype", "f16"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✓ FP16 GGUF created: sql_agent_fp16.gguf")
                
                # Quantize to Q4_K_M
                print("\n2️⃣ Quantizing to Q4_K_M...")
                quantize_exe = os.path.join(LLAMA_CPP_PATH, "llama-quantize.exe")  # Windows
                if not os.path.exists(quantize_exe):
                    quantize_exe = os.path.join(LLAMA_CPP_PATH, "llama-quantize")  # Linux/Mac
                
                if os.path.exists(quantize_exe):
                    result = subprocess.run([
                        quantize_exe,
                        "sql_agent_fp16.gguf",
                        "sql_agent_q4_k_m.gguf",
                        "Q4_K_M"
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        print("✓ Q4_K_M GGUF created: sql_agent_q4_k_m.gguf")
                        
                        # Show file sizes
                        fp16_size = os.path.getsize("sql_agent_fp16.gguf") / (1024**2)
                        q4_size = os.path.getsize("sql_agent_q4_k_m.gguf") / (1024**2)
                        print(f"\nFile sizes:")
                        print(f"  FP16: {fp16_size:.1f} MB")
                        print(f"  Q4_K_M: {q4_size:.1f} MB (recommended)")
                    else:
                        print("⚠️  Quantization failed. Manual steps required.")
                else:
                    print("⚠️  llama-quantize not found. Manual quantization required.")
            else:
                print("⚠️  Conversion failed:", result.stderr)
                print("Please follow manual steps above.")
        
        except Exception as e:
            print(f"⚠️  Automated conversion failed: {e}")
            print("Please follow manual steps above.")
    else:
        print(f"⚠️  Conversion script not found at {convert_script}")
        print("Please follow manual steps above.")
else:
    print(f"\n⚠️  llama.cpp not found at {LLAMA_CPP_PATH}")
    print("Please follow manual steps above.")

print("\n" + "="*70)
print("CONVERSION GUIDE COMPLETE")
print("="*70)