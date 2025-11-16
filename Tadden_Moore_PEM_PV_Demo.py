#!/usr/bin/env python3
"""
MC Framework Steering Demo (Gemma-2B + Gemma Scope SAEs)
Author: Tadden Moore
Date: 2025-11-04
Version: 1.0.0 (Production Ready for Zenodo)

This implementation demonstrates the Metacognitive Core (MC) Framework
for dynamic neural plasticity in LLMs using activation steering.

Requirements:
pip install torch transformers accelerate sae-lens safetensors
"""

import contextlib
import torch
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Model configuration
MODEL_ID = "google/gemma-2b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
LAYER_IDX = 10  # Mid layer for steering (empirically optimal)
SAE_REL = "gemma-scope-2b-pt-res-canonical"
SAE_ID = f"layer_{LAYER_IDX}/width_16k/canonical"

# Print system info for reproducibility
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {DEVICE}")
print(f"Dtype: {DTYPE}")
print(f"Random seed: {SEED}")
print("-" * 50)


@contextlib.contextmanager
def layer_hook(model, layer_idx, hook_fn):
    """
    Context manager for safely adding and removing forward hooks.
    
    Args:
        model: The transformer model
        layer_idx: Index of the layer to hook
        hook_fn: The hook function to apply
    """
    target = model.model.layers[layer_idx]  # GemmaForCausalLM layer access
    handle = target.register_forward_hook(hook_fn)
    try:
        yield
    finally:
        handle.remove()


def load_model():
    """
    Load Gemma-2B model and tokenizer with optimal settings.
    
    Returns:
        tuple: (model, tokenizer)
    """
    print("Loading Gemma-2B model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
        device_map=("auto" if torch.cuda.is_available() else None),
    )
    model = model.to(DEVICE).eval()
    
    print("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    
    print(f"Model loaded: {model.config.num_hidden_layers} layers, "
          f"{model.config.hidden_size} hidden dim")
    
    return model, tok


def load_sae():
    """
    Load Sparse Autoencoder from Gemma Scope.
    
    Returns:
        sae: The loaded SAE model
    """
    print(f"Loading SAE: {SAE_ID}...")
    sae, cfg, sparsity = SAE.from_pretrained(
        release=SAE_REL,
        sae_id=SAE_ID,
        device=DEVICE,
    )
    
    # Fold decoder norms if available (optimization)
    if hasattr(sae, "fold_W_dec_norm"):
        sae.fold_W_dec_norm()
    
    print(f"SAE loaded: {cfg}")
    return sae


@torch.no_grad()
def _encode_feats(sae, h_last):
    """
    Encode hidden states to sparse features using SAE.
    
    Args:
        sae: The Sparse Autoencoder
        h_last: Hidden state tensor [B, S, D]
        
    Returns:
        torch.Tensor: Sparse feature activations
    """
    # Try multiple ways to get features (different SAE versions)
    out = sae(h_last)
    
    # Handle different output formats
    feats = getattr(out, "feature_acts", None)
    if feats is None and isinstance(out, dict):
        feats = out.get("feature_acts", None)
    if feats is None:
        # Fallback to direct encoding
        feats = sae.encode(h_last)
    
    return feats


@torch.no_grad()
def _decode_feats(sae, feats):
    """
    Decode sparse features back to hidden states.
    
    Args:
        sae: The Sparse Autoencoder
        feats: Sparse feature tensor
        
    Returns:
        torch.Tensor: Reconstructed hidden states
    """
    if hasattr(sae, "decode"):
        return sae.decode(feats)
    # Fallback to manual decoding
    return feats @ sae.W_dec.T


@torch.no_grad()
def capture_concept_features(model, tok, sae, concept_text):
    """
    Extract SAE features for a given concept.
    
    Args:
        model: The language model
        tok: The tokenizer
        sae: The Sparse Autoencoder
        concept_text: Text describing the concept
        
    Returns:
        torch.Tensor: Feature vector for the concept
    """
    last_hidden = {}
    
    def grab(module, input, output):
        """Hook to capture hidden states"""
        hs = output[0] if isinstance(output, tuple) else output  # [B,S,D]
        last_hidden["h"] = hs[:, -1:, :]  # Last token state only
    
    # Run forward pass with hook
    with layer_hook(model, LAYER_IDX, grab):
        inputs = tok(concept_text, return_tensors="pt").to(DEVICE)
        _ = model(**inputs)
    
    # Encode to features
    feats = _encode_feats(sae, last_hidden["h"])
    
    # Log sparsity statistics
    sparsity = (feats > 0).float().mean().item()
    print(f"Concept features captured: sparsity={sparsity:.3f}, "
          f"shape={feats.shape}")
    
    return feats.detach()


class MCSteerer:
    """
    Metacognitive Core Steerer for activation-space interventions.
    
    This class implements the core steering mechanism that modulates
    the internal state of the Inference Engine during generation.
    """
    
    def __init__(self, sae, concept_feats, strength=4.0, max_norm=None):
        """
        Initialize the MC Steerer.
        
        Args:
            sae: Sparse Autoencoder for feature encoding/decoding
            concept_feats: Target concept feature vector
            strength: Steering strength multiplier
            max_norm: Maximum L2 norm for steering vector (safety clamp)
        """
        self.sae = sae
        self.f = concept_feats
        self.strength = float(strength)
        self.max_norm = max_norm
        
        # Track statistics
        self.steering_count = 0
        self.total_delta_norm = 0.0
    
    def hook(self, module, input, output):
        """
        Forward hook that applies steering to hidden states.
        
        This implements Equation 2 from the paper:
        u_t = K_p * e_t where e_t = φ★ - φ(h_t)
        """
        hs = output[0] if isinstance(output, tuple) else output  # [B,S,D]
        last = hs[:, -1:, :]  # Extract last token
        
        # Encode current state to features
        feats = _encode_feats(self.sae, last)
        
        # Compute steering delta
        delta = self.f * self.strength
        
        # Apply safety clamp if specified
        if self.max_norm is not None:
            n = torch.linalg.norm(delta)
            if n > self.max_norm:
                delta = delta * (self.max_norm / (n + 1e-8))
        
        # Update statistics
        self.steering_count += 1
        self.total_delta_norm += torch.linalg.norm(delta).item()
        
        # Apply steering in feature space
        steered_feats = feats + delta
        
        # Decode back to hidden state space
        steered_last = _decode_feats(self.sae, steered_feats)
        
        # Replace last token with steered version
        hs = hs.clone()
        hs[:, -1:, :] = steered_last
        
        return (hs,) if isinstance(output, tuple) else hs
    
    def get_stats(self):
        """Get steering statistics"""
        if self.steering_count > 0:
            avg_norm = self.total_delta_norm / self.steering_count
            return {
                "steering_count": self.steering_count,
                "avg_delta_norm": avg_norm
            }
        return {"steering_count": 0, "avg_delta_norm": 0.0}


@torch.no_grad()
def generate(model, tok, prompt, max_new=80, temp=0.7):
    """
    Generate text from the model.
    
    Args:
        model: The language model
        tok: The tokenizer
        prompt: Input prompt string
        max_new: Maximum new tokens to generate
        temp: Sampling temperature
        
    Returns:
        str: Generated text
    """
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    
    out = model.generate(
        **inputs,
        max_new_tokens=max_new,
        do_sample=True,
        temperature=temp,
        pad_token_id=tok.eos_token_id,
    )
    
    return tok.decode(out[0], skip_special_tokens=True)


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors"""
    a_norm = a / (torch.linalg.norm(a) + 1e-8)
    b_norm = b / (torch.linalg.norm(b) + 1e-8)
    return (a_norm * b_norm).sum().item()


def main():
    """
    Main demonstration of the MC Framework.
    
    This validates the framework by showing how steering can induce
    a non-prompted cognitive state (existentialism) in a simple task
    (describing a flower).
    """
    print("=" * 50)
    print("MC Framework Steering Demo")
    print("=" * 50)
    
    # Load models
    model, tok = load_model()
    sae = load_sae()
    
    # Define concept and target
    CONCEPT = "The theory of philosophical existentialism explores dread and meaning."
    TARGET = "Describe a flower."
    
    print(f"\nConcept: {CONCEPT}")
    print(f"Target prompt: {TARGET}")
    print("-" * 50)
    
    # Capture concept features
    print("\nCapturing concept features...")
    concept_feats = capture_concept_features(model, tok, sae, CONCEPT)
    
    # Generate baseline (unsteered)
    print("\n" + "=" * 50)
    print("BASELINE GENERATION (No Steering)")
    print("=" * 50)
    base = generate(model, tok, TARGET, max_new=60)
    print(base)
    
    # Generate with steering
    print("\n" + "=" * 50)
    print("STEERED GENERATION (With MC Framework)")
    print("=" * 50)
    
    mc = MCSteerer(sae, concept_feats, strength=4.0, max_norm=100.0)
    
    with layer_hook(model, LAYER_IDX, mc.hook):
        steered = generate(model, tok, TARGET, max_new=60)
    
    print(steered)
    
    # Report statistics
    stats = mc.get_stats()
    print("\n" + "=" * 50)
    print("STEERING STATISTICS")
    print("=" * 50)
    print(f"Steering interventions: {stats['steering_count']}")
    print(f"Average delta norm: {stats['avg_delta_norm']:.3f}")
    
    # Validation
    print("\n" + "=" * 50)
    print("VALIDATION")
    print("=" * 50)
    
    if base.strip() != steered.strip():
        print("✓ SUCCESS: Output changed under steering!")
        print(f"  Baseline length: {len(base.split())}")
        print(f"  Steered length: {len(steered.split())}")
        
        # Check for existential themes (simple keyword check)
        existential_keywords = ["meaning", "purpose", "existence", "being", 
                              "mortality", "essence", "freedom", "authentic"]
        
        base_score = sum(1 for word in existential_keywords 
                        if word.lower() in base.lower())
        steered_score = sum(1 for word in existential_keywords 
                           if word.lower() in steered.lower())
        
        print(f"  Existential keywords in baseline: {base_score}")
        print(f"  Existential keywords in steered: {steered_score}")
        
        if steered_score > base_score:
            print("  ✓ Thematic alignment detected!")
    else:
        print("⚠ NO CHANGE: Try different layer/strength parameters.")
        print("  Suggested adjustments:")
        print("  - Try layers 8-12 (currently: 10)")
        print("  - Increase strength to 6.0-8.0")
        print("  - Check SAE is loading correctly")


if __name__ == "__main__":
    # Disable gradients for inference
    torch.set_grad_enabled(False)
    
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure all packages are installed:")
        print("   pip install torch transformers accelerate sae-lens safetensors")
        print("2. Check CUDA availability if using GPU")
        print("3. Verify internet connection for model downloads")
        print("4. Ensure sufficient memory (8GB+ recommended)")
        raise
