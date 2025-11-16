"""
BLIP-2 Optimization Experiments and Improvements

This file explores potential optimizations and improvements to BLIP-2:
1. Dynamic query allocation
2. Progressive query training
3. Multi-scale visual features
4. Efficient attention mechanisms
5. Quantization strategies
6. Hybrid training approaches

Each optimization is motivated by BLIP-2's design and aims to improve:
- Computational efficiency
- Memory footprint
- Generation quality
- Zero-shot performance
"""

import time
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from data_utils import ImageDataLoader, SimpleVisionEncoder

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
data_loader = ImageDataLoader(device=device)
vision_encoder = SimpleVisionEncoder(patch_size=16, embed_dim=768).to(device)


# ============================================
# OPTIMIZATION 1: Dynamic Query Allocation
# ============================================


class DynamicQueryAllocator(nn.Module):
    """
    OPTIMIZATION: Adaptive number of queries based on image complexity

    MOTIVATION: Not all images need 32 queries. Simple images (e.g., single object)
    might only need 8-16 queries, while complex scenes might benefit from 48-64.

    BENEFIT:
    - Reduces computation for simple images
    - Allocates more capacity for complex images
    - Improves efficiency without sacrificing quality
    """

    def __init__(self, base_queries: int = 32, min_queries: int = 8, max_queries: int = 64, query_dim: int = 768):
        super().__init__()
        self.base_queries = base_queries
        self.min_queries = min_queries
        self.max_queries = max_queries
        self.query_dim = query_dim

        self.query_pool = nn.Parameter(torch.randn(max_queries, query_dim))
        nn.init.trunc_normal_(self.query_pool, std=0.02)

    def estimate_complexity(self, image_features: torch.Tensor) -> float:
        """
        Estimate image complexity using feature statistics

        Args:
            image_features: [batch, num_patches, dim]

        Returns:
            complexity: Scalar between 0 and 1
        """
        # Check Feature variance
        feature_variance = image_features.var(dim=1).mean(dim=1)

        # Check Spatial variance
        patch_mean = image_features.mean(dim=2, keepdim=True)
        spatial_variance = ((image_features - patch_mean) ** 2).mean(dim=(1, 2))

        # Check Attention entropy: Compute pairwise similarity between patches
        normed_features = F.normalize(image_features, dim=-1)
        similarity = torch.bmm(normed_features, normed_features.transpose(1, 2))
        attention_weights = F.softmax(similarity, dim=-1)
        entropy = -(attention_weights * torch.log(attention_weights + 1e-9)).sum(dim=-1).mean(dim=1)

        # Weighted combination
        variance_norm = (feature_variance - feature_variance.min()) / (
            feature_variance.max() - feature_variance.min() + 1e-9
        )
        spatial_norm = (spatial_variance - spatial_variance.min()) / (
            spatial_variance.max() - spatial_variance.min() + 1e-9
        )
        entropy_norm = (entropy - entropy.min()) / (entropy.max() - entropy.min() + 1e-9)

        complexity_score = 0.3 * variance_norm + 0.3 * spatial_norm + 0.4 * entropy_norm

        return complexity_score.mean().item()

    def forward(self, image_features: torch.Tensor, training: bool = False) -> Tuple[torch.Tensor, int]:
        """
        Args:
            image_features: [batch, num_patches, dim]
            training: If True, use ground truth complexity; else predict

        Returns:
            queries: [batch, num_queries, dim]
            num_queries: Number of queries allocated
        """

        batch_size = image_features.shape[0]

        # Estimate image complexity using feature statistics
        complexity = self.estimate_complexity(image_features)

        # Determine number of queries based on complexity
        num_queries = int(self.min_queries + complexity * (self.max_queries - self.min_queries))
        num_queries = max(self.min_queries, min(num_queries, self.max_queries))

        # Select queries from pool
        queries = repeat(self.query_pool[:num_queries], "n d -> b n d", b=batch_size)

        return queries, num_queries


def demo_dynamic_query_allocation():
    """Demonstrate dynamic query allocation"""
    print("\n" + "=" * 80)
    print("OPTIMIZATION 1: Dynamic Query Allocation")
    print("=" * 80)

    batch_size = 4
    patch_dim = 768

    allocator = DynamicQueryAllocator(base_queries=32, min_queries=8, max_queries=64, query_dim=patch_dim).to(device)

    simple_class_id = 0
    complex_class_id = 6

    simple_images = data_loader.get_cifar_batch_by_class(batch_size, class_id=simple_class_id)
    complex_images = data_loader.get_cifar_batch_by_class(batch_size, class_id=complex_class_id)

    # Encode images
    with torch.no_grad():
        simple_features = vision_encoder(simple_images)
        complex_features = vision_encoder(complex_images)

    print(f"Image feature shape: {simple_features.shape}")
    print(f"[batch_size={batch_size}, num_patches=196, embed_dim=768]\n")

    # Analyze simple images
    simple_class_name = data_loader.get_class_name(simple_class_id)
    print(f"Simple Images (CIFAR-10 Class {simple_class_id}: {simple_class_name.capitalize()}):")
    print("-" * 80)
    simple_complexity = allocator.estimate_complexity(simple_features)
    simple_queries, simple_num = allocator(simple_features)

    print(f"Complexity score: {simple_complexity:.3f}")
    print(f"Allocated queries: {simple_num}/{allocator.max_queries}")
    print(f"Query shape: {simple_queries.shape}")
    print(f"Computational savings vs base (32): {(32 - simple_num) / 32 * 100:.1f}%")
    print(f"Memory savings: {(32 - simple_num) * patch_dim * 4 / 1024:.2f} KB per image")

    # Analyze complex images
    complex_class_name = data_loader.get_class_name(complex_class_id)
    print(f"\nComplex Images (CIFAR-10 Class {complex_class_id}: {complex_class_name.capitalize()}):")
    print("-" * 80)
    complex_complexity = allocator.estimate_complexity(complex_features)
    complex_queries, complex_num = allocator(complex_features)

    print(f"Complexity score: {complex_complexity:.3f}")
    print(f"Allocated queries: {complex_num}/{allocator.max_queries}")
    print(f"Query shape: {complex_queries.shape}")
    print(f"Additional capacity vs base (32): {(complex_num - 32) / 32 * 100:.1f}%")
    print(f"Additional memory: {(complex_num - 32) * patch_dim * 4 / 1024:.2f} KB per image")

    # Batch analysis across different complexity levels
    print("\nBATCH COMPLEXITY ANALYSIS")
    print("-" * 80)

    num_samples = 20
    complexities = []
    query_counts = []

    for i in range(0, num_samples * batch_size, batch_size):
        images = data_loader.get_cifar_batch(batch_size, start_idx=i)
        with torch.no_grad():
            features = vision_encoder(images)
        complexity = allocator.estimate_complexity(features)
        _, num_q = allocator(features)

        complexities.append(complexity)
        query_counts.append(num_q)

    print(f"Samples analyzed: {num_samples} batches")
    print(f"Complexity range: [{min(complexities):.3f}, {max(complexities):.3f}]")
    print(f"Query allocation range: [{min(query_counts)}, {max(query_counts)}]")
    print(f"Average queries: {sum(query_counts) / len(query_counts):.1f}")
    print(f"Std deviation: {torch.tensor(query_counts, dtype=torch.float).std().item():.1f}")


# ============================================
# OPTIMIZATION 2: Progressive Query Training
# ============================================


class ProgressiveQueryTrainer:
    """
    OPTIMIZATION: Gradually increase number of queries during training

    MOTIVATION: Starting with fewer queries and gradually adding more:
    - Provides better training stability
    - Each query learns specialized features
    - Reduces overfitting in early stages

    TRAINING SCHEDULE:
    - Epochs 0-20: 8 queries
    - Epochs 21-40: 16 queries
    - Epochs 41-60: 24 queries
    - Epochs 61+: 32 queries (full capacity)
    """

    def __init__(self, max_queries: int = 32, query_dim: int = 768):
        self.max_queries = max_queries
        self.query_dim = query_dim
        self.schedule = {0: 8, 20: 16, 40: 24, 60: 32}

    def get_num_queries(self, epoch: int) -> int:
        """Get number of queries for current epoch"""
        num_queries = 8
        for threshold, queries in self.schedule.items():
            if epoch >= threshold:
                num_queries = queries
        return num_queries


def demo_progressive_query_training():
    """Demonstrate progressive query training schedule"""
    print("\n" + "=" * 50)
    print("OPTIMIZATION 2: Progressive Query Training")
    print("=" * 50)

    trainer = ProgressiveQueryTrainer(max_queries=32, query_dim=768)

    print(f"\n{'Epoch Range':<20} {'Num Queries':<15} {'Parameters':<20}")
    print("-" * 50)

    ranges = [(0, 19), (20, 39), (40, 59), (60, 100)]

    for start, end in ranges:
        num_queries = trainer.get_num_queries(start)
        params = num_queries * 768
        print(f"{f'{start}-{end}':<20} {num_queries:<15} {f'{params:,}':<20}")


# ============================================
# OPTIMIZATION 3: Multi-Scale Visual Features
# ============================================


class MultiScaleQFormer(nn.Module):
    """
    OPTIMIZATION: Use multiple visual feature resolutions

    MOTIVATION: Different queries might benefit from different scales:
    - Fine-grained queries: High-res features (24x24 patches)
    - Global queries: Low-res features (6x6 patches)
    - Medium queries: Mid-res features (12x12 patches)

    BENEFIT: Better capture of both local details and global context
    """

    def __init__(
        self,
        num_queries: int = 32,
        query_dim: int = 768,
        scales: List[int] = [6, 12, 24],
    ):
        super().__init__()
        self.num_queries = num_queries
        self.query_dim = query_dim
        self.scales = scales
        queries_per_scale = num_queries // len(scales)

        # Separate query sets for each scale
        self.scale_queries = nn.ParameterList([nn.Parameter(torch.randn(queries_per_scale, query_dim)) for _ in scales])

        # Scale-specific cross-attention
        self.scale_cross_attn = nn.ModuleList(
            [nn.MultiheadAttention(embed_dim=query_dim, num_heads=12, batch_first=True) for _ in scales]
        )

        # Fusion layer to combine multi-scale queries
        self.fusion = nn.Sequential(nn.LayerNorm(query_dim), nn.Linear(query_dim, query_dim))

    def forward(self, image_features_multiscale: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            image_features_multiscale: List of [batch, patches_i, dim] for each scale

        Returns:
            fused_queries: [batch, num_queries, dim]
        """

        batch_size = image_features_multiscale[0].shape[0]
        scale_outputs = []

        for scale_idx, (queries, cross_attn, features) in enumerate(
            zip(self.scale_queries, self.scale_cross_attn, image_features_multiscale)
        ):
            q = repeat(queries, "n d -> b n d", b=batch_size)

            output, _ = cross_attn(query=q, key=features, value=features)
            scale_outputs.append(output)
        all_queries = torch.cat(scale_outputs, dim=1)
        fused = self.fusion(all_queries)

        return fused


def demo_multiscale_qformer():
    """Demonstrate multi-scale Q-Former"""
    print("\n" + "=" * 80)
    print("OPTIMIZATION 3: Multi-Scale Visual Features")
    print("=" * 80)

    batch_size = 4
    query_dim = 768
    scales = [6, 12, 24]

    images = data_loader.get_cifar_batch(batch_size)

    # Create multi-scale encoders
    multiscale_encoders = []
    for scale_size in scales:
        encoder = SimpleVisionEncoder(patch_size=224 // scale_size, embed_dim=query_dim).to(device)
        multiscale_encoders.append(encoder)

    # Encode features
    with torch.no_grad():
        multiscale_features = []
        for encoder in multiscale_encoders:
            features = encoder(images)
            multiscale_features.append(features)

    print("\nMulti-Scale Feature Pyramid:")
    for scale, features in zip(scales, multiscale_features):
        print(f"Scale {scale}: {features.shape} - {scale*scale} patches")

    # Create multi-scale Q-Former
    ms_qformer = MultiScaleQFormer(num_queries=30, query_dim=query_dim, scales=scales).to(device)

    with torch.no_grad():
        fused_queries = ms_qformer(multiscale_features)

    print(f"\nFused Output: {fused_queries.shape}")

    print("\nQuery Allocation:")
    print(f"Fine-grained features: {scales[2]}x{scales[2]} scale - Local details")
    print(f"Medium features: {scales[1]}x{scales[1]} scale - Object parts")
    print(f"Coarse features: {scales[0]}x{scales[0]} scale - Global context")


# ============================================
# OPTIMIZATION 4: Sparse Cross-Attention
# ============================================


class SparseQFormerBlock(nn.Module):
    """
    OPTIMIZATION: Sparse cross-attention for efficiency

    MOTIVATION: Not every query needs to attend to all 196 patches
    - Queries can focus on top-k most relevant patches
    - Reduces computation from O(32*196) to O(32*k)

    METHODS:
    1. Top-k selection based on similarity
    2. Local window attention
    3. Learned sparsity patterns
    """

    def __init__(self, dim: int = 768, num_heads: int = 12, top_k: int = 64):  # top-64 patches
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.top_k = top_k

        # Self-attention
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

        # Sparse cross-attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)

        # Similarity scoring
        self.similarity_proj = nn.Linear(dim, dim)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def select_top_k_patches(self, queries: torch.Tensor, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select top-k most relevant patches for each query

        Args:
            queries: [batch, num_queries, dim]
            patches: [batch, num_patches, dim]

        Returns:
            selected_patches: [batch, num_queries, top_k, dim]
            indices: [batch, num_queries, top_k]
        """
        batch_size, num_queries, dim = queries.shape
        num_patches = patches.shape[1]

        # similarity
        queries_proj = self.similarity_proj(queries)
        similarity = torch.bmm(queries_proj, patches.transpose(1, 2))

        # top-k patches
        top_k = min(self.top_k, num_patches)
        _, top_indices = torch.topk(similarity, k=top_k, dim=-1)
        patches_expanded = patches.unsqueeze(1).expand(batch_size, num_queries, num_patches, dim)
        top_indices_expanded = top_indices.unsqueeze(-1).expand(batch_size, num_queries, top_k, dim)
        selected_patches = torch.gather(patches_expanded, dim=2, index=top_indices_expanded)

        return selected_patches, top_indices

    def forward(self, queries: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: [batch, num_queries, dim]
            patches: [batch, num_patches, dim]

        Returns:
            queries: [batch, num_queries, dim]
        """
        # Self-attention
        queries_norm = self.norm1(queries)
        attn_out, _ = self.self_attn(query=queries_norm, key=queries_norm, value=queries_norm)
        queries = queries + attn_out

        # Sparse cross-attention
        selected_patches, _ = self.select_top_k_patches(queries, patches)
        batch_size, num_queries, top_k, dim = selected_patches.shape
        selected_patches_flat = selected_patches.reshape(batch_size * num_queries, top_k, dim)
        queries_flat = queries.reshape(batch_size * num_queries, 1, dim)
        queries_norm_flat = self.norm2(queries_flat)

        cross_out, _ = self.cross_attn(query=queries_norm_flat, key=selected_patches_flat, value=selected_patches_flat)

        cross_out = cross_out.reshape(batch_size, num_queries, dim)
        queries = queries + cross_out

        return queries


def demo_sparse_cross_attention():
    """Demonstrate sparse cross-attention efficiency"""
    print("\n" + "=" * 80)
    print("OPTIMIZATION 4: Sparse Cross-Attention")
    print("=" * 80)

    batch_size = 4
    num_queries = 32
    num_patches = 196
    dim = 768

    images = data_loader.get_cifar_batch(batch_size)
    with torch.no_grad():
        patches = vision_encoder(images)

    queries = torch.randn(batch_size, num_queries, dim).to(device)

    print("\nStandard Cross-Attention:")
    print(f"Queries: {num_queries}")
    print(f"Patches: {num_patches}")
    print(f"Attention operations: {num_queries * num_patches:,}")

    top_k_values = [32, 64, 96, 128]

    print("\nSparse Cross-Attention Benchmark:")
    print(f"{'Top-K':<10} {'Operations':<15} {'Time (ms)':<15} {'Reduction':<15} {'Speedup':<15}")
    print("-" * 70)

    baseline_ops = num_queries * num_patches
    num_runs = 50

    for top_k in top_k_values:
        sparse_block = SparseQFormerBlock(dim=dim, num_heads=12, top_k=top_k).to(device)

        # Warmup
        for i in range(10):
            with torch.no_grad():
                query_out = sparse_block(queries, patches)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark
        start = time.time()
        for i in range(num_runs):
            with torch.no_grad():
                query_out = sparse_block(queries, patches)  # noqa: F841

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        sparse_time = (time.time() - start) / num_runs * 1000

        ops = num_queries * top_k
        reduction = (1 - ops / baseline_ops) * 100
        speedup = baseline_ops / ops

        print(f"{top_k:<10} {ops:<15,} {sparse_time:<15.2f} {f'{reduction:.1f}%':<15} {f'{speedup:.2f}x':<15}")


# ============================================
# OPTIMIZATION 5: Mixed Precision Training
# ============================================


def demo_mixed_precision_training():
    """Demonstrate mixed precision training benefits"""
    print("\n" + "=" * 80)
    print("OPTIMIZATION 6: Mixed Precision Training")
    print("=" * 80)

    batch_size = 64
    num_patches = 196
    num_queries = 32
    dim = 768

    class SimpleQFormer(nn.Module):
        def __init__(self):
            super().__init__()
            self.cross_attn = nn.MultiheadAttention(dim, 12, batch_first=True)
            self.norm = nn.LayerNorm(dim)

        def forward(self, q, k):
            out, _ = self.cross_attn(q, k, k)
            return self.norm(out)

    model = SimpleQFormer().to(device)

    queries = torch.randn(batch_size, num_queries, dim).to(device)
    patches = torch.randn(batch_size, num_patches, dim).to(device)

    # FP32 Training
    print("\nFP32 Training:")
    torch.cuda.reset_peak_memory_stats()
    start = time.time()

    for i in range(100):
        output = model(queries, patches)
        loss = output.sum()
        loss.backward()

    fp32_time = time.time() - start
    fp32_memory = torch.cuda.max_memory_allocated() / 1e9

    print(f"Time: {fp32_time:.2f}s")
    print(f"Peak memory: {fp32_memory:.2f} GB")

    # FP16 Training with Automatic Mixed Precision
    print("\nFP16 Training:")
    model = SimpleQFormer().to(device)
    queries_fp16 = queries.half()
    patches_fp16 = patches.half()

    torch.cuda.reset_peak_memory_stats()
    start = time.time()

    scaler = torch.cuda.amp.GradScaler()

    for i in range(100):
        with torch.cuda.amp.autocast():
            output = model(queries_fp16, patches_fp16)
            loss = output.sum()

        scaler.scale(loss).backward()
        scaler.step(torch.optim.SGD(model.parameters(), lr=0.01))
        scaler.update()

    fp16_time = time.time() - start
    fp16_memory = torch.cuda.max_memory_allocated() / 1e9

    print(f"Time: {fp16_time:.2f}s")
    print(f"Peak memory: {fp16_memory:.2f} GB")

    # Comparison
    speedup = fp32_time / fp16_time
    memory_reduction = (1 - fp16_memory / fp32_memory) * 100

    print("\n" + "=" * 80)
    print(f"Speedup: {speedup:.2f}x")
    print(f"Memory reduction: {memory_reduction:.1f}%")
    print("=" * 80)


# ============================================
# OPTIMIZATION 6: Efficient Fine-tuning Strategies
# ============================================


class EfficientFineTuning:
    """
    OPTIMIZATION: Efficient fine-tuning strategies for BLIP-2

    Instead of fine-tuning all Q-Former parameters:
    1. Query-only fine-tuning: Only update query embeddings
    2. LoRA for Q-Former: Low-rank adaptation
    3. Adapter layers: Small bottleneck layers
    """

    @staticmethod
    def count_trainable_params(model: nn.Module) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def apply_lora(layer: nn.Linear, rank: int = 8):
        """
        Apply LoRA (Low-Rank Adaptation) to a linear layer

        Trainable params: rank * (d_in + d_out) << d_in * d_out
        """
        d_in = layer.in_features
        d_out = layer.out_features

        lora_A = nn.Parameter(torch.randn(d_in, rank) * 0.01)
        lora_B = nn.Parameter(torch.zeros(rank, d_out))

        layer.weight.requires_grad = False
        if layer.bias is not None:
            layer.bias.requires_grad = False

        return lora_A, lora_B


def demo_efficient_finetuning():
    """Demonstrate efficient fine-tuning strategies"""
    print("\n" + "=" * 80)
    print("OPTIMIZATION 8: Efficient Fine-tuning Strategies")
    print("=" * 80)

    # Q-Former parameters
    total_qformer_params = 188_000_000  # 188M
    query_params = 32 * 768

    strategies = []

    # Baseline
    strategies.append(
        {
            "Strategy": "Regular Fine-tuning",
            "Params": total_qformer_params,
            "Reduction": 0.0,
            "Memory (GB)": total_qformer_params * 4 / 1e9,
            "Use Case": "Maximum adaptation",
        }
    )

    # Query-only
    reduction1 = (1 - query_params / total_qformer_params) * 100
    strategies.append(
        {
            "Strategy": "Query-only",
            "Params": query_params,
            "Reduction": reduction1,
            "Memory (GB)": query_params * 4 / 1e9,
            "Use Case": "New domains",
        }
    )

    # LoRA (rank=8)
    lora_rank = 8
    num_linear_layers = 144
    lora_params_per_layer = 768 * lora_rank + lora_rank * 768
    lora_total_params = lora_params_per_layer * num_linear_layers
    reduction2 = (1 - lora_total_params / total_qformer_params) * 100
    strategies.append(
        {
            "Strategy": "LoRA (rank=8)",
            "Params": lora_total_params,
            "Reduction": reduction2,
            "Memory (GB)": lora_total_params * 4 / 1e9,
            "Use Case": "Multi-task",
        }
    )

    # Adapter Layers
    adapter_dim = 64
    adapters_per_layer = 2 * (768 * adapter_dim + adapter_dim * 768)
    adapter_total = adapters_per_layer * 12  # 12 layers
    reduction3 = (1 - adapter_total / total_qformer_params) * 100
    strategies.append(
        {
            "Strategy": "Adapters (dim=64)",
            "Params": adapter_total,
            "Reduction": reduction3,
            "Memory (GB)": adapter_total * 4 / 1e9,
            "Use Case": "Quick adaptation",
        }
    )

    print(f"{'Strategy':<20} {'Parameters':<18} {'Reduction':<15} {'Memory (GB)':<15} {'Use Case':<20}")
    print("-" * 95)

    for strategy in strategies:
        params_str = f"{strategy['Params']:,}"
        reduction_str = f"{strategy['Reduction']:.2f}%" if strategy["Reduction"] > 0 else "-"
        memory_str = f"{strategy['Memory (GB)']:.3f}"
        print(
            f"{strategy['Strategy']:<20} {params_str:<18} {reduction_str:<15} "
            f"{memory_str:<15} {strategy['Use Case']:<20}"
        )

    print("-" * 95)


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("BLIP-2 OPTIMIZATION EXPERIMENTS")
    print("=" * 80)

    # Optimization 1: Dynamic Query Allocation
    demo_dynamic_query_allocation()

    # Optimization 2: Progressive Query Training
    demo_progressive_query_training()

    # Optimization 3: Multi-Scale Visual Features
    demo_multiscale_qformer()

    # Optimization 4: Sparse Cross-Attention
    demo_sparse_cross_attention()

    # Optimization 5: Mixed Precision Training
    demo_mixed_precision_training()

    # Optimization 6: Efficient Fine-tuning
    demo_efficient_finetuning()
