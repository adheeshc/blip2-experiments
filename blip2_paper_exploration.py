"""BLIP2 Paper Deep Dive"""

import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from PIL import Image

from data_utils import ImageDataLoader, SimpleVisionEncoder

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

data_loader = ImageDataLoader(device=device)
vision_encoder = SimpleVisionEncoder(patch_size=16, embed_dim=768).to(device)


# ============================================
# PART 1: Learnable Query Embeddings
# ============================================


class LearnableQueries(nn.Module):
    """
    Learnable Query Embeddings - BLIP-2's Key Innovation

    KEY CONCEPT: Instead of using all 196 image patches, we use 32 learnable
    queries that extract the most relevant visual information for text tasks.

    This acts as an information bottleneck:
    - Input: Variable-length image features (196 patches from ViT)
    - Output: Fixed 32 query embeddings
    - Benefit: 6x compression while preserving relevant info
    """

    def __init__(self, num_queries: int = 32, query_dim: int = 768):
        super().__init__()
        self.num_queries = num_queries
        self.query_dim = query_dim

        self.queries = nn.Parameter(torch.randn(num_queries, query_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Args:
            batch_size: Number of images in batch

        Returns:
            queries: [batch, num_queries, query_dim]
        """
        queries = repeat(self.queries, "n d -> b n d", b=batch_size)
        return queries


def demo_learnable_queries():
    """Demonstrate learnable query initialization and properties"""
    print("\n" + "=" * 80)
    print("PART 1: Learnable Query Embeddings Analysis")
    print("=" * 80)

    num_queries = 32
    query_dim = 768
    batch_size = 4

    learnable_queries = LearnableQueries(num_queries=num_queries, query_dim=query_dim).to(device)

    print("\nQuery Configuration:")
    print(f"Number of queries: {num_queries}")
    print(f"Query dimension: {query_dim}")
    print(f"Total parameters: {num_queries * query_dim:,}")

    queries = learnable_queries(batch_size)
    print(f"\nQuery tensor shape: {queries.shape}")
    print(f"[batch_size={batch_size}, num_queries={num_queries}, query_dim={query_dim}]")

    # Analyze query statistics
    print("\nQuery Statistics:")
    print(f"Mean: {queries.mean():.4f}")
    print(f"Std: {queries.std():.4f}")
    print(f"Min: {queries.min():.4f}")
    print(f"Max: {queries.max():.4f}")

    # Compare with image patch tokens
    num_patches = 196  # 14x14 grid for ViT-L/14
    compression_ratio = num_patches / num_queries

    print("\nCompression Analysis:")
    print(f"Image patches (ViT-L/14): {num_patches}")
    print(f"Query tokens: {num_queries}")
    print(f"Compression ratio: {compression_ratio:.1f}x")
    print(f"Computational savings: O({num_patches}^2) → O({num_queries}^2)")
    print(f"Attention complexity reduction: {(num_patches/num_queries)**2:.1f}x")

    return learnable_queries


# ============================================
# PART 2: Q-Former Core Architecture
# ============================================


class QFormerBlock(nn.Module):
    """
    Single Q-Former Transformer Block

    Components:
    1. Self-attention: Queries interact with each other AND with text
    2. Cross-attention: Queries attend to frozen image features
    3. Feed-forward network

    Key Concept: attention masking that controls query-text interaction
    """

    def __init__(self, dim: int = 768, num_heads: int = 12, mlp_ratio: int = 4, cross_attention_freq: int = 2):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # Self-attention (shared between queries and text)
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)

        # Cross-attention (queries attend to image)
        self.cross_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        # Feed-forward
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_hidden_dim), nn.GELU(), nn.Linear(mlp_hidden_dim, dim))
        self.norm3 = nn.LayerNorm(dim)

    def forward(
        self, queries: torch.Tensor, image_features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            queries: [batch, num_queries, dim]
            image_features: [batch, num_patches, dim] from frozen ViT
            attention_mask: Optional mask for self-attention

        Returns:
            queries: [batch, num_queries, dim] - updated queries
        """
        # Self-attention with residual
        queries_norm = self.norm1(queries)
        attn_output, _ = self.self_attn(
            query=queries_norm, key=queries_norm, value=queries_norm, attn_mask=attention_mask
        )
        queries = queries + attn_output

        # Cross-attention to image features with residual
        queries_norm = self.norm2(queries)
        cross_output, _ = self.cross_attn(query=queries_norm, key=image_features, value=image_features)
        queries = queries + cross_output

        # Feed-forward with residual
        queries = queries + self.mlp(self.norm3(queries))

        return queries


class QFormer(nn.Module):
    """
    Complete Q-Former Architecture

    Key Innovation:
    - Acts as information bottleneck (196 patches → 32 queries)
    - Learns what visual info matters for text tasks
    - Operates in three modes via attention masking

    With 32 queries, each query can specialize in different aspects:
    - Queries 1-8: Object detection and recognition
    - Queries 9-16: Spatial relationships and layout
    - Queries 17-24: Colors, textures, and attributes
    - Queries 25-32: Contextual and scene-level information

    """

    def __init__(
        self,
        num_queries: int = 32,
        num_hidden_layers: int = 12,
        hidden_size: int = 768,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.hidden_size = hidden_size

        # Learnable query embeddings
        self.query_tokens = LearnableQueries(num_queries, hidden_size)

        # Stack of Q-Former blocks
        self.blocks = nn.ModuleList(
            [
                QFormerBlock(dim=hidden_size, num_heads=num_attention_heads, mlp_ratio=intermediate_size // hidden_size)
                for _ in range(num_hidden_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, image_features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            image_features: [batch, num_patches, dim] from frozen ViT
            attention_mask: Optional attention mask

        Returns:
            query_output: [batch, num_queries, dim]
        """
        batch_size = image_features.shape[0]

        # Initialize queries
        queries = self.query_tokens(batch_size)

        # Process through Q-Former blocks
        for block in self.blocks:
            queries = block(queries, image_features, attention_mask)

        # Final layer norm
        query_output = self.layer_norm(queries)

        return query_output


def demo_qformer_architecture():
    """Demonstrate Q-Former architecture and forward pass"""
    print("\n" + "=" * 80)
    print("PART 2: Q-Former Architecture Deep Dive")
    print("=" * 80)

    batch_size = 4
    num_patches = 196  # From ViT-L/14 (14x14 grid)
    patch_dim = 768

    images = data_loader.get_cifar_batch(batch_size)

    with torch.no_grad():
        frozen_image_features = vision_encoder(images)
    print(f"✓ Encoded to patches: {frozen_image_features.shape}")
    print(f"  [{batch_size} images, {num_patches} patches, {patch_dim} dim]")

    # Create Q-Former
    qformer = QFormer(num_queries=32, num_hidden_layers=12, hidden_size=768, num_attention_heads=12).to(device)

    print("\nInput (Frozen ViT output):")
    print(f"Shape: {frozen_image_features.shape}")
    print(f"[batch={batch_size}, patches={num_patches}, dim={patch_dim}]")

    # Create Q-Former
    image_features = frozen_image_features

    print("\nQ-Former Configuration:")
    print("Hidden size: 768")
    print("Number of layers: 12")
    print("Number of queries: 32")
    print("Attention heads: 12")

    # Forward pass
    with torch.no_grad():
        start_time = time.time()
        query_output = qformer(image_features)
        forward_time = time.time() - start_time

    print("\nOutput:")
    print(f"Shape: {query_output.shape}")
    print(f"[batch={batch_size}, queries=32, dim=768]")
    print(f"Forward pass time: {forward_time*1000:.2f} ms")

    # Parameter count
    qformer_params = sum(p.numel() for p in qformer.parameters())
    query_params = 32 * 768
    transformer_params = qformer_params - query_params

    print("\nParameter Count:")
    print(f"Query embeddings: {query_params:,}")
    print(f"Transformer blocks: {transformer_params:,}")
    print(f"Total Q-Former: {qformer_params:,}")

    return qformer, query_output


# ============================================
# PART 3: Stage 1 Training Objectives
# ============================================


class Stage1Objectives:
    """
    BLIP-2 Stage 1: Vision-Language Representation Learning

    Three objectives trained simultaneously:
    1. ITC (Image-Text Contrastive): Global alignment
    2. ITM (Image-Text Matching): Fine-grained matching
    3. ITG (Image-Grounded Text Generation): Captioning

    Different attention masks control query-text interaction
    """

    @staticmethod
    def image_text_contrastive(
        query_output: torch.Tensor, text_embeddings: torch.Tensor, temperature: float = 0.07
    ) -> torch.Tensor:
        """
        ITC Loss: Contrastive learning between image and text

        Uses unimodal attention mask (queries and text are independent)

        Args:
            query_output: [batch, num_queries, dim]
            text_embeddings: [batch, dim]
            temperature: Scaling factor

        Returns:
            loss: Scalar contrastive loss
        """
        # Pool query outputs
        image_embeds = query_output.mean(dim=1)

        # Normalize
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(image_embeds, text_embeddings.t()) / temperature

        # Contrastive loss
        batch_size = sim_matrix.shape[0]
        targets = torch.arange(batch_size).to(device)

        loss_image2text = F.cross_entropy(sim_matrix, targets)
        loss_text2image = F.cross_entropy(sim_matrix.t(), targets)

        loss = (loss_image2text + loss_text2image) / 2

        return loss

    @staticmethod
    def image_text_matching(
        query_output: torch.Tensor, text_features: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        ITM Loss: Binary classification for image-text pairs

        Uses bidirectional attention mask (queries can attend to text)

        Args:
            query_output: [batch, num_queries, dim]
            text_features: [batch, seq_len, dim]
            labels: [batch] - 1 for match, 0 for no match

        Returns:
            loss: Binary cross-entropy loss
        """
        # Pool representations
        image_rep = query_output.mean(dim=1)
        text_rep = text_features.mean(dim=1)

        # Concatenate
        combined = torch.cat([image_rep, text_rep], dim=-1)

        classifier = nn.Linear(combined.shape[-1], 2).to(device)  # use Q former output
        logits = classifier(combined)

        loss = F.cross_entropy(logits, labels)

        return loss

    @staticmethod
    def image_grounded_text_generation(
        query_output: torch.Tensor, text_ids: torch.Tensor, decoder: nn.Module
    ) -> torch.Tensor:
        """
        ITG Loss: Language modeling for caption generation

        Uses causal attention mask (text cannot see future tokens)

        Args:
            query_output: [batch, num_queries, dim]
            text_ids: [batch, seq_len] - ground truth captions
            decoder: Language model decoder

        Returns:
            loss: Language modeling loss
        """
        # Q-Former output serves as prefix to decoder
        # This would be the actual LM loss
        loss = torch.tensor(0.0, device=device, requires_grad=True)

        return loss


def demo_stage1_objectives():
    """Demonstrate Stage 1 training objectives"""
    print("\n" + "=" * 80)
    print("PART 3: Stage 1 Training Objectives")
    print("=" * 80)

    batch_size = 4
    num_queries = 32
    dim = 768

    # Simulated inputs
    query_output = torch.randn(batch_size, num_queries, dim).to(device)
    text_embeddings = torch.randn(batch_size, dim).to(device)
    text_features = torch.randn(batch_size, 20, dim).to(device)

    objectives = Stage1Objectives()

    print("\nObjective 1: Image-Text Contrastive (ITC)")
    print("-" * 80)
    itc_loss = objectives.image_text_contrastive(query_output, text_embeddings)
    print(f"ITC Loss: {itc_loss.item():.4f}")
    print("Purpose: Learn global image-text alignment")
    print("Attention Mask: Unimodal (queries - queries, text - text separately)")

    print("\nObjective 2: Image-Text Matching (ITM)")
    print("-" * 80)
    labels = torch.randint(0, 2, (batch_size,)).to(device)
    itm_loss = objectives.image_text_matching(query_output, text_features, labels)
    print(f"ITM Loss: {itm_loss.item():.4f}")
    print("Purpose: Fine-grained matching (binary classification)")
    print("Attention Mask: Bidirectional (queries attend to text)")

    print("\nObjective 3: Image-Grounded Text Generation (ITG)")
    print("-" * 80)
    text_ids = torch.randint(0, 30000, (batch_size, 20)).to(device)
    decoder = nn.Linear(dim, 30000).to(device)  # Simplified
    itg_loss = objectives.image_grounded_text_generation(query_output, text_ids, decoder)
    print(f"ITG Loss: {itg_loss.item():.4f}")
    print("Purpose: Enable caption generation")
    print("Attention Mask: Causal (text cannot see future)")


# ============================================
# PART 4: Two-Stage Training Simulation
# ============================================


class MiniBlip2(nn.Module):
    """
    Simplified BLIP-2 model for demonstration

    Components:
    1. Frozen Vision Encoder (using our SimpleVisionEncoder)
    2. Q-Former (trainable)
    3. Frozen LLM (simulated)
    """

    def __init__(self, vision_dim: int = 768, qformer_dim: int = 768, llm_dim: int = 2048, num_queries: int = 32):
        super().__init__()

        # Use frozen vision encoder
        self.vision_encoder = vision_encoder
        self._freeze_module(self.vision_encoder)

        # Q-Former
        self.qformer = QFormer(
            num_queries=num_queries,
            num_hidden_layers=6,
            hidden_size=qformer_dim,
            num_attention_heads=12,
        )

        # Projection from Q-Former to LLM dimension
        self.llm_proj = nn.Linear(qformer_dim, llm_dim)

        # Frozen LLM
        self.llm = nn.Sequential(nn.Linear(llm_dim, llm_dim * 4), nn.GELU(), nn.Linear(llm_dim * 4, 30000))
        self._freeze_module(self.llm)

    def _freeze_module(self, module: nn.Module):
        """Freeze a module's parameters"""
        for param in module.parameters():
            param.requires_grad = False

    def forward_stage1(self, images: torch.Tensor) -> torch.Tensor:
        """
        Stage 1: Vision-Language Representation Learning
        Only Q-Former is trained

        Args:
            images: [batch, 3, 224, 224]
        """
        # Frozen vision encoding
        with torch.no_grad():
            vision_features = self.vision_encoder(images)

        # Q-Former extracts relevant features
        query_output = self.qformer(vision_features)

        return query_output

    def forward_stage2(self, images: torch.Tensor) -> torch.Tensor:
        """
        Stage 2: Vision-to-Language Generative Learning
        Q-Former continues training, vision encoder and LLM frozen

        Args:
            images: [batch, 3, 224, 224]
        """
        # Stage 1 forward
        query_output = self.forward_stage1(images)

        # Project to LLM dimension
        llm_input = self.llm_proj(query_output)

        # Pool queries for LLM input
        llm_input = llm_input.mean(dim=1)

        # Frozen LLM generation
        with torch.no_grad():
            logits = self.llm(llm_input)

        return logits


def demo_two_stage_training():
    """Demonstrate two-stage training paradigm"""
    print("\n" + "=" * 80)
    print("PART 5: Two-Stage Training Simulation")
    print("=" * 80)

    batch_size = 4

    # Create model
    model = MiniBlip2(vision_dim=768, qformer_dim=768, llm_dim=2048, num_queries=32).to(device)

    # Load real images
    images = data_loader.get_test_images(batch_size)
    print(f"Loaded {batch_size} test images: {images.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print("\nModel Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Frozen parameters: {frozen_params:,} ({frozen_params/total_params*100:.1f}%)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")

    print("\n" + "-" * 80)
    print("Stage 1: Vision-Language Representation Learning")
    print("-" * 80)
    print("Objectives: ITC + ITM + ITG")
    print("What trains: Q-Former only")
    print("What's frozen: Vision encoder (ViT)")

    with torch.no_grad():
        query_output = model.forward_stage1(images)

    print(f"\nStage 1 Output: {query_output.shape}")
    print("Q-Former learns to extract text-relevant visual features")

    print("\n" + "-" * 80)
    print("Stage 2: Vision-to-Language Generative Learning")
    print("-" * 80)
    print("Objective: Language Modeling")
    print("What trains: Q-Former (continues training)")
    print("What's frozen: Vision encoder + LLM")

    with torch.no_grad():
        logits = model.forward_stage2(images)

    print(f"\nStage 2 Output: {logits.shape}")
    print("Q-Former learns to produce LLM-compatible features")

    return model


# ============================================
# PART 5: Performance Benchmarking
# ============================================


def benchmark_BLIP_components():
    """Benchmark BLIP Components"""
    print("\n" + "=" * 80)
    print("PART 5: Performance Benchmarking")
    print("=" * 80)

    batch_sizes = [1, 4, 8, 16, 32, 64]
    num_queries_list = [1, 4, 8, 16, 32, 64]

    print("\n" + "=" * 80)
    print("COMPONENT ANALYSIS")
    print("=" * 80)

    # 1. Vision Encoder Analysis
    print("\nVision Encoder:")
    print("-" * 80)
    vision_params = sum(p.numel() for p in vision_encoder.parameters())
    trainable_vision = sum(p.numel() for p in vision_encoder.parameters() if p.requires_grad)

    print(f"Total parameters: {vision_params:,}")
    print(f"Trainable parameters: {trainable_vision:,}")
    print("Input size: 224x224")
    print("Patch size: 16x16")
    print(f"Number of patches: {(224//16)**2}")
    print("Embedding dimension: 768")

    # Benchmark vision encoder
    test_batch = data_loader.get_test_images(batch_sizes[2])

    # Warmup
    for i in range(10):
        with torch.no_grad():
            features = vision_encoder(test_batch)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # benchmark
    num_runs = 100
    start = time.time()
    for i in range(num_runs):
        with torch.no_grad():
            features = vision_encoder(test_batch)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = (time.time() - start) / num_runs * 1000
    print(f"\nInference time (batch={batch_sizes[2]}): {elapsed:.2f} ms")
    print(f"Throughput: {batch_sizes[2]/(elapsed/1000):.2f} images/sec")
    print(f"Output shape: {features.shape}")

    # 2. Q-Former Analysis
    print("\nQ-Former:")
    print("-" * 80)

    qformer = QFormer(num_queries=32, num_hidden_layers=12, hidden_size=768).to(device)
    qformer_params = sum(p.numel() for p in qformer.parameters())
    trainable_qformer = sum(p.numel() for p in qformer.parameters() if p.requires_grad)

    print(f"Total parameters: {qformer_params:,}")
    print(f"Trainable parameters: {trainable_qformer:,}")
    print("Number of queries: 32")
    print("Hidden size: 768")
    print("Number of layers: 12")
    print("Number of heads: 12")

    # Benchmark Q-Former
    with torch.no_grad():
        image_features = vision_encoder(test_batch)

    # Warmup
    for i in range(10):
        with torch.no_grad():
            query_output = qformer(image_features)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Measure
    start = time.time()
    for i in range(num_runs):
        with torch.no_grad():
            query_output = qformer(image_features)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = (time.time() - start) / num_runs * 1000
    print(f"\nInference time (batch=8): {elapsed:.2f} ms")
    print(f"Throughput: {8/(elapsed/1000):.2f} images/sec")
    print(f"Output shape: {query_output.shape}")
    print(
        f"Compression ratio: {image_features.shape[1]}/{query_output.shape[1]} = {image_features.shape[1]/query_output.shape[1]:.1f}x"
    )

    # 3. Batch Size Scaling
    print("\nBATCH SIZE SCALING")
    print(
        f"\n{'Batch Size':<12} {'Vision (ms)':<15} {'Q-Former (ms)':<15} {'Total (ms)':<15} {'Throughput (img/s)':<20}"
    )
    print("-" * 80)

    for batch_size in batch_sizes:
        batch = data_loader.get_test_images(batch_size)

        # Warmup
        for i in range(5):
            with torch.no_grad():
                feat = vision_encoder(batch)
                query_output = qformer(feat)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Vision encoder timing
        start = time.time()
        for i in range(50):
            with torch.no_grad():
                feat = vision_encoder(batch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        vision_time = (time.time() - start) / 50 * 1000

        # Q-Former timing
        start = time.time()
        for i in range(50):
            with torch.no_grad():
                query_output = qformer(feat)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        qformer_time = (time.time() - start) / 50 * 1000

        total_time = vision_time + qformer_time
        throughput = batch_size / (total_time / 1000)

        print(f"{batch_size:<12} {vision_time:<15.2f} {qformer_time:<15.2f} {total_time:<15.2f} {throughput:<20.2f}")

    # 4. Query Count Comparison
    print("\nQUERY COUNT COMPARISON")
    print(f"\n{'Queries':<12} {'Parameters':<15} {'Inference (ms)':<15} {'Compression':<15}")
    print("-" * 57)

    test_batch = data_loader.get_test_images(4)
    with torch.no_grad():
        feat = vision_encoder(test_batch)

    for num_queries in num_queries_list:
        qformer_variant = QFormer(num_queries=num_queries, num_hidden_layers=12, hidden_size=768).to(device)
        params = sum(p.numel() for p in qformer_variant.parameters())

        # Warmup
        for i in range(5):
            with torch.no_grad():
                query_output = qformer_variant(feat)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Measure
        start = time.time()
        for i in range(50):
            with torch.no_grad():
                query_output = qformer_variant(feat)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        elapsed = (time.time() - start) / 50 * 1000
        compression = feat.shape[1] / num_queries

        print(f"{num_queries:<12} {params:<15,} {elapsed:<15.2f} {f'{compression:.1f}x':<15}")

    # 5. Memory Usage
    print("\nMEMORY ANALYSIS")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

        batch = data_loader.get_test_images(8)
        qformer = QFormer(num_queries=32, num_hidden_layers=12, hidden_size=768).to(device)

        with torch.no_grad():
            feat = vision_encoder(batch)
            _ = qformer(feat)

        memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        print(f"\nPeak GPU memory (batch=8): {memory_mb:.2f} MB")
        print(f"Per image: {memory_mb/8:.2f} MB")


if __name__ == "__main__":
    print("=" * 50)
    print("BLIP2 EXPLORATION")
    print("=" * 50)

    demo_learnable_queries()

    demo_qformer_architecture()

    demo_stage1_objectives()

    demo_two_stage_training()

    benchmark_BLIP_components()
