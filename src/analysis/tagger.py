"""Auto-tag papers by keyword rules applied to title and abstract."""

from src.storage.paper_store import Paper

# Mapping of tag -> list of keyword patterns (case-insensitive)
TAG_RULES: dict[str, list[str]] = {
    "side-channel": [
        "side channel", "side-channel", "covert channel",
        "information leakage", "timing attack",
    ],
    "perf-counters": [
        "performance counter", "hardware counter", "PMU",
        "CUPTI", "perf counter", "hardware performance monitor",
    ],
    "ml-training": [
        "machine learning training", "deep learning training",
        "neural network training", "model training",
        "training workload", "training run",
    ],
    "governance": [
        "compute governance", "AI governance", "chip governance",
        "export control", "compute monitoring",
        "hardware governance", "AI safety",
    ],
    "adversarial": [
        "adversarial", "evasion", "obfuscation",
        "mimicry", "disguise", "anti-detection",
    ],
    "gpu-profiling": [
        "GPU profiling", "GPU monitoring", "GPU characteriz",
        "GPU workload", "GPU kernel", "GPU benchmark",
    ],
    "nvidia": [
        "NVIDIA", "CUDA", "CUPTI", "cuDNN", "TensorRT",
        "A100", "H100", "V100",
    ],
    "power-analysis": [
        "power consumption", "power analysis", "energy consumption",
        "power trace", "thermal",
    ],
}


def auto_tag(paper: Paper, rules: dict[str, list[str]] = TAG_RULES) -> list[str]:
    """Determine tags for a paper based on keyword rules.

    Args:
        paper: The paper to tag.
        rules: Dict mapping tag name to list of keyword patterns.

    Returns:
        List of matching tags.
    """
    text = f"{paper.title} {paper.abstract}".lower()
    tags = []

    for tag, patterns in rules.items():
        for pattern in patterns:
            if pattern.lower() in text:
                tags.append(tag)
                break  # One match per tag is enough

    return tags


def apply_tags(paper: Paper, rules: dict[str, list[str]] = TAG_RULES) -> Paper:
    """Apply auto-tags to a paper, merging with existing tags.

    Args:
        paper: The paper to tag.
        rules: Tag rules to use.

    Returns:
        The paper with updated tags.
    """
    new_tags = auto_tag(paper, rules)
    existing = set(paper.tags)
    for tag in new_tags:
        existing.add(tag)
    paper.tags = sorted(existing)
    return paper


def tag_all(papers: list[Paper], rules: dict[str, list[str]] = TAG_RULES) -> list[Paper]:
    """Apply auto-tags to a list of papers.

    Args:
        papers: Papers to tag.
        rules: Tag rules to use.

    Returns:
        The papers with updated tags.
    """
    for paper in papers:
        apply_tags(paper, rules)
    return papers
