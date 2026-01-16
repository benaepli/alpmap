#!/usr/bin/env python3
"""
Benchmark Graph Generator for Hash Set Comparisons

Generates comparison graphs for:
1. StoreHash_LF875_Linear vs Abseil vs std::unordered_set (at larger sizes)
2. StoreHash vs NoStoreHash for LF875
3. Load factor comparisons (LF85, LF875, LF90)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re

# Set style for README-friendly graphs
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.dpi'] = 150


def parse_benchmark_name(name):
    """Parse benchmark name into components."""
    # Remove quotes
    name = name.strip('"')

    # Split by '/'
    parts = name.split('/')
    if len(parts) != 3:
        return None

    impl_name = parts[0]
    operation = parts[1]
    size = int(parts[2])

    return {
        'impl': impl_name,
        'operation': operation,
        'size': size,
        'full_name': name
    }


def load_data(csv_path):
    """Load and parse benchmark data."""
    df = pd.read_csv(csv_path)

    # Parse the name column
    parsed = df['name'].apply(parse_benchmark_name)
    parsed_df = pd.DataFrame(parsed.tolist())

    # Combine with original data
    result = pd.concat([df, parsed_df], axis=1)
    result = result.dropna(subset=['impl'])

    # Calculate ns per operation
    result['ns_per_op'] = result['cpu_time']

    return result


def get_impl_category(impl):
    """Categorize implementation for filtering."""
    if impl.startswith('Std_UnorderedSet'):
        return 'std', 'String' if 'String' in impl else 'Int64'
    elif impl.startswith('Absl_FlatHashSet'):
        return 'abseil', 'String' if 'String' in impl else 'Int64'
    elif impl.startswith('Alp_'):
        dtype = 'String' if 'String' in impl else 'Int64'
        return 'alp', dtype
    return None, None


def create_comparison_vs_baseline(df, output_dir):
    """
    Create comparison: alpmap (best variant) vs Abseil vs std::unordered_set
    Shows relative performance normalized to alpmap (1.0 = same as alpmap).
    """
    operations = ['Insert', 'LookupHit', 'LookupMiss', 'Erase', 'Iterate']
    data_types = ['Int64', 'String']

    # Use all sizes for better visualization
    all_sizes = sorted(df['size'].unique())

    for dtype in data_types:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        # Define implementations to compare
        if dtype == 'Int64':
            alp_impl = f'Alp_{dtype}_Rapid_Linear'
            alp_label = 'alpmap (Rapid)'
        else:
            alp_impl = f'Alp_{dtype}_Rapid_StoreHash_LF875_Linear'
            alp_label = 'alpmap (LF87.5%)'
        abseil_impl = f'Absl_FlatHashSet_{dtype}'
        std_impl = f'Std_UnorderedSet_{dtype}'

        for idx, op in enumerate(operations):
            ax = axes[idx]

            # Get alpmap baseline data
            alp_data = df[(df['impl'] == alp_impl) & (df['operation'] == op)].set_index('size')['ns_per_op']

            for impl, label, color, marker in [
                (alp_impl, alp_label, '#2ecc71', 'o'),
                (abseil_impl, 'Abseil FlatHashSet', '#3498db', 's'),
                (std_impl, 'std::unordered_set', '#e74c3c', '^'),
            ]:
                subset = df[(df['impl'] == impl) & (df['operation'] == op)]
                if not subset.empty:
                    subset = subset.sort_values('size')
                    # Normalize to alpmap baseline (higher = slower than alpmap)
                    relative_perf = subset.set_index('size')['ns_per_op'] / alp_data
                    relative_perf = relative_perf.dropna()
                    ax.plot(relative_perf.index, relative_perf.values,
                            marker=marker, label=label, color=color,
                            linewidth=2, markersize=6)

            ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
            ax.set_xscale('log', base=2)
            ax.set_xlabel('Number of Elements')
            ax.set_ylabel('Relative Time (1.0 = alpmap)')
            ax.set_title(f'{op}')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

            # Set y-axis to show meaningful range
            ax.set_ylim(bottom=0)

        axes[5].axis('off')

        plt.suptitle(
            f'{dtype}: Performance Relative to alpmap (lower is better)\n(alpmap: Rapid hash, StoreHash, LF=87.5%, Linear probing)',
            fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'comparison_{dtype.lower()}_vs_baseline.png'),
                    bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Created: comparison_{dtype.lower()}_vs_baseline.png")


def create_storehash_comparison(df, output_dir):
    """
    Create comparison: StoreHash vs NoStoreHash for LF875.
    Shows relative performance normalized to StoreHash Linear.
    """
    operations = ['Insert', 'LookupHit', 'LookupMiss', 'Erase', 'Iterate']
    data_types = ['Int64', 'String']

    for dtype in data_types:
        # Only String has NoStoreHash variants based on the data
        if dtype == 'Int64':
            continue

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        storehash_linear = f'Alp_{dtype}_Rapid_StoreHash_LF875_Linear'
        nostorehash_linear = f'Alp_{dtype}_Rapid_NoStoreHash_LF875_Linear'
        storehash_quad = f'Alp_{dtype}_Rapid_StoreHash_LF875_Quadratic'
        nostorehash_quad = f'Alp_{dtype}_Rapid_NoStoreHash_LF875_Quadratic'

        for idx, op in enumerate(operations):
            ax = axes[idx]

            # Get StoreHash Linear as baseline
            baseline_data = df[(df['impl'] == storehash_linear) & (df['operation'] == op)].set_index('size')[
                'ns_per_op']

            for impl, label, color, marker in [
                (storehash_linear, 'StoreHash Linear', '#2ecc71', 'o'),
                (nostorehash_linear, 'NoStoreHash Linear', '#e67e22', 's'),
                (storehash_quad, 'StoreHash Quadratic', '#3498db', '^'),
                (nostorehash_quad, 'NoStoreHash Quadratic', '#9b59b6', 'd'),
            ]:
                subset = df[(df['impl'] == impl) & (df['operation'] == op)]
                if not subset.empty:
                    subset = subset.sort_values('size')
                    relative_perf = subset.set_index('size')['ns_per_op'] / baseline_data
                    relative_perf = relative_perf.dropna()
                    ax.plot(relative_perf.index, relative_perf.values,
                            marker=marker, label=label, color=color,
                            linewidth=2, markersize=5)

            ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
            ax.set_xscale('log', base=2)
            ax.set_xlabel('Number of Elements')
            ax.set_ylabel('Relative Time (1.0 = StoreHash Linear)')
            ax.set_title(f'{op}')
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)

        axes[5].axis('off')

        plt.suptitle(f'{dtype}: StoreHash vs NoStoreHash (LF=87.5%)\n(lower is better)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'storehash_comparison_{dtype.lower()}.png'),
                    bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Created: storehash_comparison_{dtype.lower()}.png")


def create_load_factor_comparison(df, output_dir):
    """
    Create load factor comparison (LF85, LF875, LF90) for StoreHash Linear.
    Shows relative performance normalized to LF87.5%.
    Only for String type as Int64 doesn't have LF variants.
    """
    operations = ['Insert', 'LookupHit', 'LookupMiss', 'Erase', 'Iterate']

    # Only String has load factor variants
    dtype = 'String'

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Define load factor implementations (StoreHash, Linear probing)
    baseline_impl = f'Alp_{dtype}_Rapid_StoreHash_LF875_Linear'
    lf_impls = [
        (f'Alp_{dtype}_Rapid_StoreHash_LF85_Linear', 'LF 85%', '#e74c3c'),
        (f'Alp_{dtype}_Rapid_StoreHash_LF875_Linear', 'LF 87.5%', '#2ecc71'),
        (f'Alp_{dtype}_Rapid_StoreHash_LF90_Linear', 'LF 90%', '#3498db'),
    ]

    for idx, op in enumerate(operations):
        ax = axes[idx]

        # Get LF87.5% as baseline
        baseline_data = df[(df['impl'] == baseline_impl) & (df['operation'] == op)].set_index('size')['ns_per_op']

        for impl, label, color in lf_impls:
            subset = df[(df['impl'] == impl) & (df['operation'] == op)]
            if not subset.empty:
                subset = subset.sort_values('size')
                relative_perf = subset.set_index('size')['ns_per_op'] / baseline_data
                relative_perf = relative_perf.dropna()
                ax.plot(relative_perf.index, relative_perf.values,
                        marker='o', label=label, color=color,
                        linewidth=2, markersize=5)

        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_xscale('log', base=2)
        ax.set_xlabel('Number of Elements')
        ax.set_ylabel('Relative Time (1.0 = LF 87.5%)')
        ax.set_title(f'{op}')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)

    axes[5].axis('off')

    plt.suptitle(f'{dtype}: Load Factor Comparison (StoreHash, Linear probing)\n(lower is better)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'load_factor_comparison_{dtype.lower()}.png'),
                bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Created: load_factor_comparison_{dtype.lower()}.png")


def create_throughput_bar_chart(df, output_dir):
    """
    Create bar chart showing throughput (items/sec) at a specific large size.
    """
    target_size = 2097152  # Good representative large size
    operations = ['Insert', 'LookupHit', 'LookupMiss', 'Erase']

    for dtype in ['Int64', 'String']:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        # Int64 uses base Rapid_Linear, String uses StoreHash_LF875_Linear
        if dtype == 'Int64':
            alp_impl = f'Alp_{dtype}_Rapid_Linear'
            alp_label = 'alpmap\n(Rapid)'
        else:
            alp_impl = f'Alp_{dtype}_Rapid_StoreHash_LF875_Linear'
            alp_label = 'alpmap\n(LF87.5%)'

        impls_to_compare = [
            (alp_impl, alp_label, '#2ecc71'),
            (f'Absl_FlatHashSet_{dtype}', 'Abseil', '#3498db'),
            (f'Std_UnorderedSet_{dtype}', 'std::unordered_set', '#e74c3c'),
        ]

        for idx, op in enumerate(operations):
            ax = axes[idx]

            values = []
            labels = []
            colors = []

            for impl, label, color in impls_to_compare:
                subset = df[(df['impl'] == impl) &
                            (df['operation'] == op) &
                            (df['size'] == target_size)]
                if not subset.empty:
                    throughput = subset['items_per_second'].values[0]
                    values.append(throughput / 1e6)  # Convert to millions
                    labels.append(label)
                    colors.append(color)

            if values:
                bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=0.5)
                ax.set_ylabel('Throughput (M items/sec)')
                ax.set_title(f'{op} @ {target_size:,} elements')

                # Add value labels on bars
                for bar, val in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                            f'{val:.1f}M', ha='center', va='bottom', fontsize=9)

        plt.suptitle(f'{dtype}: Throughput Comparison at {target_size:,} Elements',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'throughput_bar_{dtype.lower()}.png'),
                    bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Created: throughput_bar_{dtype.lower()}.png")


def create_linear_vs_quadratic(df, output_dir):
    """
    Create comparison of Linear vs Quadratic probing.
    Shows relative performance normalized to Linear probing.
    """
    operations = ['Insert', 'LookupHit', 'LookupMiss', 'Erase', 'Iterate']

    for dtype in ['Int64', 'String']:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        # Int64 uses base Rapid variants, String uses StoreHash LF875
        if dtype == 'Int64':
            linear_impl = f'Alp_{dtype}_Rapid_Linear'
            quad_impl = f'Alp_{dtype}_Rapid_Quadratic'
            title_suffix = '(Rapid hash)'
        else:
            linear_impl = f'Alp_{dtype}_Rapid_StoreHash_LF875_Linear'
            quad_impl = f'Alp_{dtype}_Rapid_StoreHash_LF875_Quadratic'
            title_suffix = '(StoreHash, LF=87.5%)'

        for idx, op in enumerate(operations):
            ax = axes[idx]

            # Get Linear as baseline
            baseline_data = df[(df['impl'] == linear_impl) & (df['operation'] == op)].set_index('size')['ns_per_op']

            for impl, label, color, marker in [
                (linear_impl, 'Linear Probing', '#2ecc71', 'o'),
                (quad_impl, 'Quadratic Probing', '#9b59b6', 's'),
            ]:
                subset = df[(df['impl'] == impl) & (df['operation'] == op)]
                if not subset.empty:
                    subset = subset.sort_values('size')
                    relative_perf = subset.set_index('size')['ns_per_op'] / baseline_data
                    relative_perf = relative_perf.dropna()
                    ax.plot(relative_perf.index, relative_perf.values,
                            marker=marker, label=label, color=color,
                            linewidth=2, markersize=6)

            ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
            ax.set_xscale('log', base=2)
            ax.set_xlabel('Number of Elements')
            ax.set_ylabel('Relative Time (1.0 = Linear)')
            ax.set_title(f'{op}')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)

        axes[5].axis('off')

        plt.suptitle(f'{dtype}: Linear vs Quadratic Probing {title_suffix}\n(lower is better)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'probing_comparison_{dtype.lower()}.png'),
                    bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Created: probing_comparison_{dtype.lower()}.png")


def create_speedup_chart(df, output_dir):
    """
    Create speedup chart showing alpmap's speedup over baselines at various sizes.
    """
    operations = ['Insert', 'LookupHit', 'LookupMiss', 'Erase']
    large_sizes = [4096, 32768, 262144, 2097152]

    for dtype in ['Int64', 'String']:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        # Int64 uses base Rapid_Linear, String uses StoreHash_LF875_Linear
        if dtype == 'Int64':
            alp_impl = f'Alp_{dtype}_Rapid_Linear'
            title_suffix = '(Rapid hash)'
        else:
            alp_impl = f'Alp_{dtype}_Rapid_StoreHash_LF875_Linear'
            title_suffix = '(StoreHash, LF=87.5%)'
        abseil_impl = f'Absl_FlatHashSet_{dtype}'
        std_impl = f'Std_UnorderedSet_{dtype}'

        for idx, op in enumerate(operations):
            ax = axes[idx]

            speedup_abseil = []
            speedup_std = []
            sizes_found = []

            for size in large_sizes:
                alp_data = df[(df['impl'] == alp_impl) &
                              (df['operation'] == op) &
                              (df['size'] == size)]
                abseil_data = df[(df['impl'] == abseil_impl) &
                                 (df['operation'] == op) &
                                 (df['size'] == size)]
                std_data = df[(df['impl'] == std_impl) &
                              (df['operation'] == op) &
                              (df['size'] == size)]

                if not alp_data.empty and not abseil_data.empty and not std_data.empty:
                    alp_time = alp_data['ns_per_op'].values[0]
                    abseil_time = abseil_data['ns_per_op'].values[0]
                    std_time = std_data['ns_per_op'].values[0]

                    speedup_abseil.append(abseil_time / alp_time)
                    speedup_std.append(std_time / alp_time)
                    sizes_found.append(size)

            if sizes_found:
                x = np.arange(len(sizes_found))
                width = 0.35

                bars1 = ax.bar(x - width / 2, speedup_abseil, width,
                               label='vs Abseil', color='#3498db', edgecolor='black', linewidth=0.5)
                bars2 = ax.bar(x + width / 2, speedup_std, width,
                               label='vs std::unordered_set', color='#e74c3c', edgecolor='black', linewidth=0.5)

                ax.axhline(y=1, color='gray', linestyle='--', linewidth=1, alpha=0.7)
                ax.set_ylabel('Speedup (higher = alpmap faster)')
                ax.set_title(f'{op}')
                ax.set_xticks(x)
                ax.set_xticklabels([f'{s // 1000}K' if s >= 1000 else str(s) for s in sizes_found])
                ax.set_xlabel('Number of Elements')
                ax.legend(loc='best')

                # Add value labels
                for bar in bars1:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height,
                            f'{height:.2f}x', ha='center', va='bottom', fontsize=8)
                for bar in bars2:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2, height,
                            f'{height:.2f}x', ha='center', va='bottom', fontsize=8)

        plt.suptitle(f'{dtype}: alpmap Speedup vs Baselines {title_suffix}',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'speedup_{dtype.lower()}.png'),
                    bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Created: speedup_{dtype.lower()}.png")


def main():
    import sys

    # Get input/output paths
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = 'current.csv'

    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = 'graphs'

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data from {csv_path}...")
    df = load_data(csv_path)
    print(f"Loaded {len(df)} benchmark results")

    # Print available implementations for debugging
    print("\nAvailable implementations:")
    for impl in sorted(df['impl'].unique()):
        print(f"  - {impl}")

    print("\nGenerating graphs...")

    # Generate all graphs
    create_comparison_vs_baseline(df, output_dir)
    create_storehash_comparison(df, output_dir)
    create_load_factor_comparison(df, output_dir)
    create_throughput_bar_chart(df, output_dir)
    create_linear_vs_quadratic(df, output_dir)
    create_speedup_chart(df, output_dir)

    print(f"\nAll graphs saved to {output_dir}/")
    print("\nGenerated files:")
    for f in sorted(os.listdir(output_dir)):
        if f.endswith('.png'):
            print(f"  - {f}")


if __name__ == '__main__':
    main()
