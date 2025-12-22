#!/usr/bin/env python3
"""
K-1 Tree Visualization with Domain Specialization

Visualizes the hierarchical tree structure with nodes color-coded by their
primary domain specialization.
"""

import sys
sys.path.insert(0, '/home/x/projects/self-learning-k-1')


def visualize_tree(model, output_file='tree_visualization.txt'):
    """
    Create a text-based tree visualization showing domain specialization.
    
    Args:
        model: Trained K-1 model with domain tracking
        output_file: Path to save visualization
    """
    
    output_lines = []
    output_lines.append("=" * 90)
    output_lines.append("K-1 HIERARCHICAL TREE WITH DOMAIN SPECIALIZATION")
    output_lines.append("=" * 90)
    output_lines.append("")
    output_lines.append("Legend:")
    output_lines.append("  📖 = WikiText Specialist (>50%)")
    output_lines.append("  💻 = Code Specialist (>50%)")
    output_lines.append("  🔬 = Scientific Specialist (>50%)")
    output_lines.append("  ⚖️  = Generalist (<50% in any domain)")
    output_lines.append("")
    output_lines.append("  ⭐ = Strong specialist (>70%)")
    output_lines.append("  ⭐⭐ = Very strong specialist (>80%)")
    output_lines.append("")
    
    def get_domain_emoji(node):
        """Get emoji for node's primary domain."""
        if not node.domain_counts:
            return "❓"
        
        dist = node.get_domain_distribution()
        wiki = dist.get('wikitext', 0)
        code = dist.get('code', 0)
        sci = dist.get('scientific', 0)
        
        # Determine emoji based on highest domain
        if wiki > code and wiki > sci:
            if wiki > 80:
                return "📖⭐⭐"
            elif wiki > 70:
                return "📖⭐"
            elif wiki > 50:
                return "📖"
        elif code > wiki and code > sci:
            if code > 80:
                return "💻⭐⭐"
            elif code > 70:
                return "💻⭐"
            elif code > 50:
                return "💻"
        elif sci > wiki and sci > code:
            if sci > 80:
                return "🔬⭐⭐"
            elif sci > 70:
                return "🔬⭐"
            elif sci > 50:
                return "🔬"
        
        return "⚖️ "
    
    def format_node_info(node):
        """Format node information with domain distribution."""
        if not node.domain_counts:
            return f"Node {node.node_id} (no data)"
        
        dist = node.get_domain_distribution()
        wiki = dist.get('wikitext', 0)
        code = dist.get('code', 0)
        sci = dist.get('scientific', 0)
        primary = node.get_primary_domain()[0] or 'unknown'
        
        emoji = get_domain_emoji(node)
        
        return f"{emoji} Node {node.node_id:<3} | 📖{wiki:>4.0f}% 💻{code:>4.0f}% 🔬{sci:>4.0f}% | {primary}"
    
    def print_tree_recursive(node, prefix="", is_last=True):
        """Recursively print the tree structure."""
        # Current node
        connector = "└── " if is_last else "├── "
        node_info = format_node_info(node)
        output_lines.append(f"{prefix}{connector}{node_info}")
        
        # Children
        if node.child_nodes:
            extension = "    " if is_last else "│   "
            for i, child in enumerate(node.child_nodes):
                child_is_last = (i == len(node.child_nodes) - 1)
                print_tree_recursive(child, prefix + extension, child_is_last)
    
    # Print the tree
    output_lines.append("TREE STRUCTURE:")
    output_lines.append("-" * 90)
    root = model.root
    output_lines.append(f"{get_domain_emoji(root)} ROOT (Node 0)")
    
    for i, child in enumerate(root.child_nodes):
        is_last = (i == len(root.child_nodes) - 1)
        print_tree_recursive(child, "", is_last)
    
    output_lines.append("")
    output_lines.append("=" * 90)
    
    # Statistics
    output_lines.append("")
    output_lines.append("SPECIALIZATION STATISTICS:")
    output_lines.append("-" * 90)
    
    wiki_specialists = sum(1 for n in model.all_nodes if n.domain_counts and
                          n.get_primary_domain()[0] == 'wikitext' and
                          n.get_primary_domain()[2] > 50)
    code_specialists = sum(1 for n in model.all_nodes if n.domain_counts and
                          n.get_primary_domain()[0] == 'code' and
                          n.get_primary_domain()[2] > 50)
    sci_specialists = sum(1 for n in model.all_nodes if n.domain_counts and
                         n.get_primary_domain()[0] == 'scientific' and
                         n.get_primary_domain()[2] > 50)
    
    total_nodes = len(model.all_nodes)
    generalists = total_nodes - wiki_specialists - code_specialists - sci_specialists
    
    output_lines.append(f"Total Nodes: {total_nodes}")
    output_lines.append(f"  📖 WikiText Specialists (>50%):   {wiki_specialists:>2} nodes")
    output_lines.append(f"  💻 Code Specialists (>50%):       {code_specialists:>2} nodes")
    output_lines.append(f"  🔬 Scientific Specialists (>50%): {sci_specialists:>2} nodes")
    output_lines.append(f"  ⚖️  Generalists:                   {generalists:>2} nodes")
    output_lines.append("")
    output_lines.append("=" * 90)
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))
    
    # Also print to console
    print('\n'.join(output_lines))
    
    print(f"\n✓ Visualization saved to: {output_file}")


def visualize_from_demo_results():
    """
    Create visualization from demo results.
    
    Note: This requires running demo_multi_domain.py first or loading
    a trained model with domain tracking.
    """
    print("=" * 70)
    print("K-1 TREE VISUALIZATION")
    print("=" * 70)
    print()
    print("To visualize the tree with domain specialization:")
    print()
    print("Option 1: Run after demo_multi_domain.py")
    print("  1. Run: python demo_multi_domain.py")
    print("  2. Save model with: torch.save(model.state_dict(), 'k1_specialized.pth')")
    print("  3. Load model and call visualize_tree(model)")
    print()
    print("Option 2: Use saved model results")
    print("  (Visualization will be added to demo_multi_domain.py automatically)")
    print()
    print("=" * 70)


if __name__ == '__main__':
    visualize_from_demo_results()
