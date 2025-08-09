#!/usr/bin/env python3
"""
Docker Layer Visualizer
Generate visual representations of Docker image layer structure
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np
from datetime import datetime
import argparse

class LayerVisualizer:
    def __init__(self, work_dir: str):
        self.work_dir = work_dir
        self.load_data()
        
    def load_data(self):
        """Load manifest and config data"""
        manifest_path = os.path.join(self.work_dir, "extracted", "manifest.json")
        with open(manifest_path, 'r') as f:
            self.manifest = json.load(f)[0]
            
        config_path = os.path.join(self.work_dir, "extracted", self.manifest['Config'])
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
    def calculate_layer_sizes(self):
        """Calculate size of each layer"""
        import tarfile
        
        layer_data = []
        
        for i, layer_path in enumerate(self.manifest['Layers']):
            layer_tar = os.path.join(self.work_dir, "extracted", layer_path)
            
            # Calculate layer size
            size = os.path.getsize(layer_tar)
            
            # Get command from history
            if i < len(self.config['history']):
                command = self.config['history'][i].get('created_by', 'Unknown')
                # Truncate long commands
                if len(command) > 50:
                    command = command[:47] + "..."
            else:
                command = "Unknown"
                
            layer_data.append({
                'index': i,
                'size': size,
                'command': command,
                'size_mb': size / (1024 * 1024)
            })
            
        return layer_data
        
    def create_layer_stack_visualization(self, layer_data):
        """Create a stacked bar visualization of layers"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
        
        # Left plot: Stacked layers
        ax1.set_title('Docker Image Layer Stack', fontsize=16, fontweight='bold')
        
        # Calculate cumulative sizes
        cumulative = 0
        colors = plt.cm.viridis(np.linspace(0, 1, len(layer_data)))
        
        for i, layer in enumerate(layer_data):
            # Create rectangle for each layer
            rect = Rectangle((0, cumulative), 1, layer['size_mb'], 
                           facecolor=colors[i], edgecolor='black', linewidth=1)
            ax1.add_patch(rect)
            
            # Add layer label if size is significant
            if layer['size_mb'] > 5:  # Only label layers > 5MB
                ax1.text(0.5, cumulative + layer['size_mb']/2, 
                        f"Layer {i}: {layer['size_mb']:.1f} MB",
                        ha='center', va='center', fontsize=10, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                        
            cumulative += layer['size_mb']
            
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, cumulative)
        ax1.set_ylabel('Size (MB)', fontsize=12)
        ax1.set_xticks([])
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Right plot: Layer size distribution
        ax2.set_title('Layer Size Distribution', fontsize=16, fontweight='bold')
        
        layer_indices = [f"L{l['index']}" for l in layer_data]
        layer_sizes = [l['size_mb'] for l in layer_data]
        
        bars = ax2.bar(layer_indices, layer_sizes, color=colors)
        ax2.set_xlabel('Layer', fontsize=12)
        ax2.set_ylabel('Size (MB)', fontsize=12)
        ax2.grid(True, axis='y', alpha=0.3)
        
        # Rotate x labels if many layers
        if len(layer_data) > 10:
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
            
        # Add value labels on bars
        for bar, size in zip(bars, layer_sizes):
            if size > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{size:.1f}', ha='center', va='bottom', fontsize=9)
                        
        plt.tight_layout()
        
        # Save visualization
        viz_path = os.path.join(self.work_dir, "layer_stack_visualization.png")
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        print(f"[+] Layer stack visualization saved to: {viz_path}")
        
        plt.close()
        
    def create_layer_timeline(self, layer_data):
        """Create timeline visualization showing layer creation"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Parse creation times from history
        times = []
        for i, entry in enumerate(self.config['history']):
            if i < len(layer_data):
                created = entry.get('created', '')
                if created:
                    try:
                        # Parse ISO format timestamp
                        dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                        times.append(dt)
                    except:
                        times.append(None)
                else:
                    times.append(None)
                    
        # Create timeline
        valid_times = [(i, t, layer_data[i]) for i, t in enumerate(times) if t]
        
        if valid_times:
            ax.set_title('Layer Creation Timeline', fontsize=16, fontweight='bold')
            
            # Plot layers on timeline
            for i, (idx, time, layer) in enumerate(valid_times):
                y_pos = i
                
                # Draw timeline point
                ax.scatter(time, y_pos, s=layer['size_mb']*10, 
                          c=f'C{idx%10}', alpha=0.6, edgecolors='black')
                          
                # Add layer info
                info_text = f"Layer {idx}: {layer['size_mb']:.1f} MB"
                ax.text(time, y_pos + 0.1, info_text, fontsize=9, ha='left')
                
            ax.set_yticks(range(len(valid_times)))
            ax.set_yticklabels([f"L{v[0]}" for v in valid_times])
            ax.set_xlabel('Creation Time', fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Format x-axis
            fig.autofmt_xdate()
            
        plt.tight_layout()
        
        # Save visualization
        timeline_path = os.path.join(self.work_dir, "layer_timeline.png")
        plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
        print(f"[+] Layer timeline saved to: {timeline_path}")
        
        plt.close()
        
    def create_command_analysis(self, layer_data):
        """Analyze and visualize commands used in layers"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Categorize commands
        command_categories = {
            'Package Manager': ['apt', 'yum', 'apk', 'pip', 'npm', 'gem'],
            'File Operations': ['COPY', 'ADD', 'mkdir', 'chmod', 'chown'],
            'Network': ['wget', 'curl', 'git'],
            'Configuration': ['ENV', 'WORKDIR', 'USER', 'EXPOSE'],
            'Build': ['make', 'gcc', 'g++', 'javac'],
            'Other': []
        }
        
        category_sizes = {cat: 0 for cat in command_categories}
        category_counts = {cat: 0 for cat in command_categories}
        
        for layer in layer_data:
            cmd = layer['command']
            categorized = False
            
            for category, keywords in command_categories.items():
                if category != 'Other' and any(kw in cmd for kw in keywords):
                    category_sizes[category] += layer['size_mb']
                    category_counts[category] += 1
                    categorized = True
                    break
                    
            if not categorized:
                category_sizes['Other'] += layer['size_mb']
                category_counts['Other'] += 1
                
        # Plot 1: Size by category
        ax1.set_title('Layer Size by Command Category', fontsize=14, fontweight='bold')
        categories = list(category_sizes.keys())
        sizes = list(category_sizes.values())
        
        # Filter out empty categories
        non_empty = [(c, s) for c, s in zip(categories, sizes) if s > 0]
        if non_empty:
            categories, sizes = zip(*non_empty)
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
            wedges, texts, autotexts = ax1.pie(sizes, labels=categories, autopct='%1.1f%%',
                                               colors=colors, startangle=90)
            
            # Make percentage text more readable
            for autotext in autotexts:
                autotext.set_color('black')
                autotext.set_fontsize(10)
                autotext.set_fontweight('bold')
                
        # Plot 2: Most common operations
        ax2.set_title('Most Common Layer Operations', fontsize=14, fontweight='bold')
        
        # Extract operation types
        operations = {}
        for layer in layer_data:
            cmd = layer['command']
            # Extract first word/operation
            if cmd and cmd != 'Unknown':
                op = cmd.split()[0].strip('#(nop)').strip()
                if op:
                    operations[op] = operations.get(op, 0) + 1
                    
        # Get top 10 operations
        top_ops = sorted(operations.items(), key=lambda x: x[1], reverse=True)[:10]
        
        if top_ops:
            ops, counts = zip(*top_ops)
            y_pos = np.arange(len(ops))
            
            bars = ax2.barh(y_pos, counts, color='skyblue', edgecolor='navy')
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(ops)
            ax2.set_xlabel('Count')
            ax2.grid(True, axis='x', alpha=0.3)
            
            # Add value labels
            for bar, count in zip(bars, counts):
                ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                        str(count), va='center')
                        
        plt.tight_layout()
        
        # Save visualization
        cmd_analysis_path = os.path.join(self.work_dir, "command_analysis.png")
        plt.savefig(cmd_analysis_path, dpi=300, bbox_inches='tight')
        print(f"[+] Command analysis saved to: {cmd_analysis_path}")
        
        plt.close()
        
    def generate_summary_report(self, layer_data):
        """Generate a visual summary report"""
        fig = plt.figure(figsize=(16, 20))
        
        # Title
        fig.suptitle('Docker Image Analysis Summary', fontsize=20, fontweight='bold')
        
        # Grid layout
        gs = fig.add_gridspec(5, 2, height_ratios=[1, 2, 2, 2, 1], hspace=0.3, wspace=0.3)
        
        # Summary statistics
        ax_stats = fig.add_subplot(gs[0, :])
        ax_stats.axis('off')
        
        total_size = sum(l['size_mb'] for l in layer_data)
        stats_text = (
            f"Total Image Size: {total_size:.1f} MB | "
            f"Number of Layers: {len(layer_data)} | "
            f"Average Layer Size: {total_size/len(layer_data):.1f} MB | "
            f"Largest Layer: {max(l['size_mb'] for l in layer_data):.1f} MB"
        )
        
        ax_stats.text(0.5, 0.5, stats_text, ha='center', va='center',
                     fontsize=14, bbox=dict(boxstyle="round,pad=0.5", 
                     facecolor='lightblue', alpha=0.8))
                     
        # Layer sizes bar chart
        ax_sizes = fig.add_subplot(gs[1, :])
        layer_indices = [f"Layer {l['index']}" for l in layer_data]
        layer_sizes = [l['size_mb'] for l in layer_data]
        
        bars = ax_sizes.bar(layer_indices, layer_sizes, 
                           color=plt.cm.viridis(np.linspace(0, 1, len(layer_data))))
        ax_sizes.set_title('Layer Sizes', fontsize=16)
        ax_sizes.set_ylabel('Size (MB)')
        ax_sizes.grid(True, axis='y', alpha=0.3)
        
        if len(layer_data) > 10:
            plt.setp(ax_sizes.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
        # Cumulative size
        ax_cumulative = fig.add_subplot(gs[2, 0])
        cumulative_sizes = np.cumsum([l['size_mb'] for l in layer_data])
        ax_cumulative.plot(range(len(layer_data)), cumulative_sizes, 
                          'b-', linewidth=2, marker='o')
        ax_cumulative.fill_between(range(len(layer_data)), cumulative_sizes, 
                                  alpha=0.3, color='blue')
        ax_cumulative.set_title('Cumulative Image Size', fontsize=14)
        ax_cumulative.set_xlabel('Layer Index')
        ax_cumulative.set_ylabel('Cumulative Size (MB)')
        ax_cumulative.grid(True, alpha=0.3)
        
        # Size distribution histogram
        ax_hist = fig.add_subplot(gs[2, 1])
        ax_hist.hist(layer_sizes, bins=min(20, len(layer_data)), 
                    color='green', alpha=0.7, edgecolor='black')
        ax_hist.set_title('Layer Size Distribution', fontsize=14)
        ax_hist.set_xlabel('Size (MB)')
        ax_hist.set_ylabel('Frequency')
        ax_hist.grid(True, axis='y', alpha=0.3)
        
        # Top 5 largest layers details
        ax_top = fig.add_subplot(gs[3, :])
        ax_top.axis('off')
        
        sorted_layers = sorted(layer_data, key=lambda x: x['size_mb'], reverse=True)[:5]
        
        table_data = []
        for layer in sorted_layers:
            cmd_preview = layer['command'][:60] + "..." if len(layer['command']) > 60 else layer['command']
            table_data.append([
                f"Layer {layer['index']}",
                f"{layer['size_mb']:.1f} MB",
                cmd_preview
            ])
            
        table = ax_top.table(cellText=table_data,
                           colLabels=['Layer', 'Size', 'Command'],
                           cellLoc='left',
                           loc='center',
                           colWidths=[0.15, 0.15, 0.7])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        ax_top.text(0.5, 0.9, 'Top 5 Largest Layers', ha='center', 
                   fontsize=14, fontweight='bold', transform=ax_top.transAxes)
                   
        # Footer
        ax_footer = fig.add_subplot(gs[4, :])
        ax_footer.axis('off')
        ax_footer.text(0.5, 0.5, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                      ha='center', va='center', fontsize=10, style='italic')
                      
        # Save summary
        summary_path = os.path.join(self.work_dir, "analysis_summary.png")
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        print(f"[+] Analysis summary saved to: {summary_path}")
        
        plt.close()
        
    def run_visualization(self):
        """Run all visualizations"""
        print("\n[GENERATING VISUALIZATIONS]")
        print("=" * 60)
        
        # Calculate layer sizes
        layer_data = self.calculate_layer_sizes()
        
        # Generate visualizations
        self.create_layer_stack_visualization(layer_data)
        self.create_layer_timeline(layer_data)
        self.create_command_analysis(layer_data)
        self.generate_summary_report(layer_data)
        
        print("\n[+] All visualizations complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Docker Layer Visualizer"
    )
    parser.add_argument("work_dir", help="Work directory from main analyzer")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.work_dir):
        print(f"Error: Work directory {args.work_dir} not found")
        exit(1)
        
    visualizer = LayerVisualizer(args.work_dir)
    visualizer.run_visualization()


if __name__ == "__main__":
    main()
