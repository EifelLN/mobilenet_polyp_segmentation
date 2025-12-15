"""
Main Entry Point for Polyp Segmentation Project
================================================
This script provides a unified interface to:
- Train models (baseline, distillation, teacher)
- Evaluate models on test datasets
- Plot training history from checkpoint JSON files
- Run the Gradio demo app

Usage:
    python main.py                           # Default: Launch Gradio app
    python main.py app                       # Launch Gradio app
    python main.py train --mode distillation # Train with knowledge distillation
    python main.py train --mode baseline     # Train baseline model
    python main.py train --mode teacher      # Train teacher model
    python main.py eval --checkpoint path/to/model.pth
    python main.py plot                      # Plot training history
"""

import argparse
import sys
import os


def run_app(args):
    """Launch the Gradio demo application."""
    print("Launching Gradio App...")
    
    # Import and run app
    from app import demo, MODEL_PATH, DEVICE
    
    print(f"Model: {MODEL_PATH}")
    print(f"Device: {DEVICE}")
    
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True
    )


def run_train(args):
    """Run training with specified mode."""
    print(f"Starting Training (Mode: {args.mode})...")
    
    # Change to src directory for relative imports
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
    sys.path.insert(0, src_dir)
    os.chdir(src_dir)
    
    from train import load_config, train_baseline, train_distillation, train_teacher
    import torch
    
    # Adjust config path to be relative to project root
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join('..', config_path)
    
    config = load_config(config_path)
    
    # Adjust paths in config to be relative to project root
    config['data_root'] = os.path.join('..', config['data_root'])
    config['save_dir'] = os.path.join('..', config['save_dir'])
    if 'teacher_checkpoint' in config:
        config['teacher_checkpoint'] = os.path.join('..', config['teacher_checkpoint'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.mode == 'baseline':
        train_baseline(config, device)
    elif args.mode == 'distillation':
        train_distillation(config, device)
    elif args.mode == 'teacher':
        train_teacher(config, device)


def run_plot(args):
    """Plot training history from checkpoint JSON files."""
    print("Plotting Training History...")
    
    # Change to src directory for relative imports
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
    sys.path.insert(0, src_dir)
    
    from plot_training_history import (
        find_history_files, load_history, plot_training_comparison,
        plot_individual_metrics, print_summary
    )
    
    # Get checkpoints directory path
    checkpoints_dir = args.checkpoints_dir
    if not os.path.isabs(checkpoints_dir):
        checkpoints_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), checkpoints_dir)
    
    # Find or use specified history files
    if args.history_files:
        history_files = args.history_files
    else:
        history_files = find_history_files(checkpoints_dir)
    
    if not history_files:
        print(f"No history files found in {checkpoints_dir}/")
        print("Run training scripts first to generate *_history.json files.")
        return
    
    print(f"Found {len(history_files)} history file(s):")
    for f in history_files:
        print(f"  - {f}")
    
    # Load all histories
    histories = []
    for filepath in history_files:
        try:
            hist = load_history(filepath)
            histories.append(hist)
            print(f"Loaded: {hist.get('model_name', 'Unknown')} ({len(hist['epochs'])} epochs)")
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
    
    if not histories:
        print("No valid history files could be loaded.")
        return
    
    # Print summary
    print_summary(histories)
    
    # Get output path
    output_path = args.output
    if not os.path.isabs(output_path):
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_path)
    
    # Create plots
    plot_training_comparison(histories, save_path=output_path)
    plot_individual_metrics(histories, save_dir=os.path.dirname(output_path) or 'results')
    
    print(f"\nDone! Check {os.path.dirname(output_path)} for generated plots.")


def run_eval(args):
    """Run evaluation on test datasets."""
    print(f"Starting Evaluation...")
    print(f"   Checkpoint: {args.checkpoint}")
    
    # Change to src directory for relative imports
    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
    sys.path.insert(0, src_dir)
    os.chdir(src_dir)
    
    from evaluate import load_config, get_test_transform, evaluate
    from models.student import StudentModel
    from models.teacher import Teacher
    from utils.data_loader import PolypDataset
    from torch.utils.data import DataLoader
    import torch
    import glob
    
    # Load config
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join('..', config_path)
    config = load_config(config_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    
    checkpoint_path = args.checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join('..', checkpoint_path)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    # Load state_dict and detect model type
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Check if it's a teacher model (ResNet50) or student model (MobileNetV2)
    # Teacher models have keys like "model.encoder.layer1..." or "model.encoder.conv1..."
    # Student models have keys like "model.encoder.features..."
    sample_key = list(state_dict.keys())[0] if state_dict else ""
    
    if "encoder.layer" in sample_key or "encoder.conv1" in sample_key or "encoder.bn1" in sample_key:
        # This is a Teacher model (ResNet50-based)
        print(f"   Model Type: Teacher (ResNet50 U-Net++)")
        model = Teacher(device=device)
        # Remove 'model.' prefix if present for loading
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.'):
                new_state_dict[k[6:]] = v  # Remove 'model.' prefix
            else:
                new_state_dict[k] = v
        model.model.load_state_dict(new_state_dict)
        print(f"Teacher model loaded from: {checkpoint_path}")
    else:
        # This is a Student model (MobileNetV2-based)
        print(f"   Model Type: Student (MobileNetV2 U-Net)")
        model = StudentModel(encoder_name=config['student_backbone'])
        model.load_state_dict(state_dict)
        print(f"Student model loaded from: {checkpoint_path}")
    
    model.to(device)
    model.eval()
    
    # Find test datasets
    test_root = os.path.join('..', config['data_root'], 'TestDataset')
    test_folders = sorted(glob.glob(os.path.join(test_root, "*")))
    
    if not test_folders:
        print(f"No test folders found in {test_root}")
        return
    
    print(f"\nFound {len(test_folders)} test datasets: {[os.path.basename(f) for f in test_folders]}")
    print("-" * 60)
    print(f"{'Dataset':<20} | {'Dice':<10} | {'IoU':<10} | {'FPS':<10}")
    print("-" * 60)
    
    total_dice = 0
    
    for folder_path in test_folders:
        dataset_name = os.path.basename(folder_path)
        
        if not os.path.exists(os.path.join(folder_path, "images")):
            continue
        
        dataset = PolypDataset(
            root_dir=folder_path,
            transform=get_test_transform(config['img_size']),
            img_size=config['img_size']
        )
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
        
        save_vis_path = None
        if args.save_visuals:
            save_vis_path = os.path.join('..', 'results', 'visuals', dataset_name)
        
        dice, iou, fps = evaluate(model, loader, device, save_dir=save_vis_path)
        
        print(f"{dataset_name:<20} | {dice:.4f}     | {iou:.4f}     | {fps:.1f}")
        total_dice += dice
    
    print("-" * 60)
    print(f"Average Dice across all datasets: {total_dice / len(test_folders):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Polyp Segmentation - Unified Command Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Launch Gradio app (default)
  python main.py app                                # Launch Gradio app
  python main.py app --share                        # Launch with public URL
  python main.py train --mode distillation          # Train with KD
  python main.py train --mode baseline              # Train baseline
  python main.py train --mode teacher               # Train teacher
  python main.py eval --checkpoint checkpoints/model.pth
  python main.py eval --checkpoint checkpoints/model.pth --save_visuals
  python main.py plot                               # Plot training history
  python main.py plot --output results/my_plot.png  # Custom output path
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # ===== App subparser =====
    app_parser = subparsers.add_parser('app', help='Launch Gradio demo application')
    app_parser.add_argument('--host', type=str, default='127.0.0.1', help='Server host (default: 127.0.0.1)')
    app_parser.add_argument('--port', type=int, default=7860, help='Server port (default: 7860)')
    app_parser.add_argument('--share', action='store_true', help='Create public Gradio link')
    
    # ===== Train subparser =====
    train_parser = subparsers.add_parser('train', help='Train polyp segmentation models')
    train_parser.add_argument(
        '--mode',
        type=str,
        choices=['baseline', 'distillation', 'teacher'],
        default='distillation',
        help='Training mode (default: distillation)'
    )
    train_parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )
    
    # ===== Eval subparser =====
    eval_parser = subparsers.add_parser('eval', help='Evaluate trained models')
    eval_parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint (.pth file)'
    )
    eval_parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file'
    )
    eval_parser.add_argument(
        '--save_visuals',
        action='store_true',
        help='Save visual predictions'
    )
    
    # ===== Plot subparser =====
    plot_parser = subparsers.add_parser('plot', help='Plot training history from checkpoint JSON files')
    plot_parser.add_argument(
        '--checkpoints_dir',
        type=str,
        default='checkpoints',
        help='Directory containing history JSON files (default: checkpoints)'
    )
    plot_parser.add_argument(
        '--output',
        type=str,
        default='results/training_comparison.png',
        help='Output path for the comparison plot (default: results/training_comparison.png)'
    )
    plot_parser.add_argument(
        '--history_files',
        type=str,
        nargs='*',
        default=None,
        help='Specific history files to plot (optional, uses all *_history.json if not specified)'
    )
    
    args = parser.parse_args()
    
    # Default to app if no command specified
    if args.command is None:
        # Create default args for app
        args.command = 'app'
        args.host = '127.0.0.1'
        args.port = 7860
        args.share = False
    
    # Execute the appropriate command
    if args.command == 'app':
        run_app(args)
    elif args.command == 'train':
        run_train(args)
    elif args.command == 'eval':
        run_eval(args)
    elif args.command == 'plot':
        run_plot(args)


if __name__ == "__main__":
    main()
