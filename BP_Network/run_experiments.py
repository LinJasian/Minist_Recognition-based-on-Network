"""
主运行脚本
按顺序运行所有实验
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
from experiments import exp1_basic_bp, exp2_improved_bp, exp3_cnn, exp4_comprehensive_comparison
from utils.config import RESULTS_DIR


def run_all_experiments():
    """运行所有实验"""
    
    print("\n" + "="*70)
    print(" "*15 + "Handwritten Digit Recognition - Complete Experiment Flow")
    print("="*70)
    
    all_results = {}
    
    # Experiment 1: Basic BP Network
    print("\n\n【Starting Experiment 1】" + "-"*50)
    try:
        results_exp1 = exp1_basic_bp.main()
        all_results['exp1_basic_bp'] = results_exp1
        print("✓ Experiment 1 completed successfully")
    except Exception as e:
        print(f"✗ Experiment 1 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Experiment 2: Improved BP Network
    print("\n\n【Starting Experiment 2】" + "-"*50)
    try:
        results_exp2 = exp2_improved_bp.main()
        all_results['exp2_improved_bp'] = results_exp2
        print("✓ Experiment 2 completed successfully")
    except Exception as e:
        print(f"✗ Experiment 2 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Experiment 3: CNN
    print("\n\n【Starting Experiment 3】" + "-"*50)
    try:
        results_exp3 = exp3_cnn.main()
        all_results['exp3_cnn'] = results_exp3
        print("✓ Experiment 3 completed successfully")
    except Exception as e:
        print(f"✗ Experiment 3 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Experiment 4: Comprehensive Comparison
    print("\n\n【Starting Experiment 4】" + "-"*50)
    try:
        results_exp4 = exp4_comprehensive_comparison.main()
        all_results['exp4_comprehensive'] = results_exp4
        print("✓ Experiment 4 completed successfully")
    except Exception as e:
        print(f"✗ Experiment 4 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Save all results
    summary_file = os.path.join(RESULTS_DIR, "all_experiments_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\n\nAll experiment results summarized to: {summary_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("Experiment Summary")
    print("="*70)
    
    if 'exp4_comprehensive' in all_results:
        summary = all_results['exp4_comprehensive'].get('summary', {})
        if summary:
            print(f"\nBest Model: {summary.get('best_model', 'N/A')}")
            print(f"Best Accuracy: {summary.get('best_accuracy', 'N/A'):.4f}")
            
            if 'ranking' in summary:
                print("\nModel Performance Ranking:")
                for rank, (name, acc) in enumerate(summary['ranking'], 1):
                    print(f"  {rank}. {name}: {acc:.4f}")
    
    print("\n" + "="*70 + "\n")


def run_single_experiment(exp_num):
    """运行单个实验
    
    Args:
        exp_num: 实验编号 (1-4)
    """
    experiments = {
        1: exp1_basic_bp,
        2: exp2_improved_bp,
        3: exp3_cnn,
        4: exp4_comprehensive_comparison,
    }
    
    if exp_num not in experiments:
        print(f"错误: 无效的实验编号 {exp_num}，应该是1-4之间的数字")
        return
    
    print(f"\n正在运行实验{exp_num}...\n")
    try:
        results = experiments[exp_num].main()
        print(f"\n✓ 实验{exp_num}成功完成")
        return results
    except Exception as e:
        print(f"\n✗ 实验{exp_num}失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='手写数字识别实验框架')
    parser.add_argument('--exp', type=int, choices=[1, 2, 3, 4],
                       help='运行指定的实验编号 (1-4)，不指定则运行所有实验')
    parser.add_argument('--all', action='store_true',
                       help='运行所有实验')
    
    args = parser.parse_args()
    
    if args.exp:
        run_single_experiment(args.exp)
    else:
        # 默认运行所有实验
        run_all_experiments()
