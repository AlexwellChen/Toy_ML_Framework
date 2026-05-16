#!/bin/bash
# 编译并训练模型，最后生成 loss 曲线图
set -e

cd "$(dirname "$0")/.."

echo "========================================"
echo "  Building toy_ml..."
echo "========================================"
make -f Makefile.simple

echo ""
echo "========================================"
echo "  Training model..."
echo "========================================"
./toy_ml

echo ""
echo "========================================"
echo "  Plotting loss curve..."
echo "========================================"

python3 -c "
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv

iters, loss, train_acc, val_acc, test_acc = [], [], [], [], []
with open('output/mnist_metrics.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        iters.append(int(row['iteration']))
        loss.append(float(row['loss']))
        train_acc.append(float(row['train_acc']) * 100)
        val_acc.append(float(row['val_acc']) * 100)
        test_acc.append(float(row['test_acc']) * 100)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

ax1.plot(iters, loss, 'b-', linewidth=1.5, alpha=0.9)
ax1.set_ylabel('Loss (MSE)', fontsize=13)
ax1.set_title('MNIST Training - Loss Curve', fontsize=15, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_yscale('log')
ax1.annotate(f'Start: {loss[0]:.4f}', xy=(iters[0], loss[0]),
             xytext=(200, loss[0]*3), fontsize=9, color='red',
             arrowprops=dict(arrowstyle='->', color='red', lw=1.2))
ax1.annotate(f'End: {loss[-1]:.2e}', xy=(iters[-1], loss[-1]),
             xytext=(iters[-1]-1200, loss[-1]*100), fontsize=9, color='green',
             arrowprops=dict(arrowstyle='->', color='green', lw=1.2))

ax2.plot(iters, train_acc, 'b-', linewidth=1.5, label='Train Acc', alpha=0.8)
ax2.plot(iters, val_acc, 'g-', linewidth=1.5, label='Val Acc', alpha=0.8)
ax2.plot(iters, test_acc, 'r-', linewidth=1.5, label='Test Acc', alpha=0.8)
ax2.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90% target')
ax2.set_xlabel('Iteration', fontsize=13)
ax2.set_ylabel('Accuracy (%)', fontsize=13)
ax2.set_title('Accuracy Curves', fontsize=15, fontweight='bold')
ax2.legend(fontsize=10, loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 105)

stats = (f'Final Test Acc: {test_acc[-1]:.2f}%\n'
         f'Best Test Acc: {max(test_acc):.2f}%\n'
         f'Best Val Acc: {max(val_acc):.2f}%\n'
         f'Final Loss: {loss[-1]:.2e}')
ax2.text(0.02, 0.35, stats, transform=ax2.transAxes, fontsize=11,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
         verticalalignment='top')

plt.tight_layout()
plt.savefig('output/mnist_loss_curve.png', dpi=150, bbox_inches='tight')
print(f'Saved: output/mnist_loss_curve.png')
print(f'Final Test Accuracy: {test_acc[-1]:.2f}%')
print(f'Best Val Accuracy:  {max(val_acc):.2f}%')
print(f'Final Loss:         {loss[-1]:.2e}')
"

echo ""
echo "========================================"
echo "  Training complete!"
echo "  Outputs: output/mnist_weights.bin"
echo "           output/mnist_graph.txt"
echo "           output/mnist_metrics.csv"
echo "           output/mnist_loss_curve.png"
echo "========================================"
