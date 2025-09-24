"""
DLA-Net Full: Dynamic Low-Rank Adaptive Network
A comprehensive implementation of DLA-Net with all comparison algorithms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import time
import numpy as np
import torch.nn.utils.prune as prune

# --- DLoRF Component ---
class DLoRF(nn.Module):
    def __init__(self, in_features, out_features, k_max, p=1):
        super(DLoRF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k_max = k_max
        self.p = p
        self.U = nn.Parameter(torch.Tensor(in_features, k_max))
        self.V = nn.Parameter(torch.Tensor(out_features, k_max))
        self.s = nn.Parameter(torch.Tensor(k_max))
        self.register_buffer("mask", torch.ones(k_max, dtype=torch.bool))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.U, a=5**0.5)
        nn.init.kaiming_uniform_(self.V, a=5**0.5)
        nn.init.constant_(self.s, 1.0)
        self.mask = torch.ones(self.k_max, dtype=torch.bool, device=self.s.device)

    def forward(self, input):
        s_masked = self.s * self.mask.float()
        s_diag = torch.diag(s_masked)
        W = self.V @ s_diag @ self.U.T
        return F.linear(input, W)

    def weighted_l1_regularization(self):
        weights_f = torch.arange(1, self.k_max + 1, dtype=torch.float32, device=self.s.device)**self.p
        return torch.sum(weights_f * torch.abs(self.s * self.mask.float()))

    def prune_and_reinitialize(self, threshold_p, k_min):
        new_mask = (torch.abs(self.s) > threshold_p)
        self.mask = new_mask.to(self.s.device)
        current_rank = torch.sum(self.mask).item()
        if current_rank < k_min:
            num_to_add = k_min - current_rank
            inactive_indices = torch.where(~self.mask)[0]
            if len(inactive_indices) >= num_to_add:
                indices_to_reactivate = inactive_indices[torch.randperm(len(inactive_indices))[:num_to_add]]
                self.mask[indices_to_reactivate] = True
            else:
                pass
        return torch.sum(self.mask).item()

# --- ASI Component ---
class ASI(nn.Module):
    def __init__(self):
        super(ASI, self).__init__()
    def forward(self, activation):
        return activation
    def group_lasso_regularization(self, activation):
        return torch.sum(torch.norm(activation, p=2, dim=0))

# --- LPQ Component ---
class LPQ(nn.Module):
    def __init__(self, num_bits=8, enabled=True):
        super(LPQ, self).__init__()
        self.num_bits = num_bits
        self.enabled = enabled
        self.scale = nn.Parameter(torch.tensor(0.1))
        self.zero_point = nn.Parameter(torch.tensor(0.0))
    def forward(self, input):
        if not self.enabled:
            return input
        scale = torch.abs(self.scale)
        quantized_output = torch.round(input / scale + self.zero_point) * scale
        return quantized_output

# --- DLANet Model (Integration) ---
class DLANet(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, k_max_lorf, p_lorf=1, num_bits_lpq=8, lpq_enabled=False):
        super(DLANet, self).__init__()
        self.dlorf_layer = DLoRF(in_features, hidden_features, k_max_lorf, p=p_lorf)
        self.activation_fn = nn.ReLU() 
        self.asi_module = ASI()
        self.lpq_module = LPQ(num_bits=num_bits_lpq, enabled=lpq_enabled)
        self.output_layer = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x_dlorf = self.dlorf_layer(x)
        x_activated = self.activation_fn(x_dlorf)
        x_quantized = self.lpq_module(x_activated)
        output = self.output_layer(x_quantized)
        return output, x_activated

    def calculate_total_loss(self, model_output, target, intermediate_activation, lambda_s, lambda_act, task_loss_fn=nn.CrossEntropyLoss()):
        task_loss = task_loss_fn(model_output, target)
        dlorf_reg_loss = self.dlorf_layer.weighted_l1_regularization()
        asi_reg_loss = self.asi_module.group_lasso_regularization(intermediate_activation)
        total_loss = task_loss + lambda_s * dlorf_reg_loss + lambda_act * asi_reg_loss
        return total_loss

# --- Baseline Model for Comparison ---
class BaselineNet(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(BaselineNet, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# --- SVD Pruning Net ---
class SVDNet(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, k):
        super(SVDNet, self).__init__()
        
        # Decompose the first layer using SVD on a random matrix
        # W original shape is (hidden_features, in_features)
        W = torch.randn(hidden_features, in_features)
        U, S, V = torch.svd(W)
        U_k = U[:, :k]  # Shape (hidden_features, k)
        S_k_diag = torch.diag(S[:k])  # Shape (k, k)
        V_k = V[:, :k]  # Shape (in_features, k)
        
        # New layers for the compressed model
        self.fc1_svd_a = nn.Linear(in_features, k, bias=False)
        self.fc1_svd_b = nn.Linear(k, hidden_features, bias=True)
        
        # Correctly initialize the weights
        # V_k.T @ S_k_diag shape: (k, in_features) @ (k, k) -> Incorrect
        # S_k_diag @ V_k.T shape: (k, k) @ (k, in_features) -> Correct
        
        # PyTorch expects weight shape (out_features, in_features)
        # So we need (k, in_features) for fc1_svd_a
        self.fc1_svd_a.weight.data = S_k_diag @ V_k.T 
        # For fc1_svd_b, we need (hidden_features, k)
        self.fc1_svd_b.weight.data = U_k
        self.fc1_svd_b.bias.data = torch.zeros(hidden_features)
        
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)
    
    def forward(self, x):
        x = self.fc1_svd_a(x)
        x = self.fc1_svd_b(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# --- Pruning Model for Comparison ---
class MagnitudePruningNet(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(MagnitudePruningNet, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# --- Knowledge Distillation Net (Student) ---
class KDStudentNet(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(KDStudentNet, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features // 2, out_features)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# --- Training and Evaluation Functions ---
def train_and_evaluate(model, dataloader, optimizer, num_epochs, is_dlanet=False, **kwargs):
    model.train()
    task_loss_fn = nn.CrossEntropyLoss()
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_loss = 0
        correct_predictions = 0
        total_samples = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            if is_dlanet:
                outputs, intermediate_activation = model(inputs)
                loss = model.calculate_total_loss(outputs, targets, intermediate_activation, kwargs.get('lambda_s', 0.0), kwargs.get('lambda_act', 0.0), task_loss_fn)
            else:
                outputs = model(inputs)
                loss = task_loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
            if is_dlanet and kwargs.get('dpr_interval') and (batch_idx + 1) % kwargs['dpr_interval'] == 0:
                model.dlorf_layer.prune_and_reinitialize(kwargs.get('threshold_p'), kwargs.get('k_min'))
        avg_loss = epoch_loss / len(dataloader)
        accuracy = 100 * correct_predictions / total_samples
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    end_time = time.time()
    training_time = end_time - start_time
    return training_time, accuracy

def train_and_prune_magnitude(model, dataloader, optimizer, num_epochs, prune_amount=0.5):
    print("--- Phase 1: Pre-training ---")
    training_time_pre, accuracy_pre = train_and_evaluate(model, dataloader, optimizer, num_epochs)
    
    print("\n--- Phase 2: Pruning ---")
    prune.global_unstructured(( (model.fc1, 'weight'), (model.fc2, 'weight'), ), pruning_method=prune.L1Unstructured, amount=prune_amount)
    
    pruned_params = sum(p.numel() for p in model.parameters())
    pruned_flops = 2 * (model.fc1.weight.numel() + model.fc2.weight.numel())
    
    print("\n--- Phase 3: Fine-tuning ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    training_time_fine, accuracy_fine = train_and_evaluate(model, dataloader, optimizer, num_epochs)
    total_training_time = training_time_pre + training_time_fine
    
    return total_training_time, accuracy_fine, pruned_params, pruned_flops

def train_knowledge_distillation(student_model, teacher_model, train_dataloader, num_epochs, alpha=0.5, T=10):
    teacher_model.eval()
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
    task_loss_fn = nn.CrossEntropyLoss()
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        correct_predictions = 0
        total_samples = 0
        for inputs, targets in train_dataloader:
            optimizer.zero_grad()
            student_outputs = student_model(inputs)
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)

            soft_teacher_outputs = F.softmax(teacher_outputs / T, dim=1)
            soft_student_outputs = F.log_softmax(student_outputs / T, dim=1)

            distillation_loss = nn.KLDivLoss(reduction='batchmean')(soft_student_outputs, soft_teacher_outputs)
            task_loss = task_loss_fn(student_outputs, targets)
            loss = alpha * task_loss + (1 - alpha) * distillation_loss
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            _, predicted = torch.max(student_outputs.data, 1)
            total_samples += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
            
        avg_loss = epoch_loss / len(train_dataloader)
        accuracy = 100 * correct_predictions / total_samples
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    end_time = time.time()
    training_time = end_time - start_time
    return training_time, accuracy

# --- Utility Functions ---
def generate_low_rank_data(num_samples, in_features, out_features, r_star, num_classes):
    """Generate synthetic low-rank data for testing."""
    U_true = torch.randn(in_features, r_star)
    V_true = torch.randn(r_star, out_features)
    W_true = U_true @ V_true
    X = torch.randn(num_samples, in_features)
    logits = X @ W_true + 0.1 * torch.randn(num_samples, out_features)
    y = torch.argmax(logits, dim=1)
    return X, y

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_flops(model, in_features, hidden_features, out_features, k_max_lorf_actual=None):
    """Estimate FLOPs for different model types."""
    if isinstance(model, BaselineNet) or isinstance(model, MagnitudePruningNet) or isinstance(model, KDStudentNet):
        return 2 * (in_features * hidden_features + hidden_features * out_features)
    elif isinstance(model, DLANet):
        if k_max_lorf_actual is None:
            k_max_lorf_actual = torch.sum(model.dlorf_layer.mask).item()
        return 2 * (in_features * k_max_lorf_actual + k_max_lorf_actual * hidden_features) + 2 * (hidden_features * out_features)
    elif isinstance(model, SVDNet):
        k = model.fc1_svd_a.out_features
        # Correct FLOPs calculation for SVD layers
        return 2 * (in_features * k + k * hidden_features) + 2 * (hidden_features * out_features)
    return 0

# --- Main Execution ---
if __name__ == '__main__':
    in_features = 512
    hidden_features = 512
    out_features = 10 
    r_star = 5
    k_max_lorf = 64
    num_epochs = 10
    batch_size = 64
    learning_rate = 0.001
    
    X_train, y_train = generate_low_rank_data(2000, in_features, out_features, r_star, out_features)
    train_dataset = TensorDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    print("--- Experiment 1: Baseline Net ---")
    baseline_model = BaselineNet(in_features, hidden_features, out_features)
    baseline_optimizer = torch.optim.Adam(baseline_model.parameters(), lr=learning_rate)
    baseline_training_time, baseline_accuracy = train_and_evaluate(baseline_model, train_dataloader, baseline_optimizer, num_epochs)
    baseline_params = count_parameters(baseline_model)
    baseline_flops = estimate_flops(baseline_model, in_features, hidden_features, out_features)
    
    print("\n--- Experiment 2: DLA-Net Full ---")
    dlanet_full_model = DLANet(in_features, hidden_features, out_features, k_max_lorf=k_max_lorf, lpq_enabled=False)
    dlanet_full_optimizer = torch.optim.Adam(dlanet_full_model.parameters(), lr=learning_rate)
    dlanet_full_training_time, dlanet_full_accuracy = train_and_evaluate(dlanet_full_model, train_dataloader, dlanet_full_optimizer, num_epochs,
        is_dlanet=True, lambda_s=1e-8, lambda_act=1e-8, dpr_interval=50, threshold_p=1e-5, k_min=r_star)
    final_dlanet_rank = dlanet_full_model.dlorf_layer.prune_and_reinitialize(1e-5, r_star)
    dlanet_full_params = count_parameters(dlanet_full_model)
    dlanet_full_flops = estimate_flops(dlanet_full_model, in_features, hidden_features, out_features, k_max_lorf_actual=final_dlanet_rank)

    print("\n--- Experiment 3: Magnitude Pruning Net ---")
    pruning_model = MagnitudePruningNet(in_features, hidden_features, out_features)
    pruning_optimizer = torch.optim.Adam(pruning_model.parameters(), lr=learning_rate)
    pruning_training_time, pruning_accuracy, pruning_params, pruning_flops = train_and_prune_magnitude(pruning_model, train_dataloader, pruning_optimizer, num_epochs, prune_amount=0.9)
    
    print("\n--- Experiment 4: SVD-Net ---")
    svd_rank = 10
    svd_model = SVDNet(in_features, hidden_features, out_features, k=svd_rank)
    svd_optimizer = torch.optim.Adam(svd_model.parameters(), lr=learning_rate)
    svd_training_time, svd_accuracy = train_and_evaluate(svd_model, train_dataloader, svd_optimizer, num_epochs)
    svd_params = count_parameters(svd_model)
    svd_flops = estimate_flops(svd_model, in_features, hidden_features, out_features)

    print("\n--- Experiment 5: Knowledge Distillation ---")
    teacher_model = BaselineNet(in_features, hidden_features, out_features)
    teacher_optimizer = torch.optim.Adam(teacher_model.parameters(), lr=learning_rate)
    print("--- Training Teacher Model for KD ---")
    train_and_evaluate(teacher_model, train_dataloader, teacher_optimizer, num_epochs)
    
    print("\n--- Training Student Model with Knowledge Distillation ---")
    kd_student_model = KDStudentNet(in_features, hidden_features, out_features)
    kd_training_time, kd_accuracy = train_knowledge_distillation(kd_student_model, teacher_model, train_dataloader, num_epochs)
    kd_params = count_parameters(kd_student_model)
    kd_flops = estimate_flops(kd_student_model, in_features, hidden_features, out_features)

    print("\n\n--- Final Comparison ---")
    print("| Model                     | Accuracy (%) | Params       | FLOPs        | Training Time (s) |")
    print("|---------------------------|--------------|--------------|--------------|-------------------|")
    print(f"| BaselineNet               | {baseline_accuracy:.2f}     | {baseline_params:,} | {baseline_flops:,} | {baseline_training_time:.2f}        |")
    print(f"| DLA-Net Full              | {dlanet_full_accuracy:.2f}     | {dlanet_full_params:,} | {dlanet_full_flops:,} | {dlanet_full_training_time:.2f}        |")
    print(f"| Magnitude Pruning Net     | {pruning_accuracy:.2f}     | {pruning_params:,} | {pruning_flops:,} | {pruning_training_time:.2f}        |")
    print(f"| SVD-Net                   | {svd_accuracy:.2f}     | {svd_params:,} | {svd_flops:,} | {svd_training_time:.2f}        |")
    print(f"| Knowledge Distillation    | {kd_accuracy:.2f}     | {kd_params:,} | {kd_flops:,} | {kd_training_time:.2f}        |")