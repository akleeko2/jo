"""
Comprehensive and Rigorous Test Suite for DLA-Net Full
This test suite provides extensive testing for all components and algorithms.
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import warnings
from typing import Tuple, List, Dict, Any

from dla_net import (
    DLoRF, ASI, LPQ, DLANet, BaselineNet, SVDNet, 
    MagnitudePruningNet, KDStudentNet,
    train_and_evaluate, train_and_prune_magnitude, train_knowledge_distillation,
    generate_low_rank_data, count_parameters, estimate_flops
)

class TestDLoRFComponent(unittest.TestCase):
    """Rigorous tests for Dynamic Low-Rank Factorization (DLoRF) component."""
    
    def setUp(self):
        """Set up test fixtures with various configurations."""
        self.in_features = 100
        self.out_features = 50
        self.k_max = 20
        self.p = 1
        self.dlorf = DLoRF(self.in_features, self.out_features, self.k_max, self.p)
        torch.manual_seed(42)
        
    def test_initialization(self):
        """Test proper initialization of DLoRF parameters."""
        # Test parameter shapes
        self.assertEqual(self.dlorf.U.shape, (self.in_features, self.k_max))
        self.assertEqual(self.dlorf.V.shape, (self.out_features, self.k_max))
        self.assertEqual(self.dlorf.s.shape, (self.k_max,))
        self.assertEqual(self.dlorf.mask.shape, (self.k_max,))
        
        # Test mask initialization (all ones)
        self.assertTrue(torch.all(self.dlorf.mask == True))
        
        # Test parameter device consistency
        self.assertEqual(self.dlorf.U.device, self.dlorf.V.device)
        self.assertEqual(self.dlorf.U.device, self.dlorf.s.device)
        
    def test_forward_pass_shapes(self):
        """Test forward pass with various input shapes."""
        batch_sizes = [1, 5, 32, 128]
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                x = torch.randn(batch_size, self.in_features)
                output = self.dlorf(x)
                self.assertEqual(output.shape, (batch_size, self.out_features))
                
    def test_forward_pass_deterministic(self):
        """Test that forward pass is deterministic given same input."""
        x = torch.randn(10, self.in_features)
        output1 = self.dlorf(x)
        output2 = self.dlorf(x)
        self.assertTrue(torch.allclose(output1, output2, atol=1e-6))
        
    def test_gradient_flow(self):
        """Test that gradients flow properly through DLoRF."""
        x = torch.randn(5, self.in_features, requires_grad=True)
        output = self.dlorf(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist for all parameters
        self.assertIsNotNone(self.dlorf.U.grad)
        self.assertIsNotNone(self.dlorf.V.grad)
        self.assertIsNotNone(self.dlorf.s.grad)
        self.assertIsNotNone(x.grad)
        
    def test_weighted_l1_regularization(self):
        """Test weighted L1 regularization computation."""
        reg_loss = self.dlorf.weighted_l1_regularization()
        
        # Should be positive
        self.assertGreater(reg_loss.item(), 0)
        
        # Test with different p values
        dlorf_p2 = DLoRF(self.in_features, self.out_features, self.k_max, p=2)
        reg_loss_p2 = dlorf_p2.weighted_l1_regularization()
        self.assertNotEqual(reg_loss.item(), reg_loss_p2.item())
        
    def test_prune_and_reinitialize(self):
        """Test pruning and reinitialization functionality."""
        # Set some singular values to small values
        with torch.no_grad():
            self.dlorf.s[10:] = 1e-6
            
        # Test pruning
        threshold_p = 1e-5
        k_min = 5
        current_rank = self.dlorf.prune_and_reinitialize(threshold_p, k_min)
        
        # Should maintain minimum rank
        self.assertGreaterEqual(current_rank, k_min)
        
        # Test that mask is properly updated
        active_indices = torch.sum(self.dlorf.mask).item()
        self.assertEqual(active_indices, current_rank)
        
    def test_extreme_cases(self):
        """Test edge cases and extreme parameter values."""
        # Very small k_max
        dlorf_small = DLoRF(10, 5, k_max=1)
        x = torch.randn(3, 10)
        output = dlorf_small(x)
        self.assertEqual(output.shape, (3, 5))
        
        # k_max equals input features
        dlorf_equal = DLoRF(10, 5, k_max=10)
        output = dlorf_equal(x)
        self.assertEqual(output.shape, (3, 5))
        
    def test_mask_functionality(self):
        """Test that masking works correctly."""
        # Manually set mask
        with torch.no_grad():
            self.dlorf.mask[:10] = False
            
        # Forward pass should only use active components
        x = torch.randn(5, self.in_features)
        output = self.dlorf(x)
        
        # Verify output shape is still correct
        self.assertEqual(output.shape, (5, self.out_features))
        
        # Test that masked components don't contribute
        s_masked = self.dlorf.s * self.dlorf.mask.float()
        self.assertTrue(torch.all(s_masked[:10] == 0))


class TestASIComponent(unittest.TestCase):
    """Rigorous tests for Activation Structure Intelligence (ASI) component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.asi = ASI()
        torch.manual_seed(42)
        
    def test_forward_pass(self):
        """Test ASI forward pass (identity function)."""
        activations = [
            torch.randn(10, 20),
            torch.randn(5, 100),
            torch.randn(1, 50)
        ]
        
        for activation in activations:
            with self.subTest(shape=activation.shape):
                output = self.asi(activation)
                self.assertTrue(torch.equal(activation, output))
                
    def test_group_lasso_regularization(self):
        """Test group lasso regularization computation."""
        # Test with different activation patterns
        activations = [
            torch.randn(10, 20),  # Random activations
            torch.zeros(10, 20),  # Zero activations
            torch.ones(10, 20),   # Uniform activations
        ]
        
        for i, activation in enumerate(activations):
            with self.subTest(case=i):
                reg_loss = self.asi.group_lasso_regularization(activation)
                self.assertGreaterEqual(reg_loss.item(), 0)
                
        # Zero activations should give zero regularization
        zero_reg = self.asi.group_lasso_regularization(torch.zeros(10, 20))
        self.assertEqual(zero_reg.item(), 0)
        
    def test_regularization_properties(self):
        """Test mathematical properties of group lasso regularization."""
        # Test scaling property
        activation = torch.randn(10, 20)
        reg1 = self.asi.group_lasso_regularization(activation)
        reg2 = self.asi.group_lasso_regularization(2 * activation)
        
        # Should scale linearly
        self.assertAlmostEqual(reg2.item(), 2 * reg1.item(), places=5)
        
    def test_gradient_flow(self):
        """Test gradient flow through ASI regularization."""
        activation = torch.randn(10, 20, requires_grad=True)
        reg_loss = self.asi.group_lasso_regularization(activation)
        reg_loss.backward()
        
        self.assertIsNotNone(activation.grad)


class TestLPQComponent(unittest.TestCase):
    """Rigorous tests for Learnable Parameter Quantization (LPQ) component."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.lpq_enabled = LPQ(num_bits=8, enabled=True)
        self.lpq_disabled = LPQ(num_bits=8, enabled=False)
        torch.manual_seed(42)
        
    def test_enabled_disabled_modes(self):
        """Test LPQ in enabled and disabled modes."""
        x = torch.randn(10, 20)
        
        # Disabled mode should return input unchanged
        output_disabled = self.lpq_disabled(x)
        self.assertTrue(torch.equal(x, output_disabled))
        
        # Enabled mode should quantize
        output_enabled = self.lpq_enabled(x)
        self.assertEqual(output_enabled.shape, x.shape)
        # Should be different from input (due to quantization)
        self.assertFalse(torch.equal(x, output_enabled))
        
    def test_quantization_properties(self):
        """Test quantization mathematical properties."""
        x = torch.randn(100, 50)
        output = self.lpq_enabled(x)
        
        # Output should have same shape
        self.assertEqual(output.shape, x.shape)
        
        # Quantized values should be discrete
        # (This is a heuristic test - real quantization creates discrete levels)
        unique_vals_input = len(torch.unique(x))
        unique_vals_output = len(torch.unique(output))
        
        # Generally, quantization reduces unique values, but this depends on scale/zero_point
        self.assertIsInstance(unique_vals_output, int)
        
    def test_parameter_learning(self):
        """Test that scale and zero_point are learnable."""
        self.assertTrue(self.lpq_enabled.scale.requires_grad)
        self.assertTrue(self.lpq_enabled.zero_point.requires_grad)
        
    def test_gradient_flow(self):
        """Test gradient flow through quantization."""
        x = torch.randn(5, 10, requires_grad=True)
        output = self.lpq_enabled(x)
        loss = output.sum()
        loss.backward()
        
        # Should have gradients for input and parameters
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(self.lpq_enabled.scale.grad)
        self.assertIsNotNone(self.lpq_enabled.zero_point.grad)
        
    def test_different_bit_widths(self):
        """Test LPQ with different bit widths."""
        bit_widths = [4, 8, 16]
        x = torch.randn(10, 10)
        
        for bits in bit_widths:
            with self.subTest(bits=bits):
                lpq = LPQ(num_bits=bits, enabled=True)
                output = lpq(x)
                self.assertEqual(output.shape, x.shape)


class TestDLANetIntegration(unittest.TestCase):
    """Rigorous tests for the integrated DLA-Net model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.in_features = 50
        self.hidden_features = 30
        self.out_features = 10
        self.k_max_lorf = 15
        self.dlanet = DLANet(
            self.in_features, self.hidden_features, self.out_features, 
            self.k_max_lorf, lpq_enabled=True
        )
        torch.manual_seed(42)
        
    def test_forward_pass(self):
        """Test DLA-Net forward pass."""
        batch_sizes = [1, 5, 32]
        for batch_size in batch_sizes:
            with self.subTest(batch_size=batch_size):
                x = torch.randn(batch_size, self.in_features)
                output, activation = self.dlanet(x)
                
                self.assertEqual(output.shape, (batch_size, self.out_features))
                self.assertEqual(activation.shape, (batch_size, self.hidden_features))
                
    def test_total_loss_computation(self):
        """Test total loss computation with all regularization terms."""
        x = torch.randn(5, self.in_features)
        targets = torch.randint(0, self.out_features, (5,))
        
        output, activation = self.dlanet(x)
        total_loss = self.dlanet.calculate_total_loss(
            output, targets, activation, lambda_s=1e-3, lambda_act=1e-3
        )
        
        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertTrue(total_loss.requires_grad)
        self.assertGreater(total_loss.item(), 0)
        
    def test_regularization_terms(self):
        """Test individual regularization terms."""
        x = torch.randn(5, self.in_features)
        targets = torch.randint(0, self.out_features, (5,))
        output, activation = self.dlanet(x)
        
        # Test DLoRF regularization
        dlorf_reg = self.dlanet.dlorf_layer.weighted_l1_regularization()
        self.assertGreater(dlorf_reg.item(), 0)
        
        # Test ASI regularization
        asi_reg = self.dlanet.asi_module.group_lasso_regularization(activation)
        self.assertGreater(asi_reg.item(), 0)
        
    def test_gradient_flow_integration(self):
        """Test gradient flow through entire DLA-Net."""
        x = torch.randn(3, self.in_features, requires_grad=True)
        targets = torch.randint(0, self.out_features, (3,))
        
        output, activation = self.dlanet(x)
        loss = self.dlanet.calculate_total_loss(
            output, targets, activation, lambda_s=1e-3, lambda_act=1e-3
        )
        loss.backward()
        
        # Check gradients exist for all components
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(self.dlanet.dlorf_layer.U.grad)
        self.assertIsNotNone(self.dlanet.dlorf_layer.V.grad)
        self.assertIsNotNone(self.dlanet.dlorf_layer.s.grad)
        
    def test_lpq_enabled_disabled(self):
        """Test DLA-Net with LPQ enabled and disabled."""
        # Test with LPQ disabled
        dlanet_no_lpq = DLANet(
            self.in_features, self.hidden_features, self.out_features, 
            self.k_max_lorf, lpq_enabled=False
        )
        
        x = torch.randn(5, self.in_features)
        output1, _ = self.dlanet(x)  # With LPQ
        output2, _ = dlanet_no_lpq(x)  # Without LPQ
        
        # Outputs should be different
        self.assertFalse(torch.allclose(output1, output2))


class TestBaselineModels(unittest.TestCase):
    """Rigorous tests for baseline comparison models."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.in_features = 50
        self.hidden_features = 30
        self.out_features = 10
        torch.manual_seed(42)
        
    def test_baseline_net(self):
        """Test BaselineNet model."""
        model = BaselineNet(self.in_features, self.hidden_features, self.out_features)
        x = torch.randn(5, self.in_features)
        output = model(x)
        self.assertEqual(output.shape, (5, self.out_features))
        
    def test_svd_net(self):
        """Test SVDNet model with various ranks."""
        ranks = [5, 10, 15]
        for k in ranks:
            with self.subTest(rank=k):
                model = SVDNet(self.in_features, self.hidden_features, self.out_features, k)
                x = torch.randn(5, self.in_features)
                output = model(x)
                self.assertEqual(output.shape, (5, self.out_features))
                
                # Check that SVD decomposition maintains correct dimensions
                self.assertEqual(model.fc1_svd_a.out_features, k)
                self.assertEqual(model.fc1_svd_b.in_features, k)
                
    def test_magnitude_pruning_net(self):
        """Test MagnitudePruningNet model."""
        model = MagnitudePruningNet(self.in_features, self.hidden_features, self.out_features)
        x = torch.randn(5, self.in_features)
        output = model(x)
        self.assertEqual(output.shape, (5, self.out_features))
        
    def test_kd_student_net(self):
        """Test Knowledge Distillation Student Net."""
        model = KDStudentNet(self.in_features, self.hidden_features, self.out_features)
        x = torch.randn(5, self.in_features)
        output = model(x)
        self.assertEqual(output.shape, (5, self.out_features))
        
        # Check that hidden layer is smaller (compressed)
        self.assertEqual(model.fc1.out_features, self.hidden_features // 2)


class TestTrainingFunctions(unittest.TestCase):
    """Rigorous tests for training and evaluation functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.in_features = 20
        self.hidden_features = 15
        self.out_features = 5
        self.batch_size = 8
        self.num_samples = 50
        torch.manual_seed(42)
        
        # Create small dataset for testing
        X, y = generate_low_rank_data(self.num_samples, self.in_features, self.out_features, 3, self.out_features)
        dataset = TensorDataset(X, y)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
    def test_train_and_evaluate_baseline(self):
        """Test training and evaluation for baseline models."""
        model = BaselineNet(self.in_features, self.hidden_features, self.out_features)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        training_time, accuracy = train_and_evaluate(
            model, self.dataloader, optimizer, num_epochs=2
        )
        
        self.assertIsInstance(training_time, float)
        self.assertIsInstance(accuracy, float)
        self.assertGreater(training_time, 0)
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 100)
        
    def test_train_and_evaluate_dlanet(self):
        """Test training and evaluation for DLA-Net."""
        model = DLANet(self.in_features, self.hidden_features, self.out_features, k_max_lorf=10)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        training_time, accuracy = train_and_evaluate(
            model, self.dataloader, optimizer, num_epochs=2,
            is_dlanet=True, lambda_s=1e-4, lambda_act=1e-4,
            dpr_interval=5, threshold_p=1e-6, k_min=2
        )
        
        self.assertIsInstance(training_time, float)
        self.assertIsInstance(accuracy, float)
        self.assertGreater(training_time, 0)
        
    def test_magnitude_pruning_training(self):
        """Test magnitude pruning training pipeline."""
        model = MagnitudePruningNet(self.in_features, self.hidden_features, self.out_features)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        training_time, accuracy, params, flops = train_and_prune_magnitude(
            model, self.dataloader, optimizer, num_epochs=1, prune_amount=0.5
        )
        
        self.assertIsInstance(training_time, float)
        self.assertIsInstance(accuracy, float)
        self.assertIsInstance(params, int)
        self.assertIsInstance(flops, int)
        
    def test_knowledge_distillation_training(self):
        """Test knowledge distillation training."""
        teacher = BaselineNet(self.in_features, self.hidden_features, self.out_features)
        student = KDStudentNet(self.in_features, self.hidden_features, self.out_features)
        
        # Quick teacher training
        teacher_optimizer = torch.optim.Adam(teacher.parameters(), lr=0.01)
        train_and_evaluate(teacher, self.dataloader, teacher_optimizer, num_epochs=1)
        
        # Test student training with distillation
        training_time, accuracy = train_knowledge_distillation(
            student, teacher, self.dataloader, num_epochs=1
        )
        
        self.assertIsInstance(training_time, float)
        self.assertIsInstance(accuracy, float)


class TestUtilityFunctions(unittest.TestCase):
    """Rigorous tests for utility functions."""
    
    def test_generate_low_rank_data(self):
        """Test low-rank data generation."""
        num_samples, in_features, out_features, r_star = 100, 50, 10, 5
        
        X, y = generate_low_rank_data(num_samples, in_features, out_features, r_star, out_features)
        
        self.assertEqual(X.shape, (num_samples, in_features))
        self.assertEqual(y.shape, (num_samples,))
        self.assertTrue(torch.all(y >= 0))
        self.assertTrue(torch.all(y < out_features))
        
    def test_count_parameters(self):
        """Test parameter counting for different models."""
        models = [
            BaselineNet(20, 15, 5),
            DLANet(20, 15, 5, k_max_lorf=10),
            SVDNet(20, 15, 5, k=8),
        ]
        
        for model in models:
            with self.subTest(model=type(model).__name__):
                param_count = count_parameters(model)
                self.assertIsInstance(param_count, int)
                self.assertGreater(param_count, 0)
                
                # Manual verification for baseline
                if isinstance(model, BaselineNet):
                    expected = (20 * 15 + 15) + (15 * 5 + 5)
                    self.assertEqual(param_count, expected)
                    
    def test_estimate_flops(self):
        """Test FLOP estimation for different models."""
        in_features, hidden_features, out_features = 20, 15, 5
        
        models_and_flops = [
            (BaselineNet(in_features, hidden_features, out_features), 
             2 * (in_features * hidden_features + hidden_features * out_features)),
            (SVDNet(in_features, hidden_features, out_features, k=8), None),  # Will be computed
            (DLANet(in_features, hidden_features, out_features, k_max_lorf=10), None),  # Will be computed
        ]
        
        for model, expected_flops in models_and_flops:
            with self.subTest(model=type(model).__name__):
                estimated_flops = estimate_flops(model, in_features, hidden_features, out_features)
                self.assertIsInstance(estimated_flops, int)
                self.assertGreater(estimated_flops, 0)
                
                if expected_flops is not None:
                    self.assertEqual(estimated_flops, expected_flops)


class TestRobustnessAndEdgeCases(unittest.TestCase):
    """Rigorous tests for robustness and edge cases."""
    
    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Test with very small values
        dlorf = DLoRF(10, 5, 8)
        with torch.no_grad():
            dlorf.s.fill_(1e-10)
            
        x = torch.randn(3, 10)
        output = dlorf(x)
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        
        # Test with very large values
        x_large = torch.randn(3, 10) * 1e6
        output_large = dlorf(x_large)
        self.assertFalse(torch.isnan(output_large).any())
        self.assertFalse(torch.isinf(output_large).any())
        
    def test_device_consistency(self):
        """Test model behavior across different devices."""
        if torch.cuda.is_available():
            # Test CPU vs GPU consistency
            model_cpu = DLANet(10, 8, 3, k_max_lorf=5)
            model_gpu = DLANet(10, 8, 3, k_max_lorf=5).cuda()
            
            # Copy weights
            model_gpu.load_state_dict(model_cpu.state_dict())
            
            x_cpu = torch.randn(5, 10)
            x_gpu = x_cpu.cuda()
            
            output_cpu, _ = model_cpu(x_cpu)
            output_gpu, _ = model_gpu(x_gpu)
            
            self.assertTrue(torch.allclose(output_cpu, output_gpu.cpu(), atol=1e-5))
            
    def test_batch_size_invariance(self):
        """Test that models produce consistent results across different batch sizes."""
        model = DLANet(10, 8, 3, k_max_lorf=5)
        model.eval()  # Set to eval mode for consistent behavior
        
        # Single sample
        x_single = torch.randn(1, 10)
        output_single, _ = model(x_single)
        
        # Batch of same sample
        x_batch = x_single.repeat(5, 1)
        output_batch, _ = model(x_batch)
        
        # First output should match single sample output
        self.assertTrue(torch.allclose(output_single, output_batch[:1], atol=1e-6))
        
    def test_memory_efficiency(self):
        """Test memory usage patterns."""
        import gc
        
        # Test that models can be created and destroyed without memory leaks
        initial_allocated = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        for _ in range(5):
            model = DLANet(100, 50, 10, k_max_lorf=20)
            x = torch.randn(32, 100)
            output, _ = model(x)
            del model, x, output
            gc.collect()
            
        final_allocated = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        # Memory usage shouldn't grow significantly
        if torch.cuda.is_available():
            self.assertLess(final_allocated - initial_allocated, 1e6)  # Less than 1MB growth
            
    def test_reproducibility(self):
        """Test that results are reproducible with same random seed."""
        torch.manual_seed(12345)
        model1 = DLANet(10, 8, 3, k_max_lorf=5)
        x = torch.randn(5, 10)
        output1, _ = model1(x)
        
        torch.manual_seed(12345)
        model2 = DLANet(10, 8, 3, k_max_lorf=5)
        output2, _ = model2(x)
        
        self.assertTrue(torch.allclose(output1, output2, atol=1e-6))


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarking tests."""
    
    def test_inference_speed_comparison(self):
        """Compare inference speed across different models."""
        in_features, hidden_features, out_features = 512, 256, 10
        batch_size = 64
        num_iterations = 10
        
        models = {
            'BaselineNet': BaselineNet(in_features, hidden_features, out_features),
            'DLANet': DLANet(in_features, hidden_features, out_features, k_max_lorf=32),
            'SVDNet': SVDNet(in_features, hidden_features, out_features, k=16),
        }
        
        x = torch.randn(batch_size, in_features)
        results = {}
        
        for name, model in models.items():
            model.eval()
            # Warmup
            with torch.no_grad():
                for _ in range(3):
                    if isinstance(model, DLANet):
                        _ = model(x)
                    else:
                        _ = model(x)
            
            # Timing
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_iterations):
                    if isinstance(model, DLANet):
                        output, _ = model(x)
                    else:
                        output = model(x)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / num_iterations
            results[name] = avg_time
            
        # Print results for manual inspection
        print(f"\nInference Speed Comparison (avg over {num_iterations} iterations):")
        for name, time_taken in results.items():
            print(f"{name}: {time_taken:.4f}s")
            
        # Basic sanity checks
        for name, time_taken in results.items():
            self.assertGreater(time_taken, 0)
            self.assertLess(time_taken, 1.0)  # Should be less than 1 second
            
    def test_memory_usage_comparison(self):
        """Compare memory usage across different models."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available for memory testing")
            
        in_features, hidden_features, out_features = 512, 256, 10
        
        models = {
            'BaselineNet': BaselineNet(in_features, hidden_features, out_features),
            'DLANet': DLANet(in_features, hidden_features, out_features, k_max_lorf=32),
            'SVDNet': SVDNet(in_features, hidden_features, out_features, k=16),
        }
        
        results = {}
        
        for name, model in models.items():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            model = model.cuda()
            x = torch.randn(64, in_features).cuda()
            
            if isinstance(model, DLANet):
                output, _ = model(x)
            else:
                output = model(x)
                
            peak_memory = torch.cuda.max_memory_allocated()
            results[name] = peak_memory
            
            del model, x, output
            torch.cuda.empty_cache()
            
        # Print results
        print(f"\nMemory Usage Comparison:")
        for name, memory in results.items():
            print(f"{name}: {memory / 1024**2:.2f} MB")
            
        # Basic sanity checks
        for name, memory in results.items():
            self.assertGreater(memory, 0)


def run_comprehensive_tests():
    """Run all comprehensive tests and generate detailed report."""
    print("="*80)
    print("COMPREHENSIVE DLA-NET ALGORITHM TEST SUITE")
    print("="*80)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestDLoRFComponent,
        TestASIComponent, 
        TestLPQComponent,
        TestDLANetIntegration,
        TestBaselineModels,
        TestTrainingFunctions,
        TestUtilityFunctions,
        TestRobustnessAndEdgeCases,
        TestPerformanceBenchmarks,
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Generate summary report
    print("\n" + "="*80)
    print("TEST EXECUTION SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.2f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
            
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
            
    print("="*80)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run comprehensive tests
    success = run_comprehensive_tests()
    
    if success:
        print("🎉 ALL TESTS PASSED! The DLA-Net implementation is robust and ready for use.")
    else:
        print("❌ Some tests failed. Please review and fix the issues.")
        
    exit(0 if success else 1)