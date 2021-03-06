# -*- coding: utf-8 -*-
import torch


class Module(torch.nn.Module):
    """Convenient intermediary parent class that implements useful module functions"""

    init_funcs = {
        1: lambda x: torch.nn.init.normal_(x, mean=0.0, std=1.0),  # biases
        2: lambda x: torch.nn.init.xavier_normal_(x, gain=1.0),  # weights
        3: lambda x: torch.nn.init.xavier_uniform_(x, gain=1.0),  # conv1D filters
        4: lambda x: torch.nn.init.xavier_uniform_(x, gain=1.0),  # conv2D filters
    }

    def num_params(self):
        return sum([p.numel() for p in self.parameters() if p.requires_grad])

    def reset_parameters(self):
        for p in self.parameters():
            init_func = self.init_funcs.get(
                len(p.shape), lambda x: torch.nn.init.constant(x, 1.0)
            )
            init_func(p)

    def load(self, load_path, strict=False):
        self.load_state_dict(torch.load(load_path), strict=strict)

    def save(self, save_path):
        torch.save(self.state_dict(), save_path)

    def save_jit(self, trace_image, save_path):
        traced_script_module = torch.jit.trace(self, trace_image)
        traced_script_module.save(save_path)
