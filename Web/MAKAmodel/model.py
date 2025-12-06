import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from mamba_ssm import Mamba

class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class KANLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=3,
        spline_order=2,
        scale_noise=0.05,
        scale_base=0.5,
        scale_spline=0.5,
        enable_standalone_scale_spline=False,
        base_activation=nn.SiLU,
        grid_eps=0.01,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 0.5
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = self.grid  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(
            A, B
        ).solution
        result = solution.permute(
            2, 0, 1
        )

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)
        splines = splines.permute(1, 0, 2)
        orig_coeff = self.scaled_spline_weight
        orig_coeff = orig_coeff.permute(1, 2, 0)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            ).long()
        ]
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )
        grid = torch.cat(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )
        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )
    
class kan(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., version=3):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.dwconv_1 = DW_bn_relu(dim=in_features)

        self.fc1 = KANLinear(
            in_features,
            out_features,
            grid_size=3,
            spline_order=2,
            scale_noise=0.05,
            scale_base=0.5,
            scale_spline=0.5,
            enable_standalone_scale_spline=False,
            base_activation=nn.SiLU,
            grid_eps=0.01,
            grid_range=[-1, 1],
        )
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.dwconv_1(x, H, W)
        x = self.fc1(x.reshape(B*N, C))
        x = x.reshape(B, N, -1).contiguous()
        x = self.drop(x)
        return x    
    
class MambaLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2): 
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim,  
            d_state=d_state,    
            d_conv=d_conv,      
            expand=expand,      
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm) + self.skip_scale * x_flat
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out

class Residual1(nn.Module):
    def __init__(self, input_channels, stages, features_per_stage, conv_op, kernel_sizes, strides, n_blocks_per_stage):
        super().__init__()
        self.blocks = nn.ModuleList()
        input_dim = input_channels
        self.fixed_arch_depth = 64
        self.fixed_filters = 64
        d_rate = 1
        for stage in range(self.fixed_arch_depth):
            block = nn.ModuleList()
            current_out_dim = self.fixed_filters
            for _ in range(n_blocks_per_stage):
                if stage % 2 == 0:
                    conv_kernel = (1, 5)
                else:
                    conv_kernel = (3, 3)
                conv = nn.Conv2d(
                    input_dim, current_out_dim,
                    kernel_size=conv_kernel, 
                    padding='same',
                    dilation=d_rate,
                    bias=False
                )
                nn.init.kaiming_normal_(conv.weight, mode='fan_in', nonlinearity='relu')
                block.extend([
                    conv,
                    nn.BatchNorm2d(current_out_dim),
                    nn.ELU(),
                    MambaLayer(
                        input_dim=current_out_dim,
                        output_dim=current_out_dim,
                    )
                ])
                input_dim = current_out_dim
                if d_rate == 1:
                    d_rate = 2
                elif d_rate == 2:
                    d_rate = 4
                else:
                    d_rate = 1
            self.blocks.append(block)
        self.adjust_channels = nn.ModuleList([
            nn.Conv2d(input_channels, self.fixed_filters, kernel_size=1)
        ] + [
            nn.Conv2d(self.fixed_filters, self.fixed_filters, kernel_size=1)
            for _ in range(1, self.fixed_arch_depth)
        ])
    def forward(self, x):
        skips = []
        for idx, block in enumerate(self.blocks):
            block_input = x
            for layer in block:
                x = layer(x)
            if block_input.shape[1] != x.shape[1]:
                block_input = self.adjust_channels[idx](block_input)
            if x.shape[2:] != block_input.shape[2:]:
                x = F.interpolate(x, size=block_input.shape[2:], mode='nearest')
            skips.append(x)
            x = x + block_input
        return skips
    
class Res2(nn.Module):
    def __init__(self, encoder, num_classes, n_conv_per_stage):
        super().__init__()
        self.stages = len(encoder.blocks)
        self.conv_layers = nn.ModuleList()
        def depthwise_conv(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim),
                nn.Conv2d(in_dim, out_dim, kernel_size=1),
                nn.BatchNorm2d(out_dim),
                nn.ELU()
            )
        for i in range(self.stages - 1, 0, -1):
            current_out_dim = encoder.blocks[i][-1].output_dim
            prev_out_dim = encoder.blocks[i-1][-1].output_dim
            self.conv_layers.append(nn.ConvTranspose2d(current_out_dim, prev_out_dim, kernel_size=2, stride=2))
            self.conv_layers.append(depthwise_conv(prev_out_dim * 2, prev_out_dim))
            self.conv_layers.append(nn.Dropout(0.2))
        self.final_conv = nn.Conv2d(encoder.blocks[0][-1].output_dim, num_classes, kernel_size=1)
        self.softplus = nn.Softplus()
    def forward(self, skips):
        x = skips[-1]
        for i in range(0, len(self.conv_layers), 3):
            x = self.conv_layers[i](x)
            target_skip_idx = len(skips) - (i // 3) - 2
            if x.shape[2:] != skips[target_skip_idx].shape[2:]:
                x = F.interpolate(x, size=skips[target_skip_idx].shape[2:], mode='nearest')
            x = torch.cat((x, skips[target_skip_idx]), dim=1)
            x = self.conv_layers[i + 1](x)
            x = self.conv_layers[i + 2](x)
        x = self.final_conv(x)
        x = self.softplus(x)
        return x

class Middle(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, version=3):
        super().__init__()
        self.kan = kan(in_features=in_features, hidden_features=hidden_features, out_features=out_features, version=version)

    def forward(self, x, H, W):
        B, C, H, W = x.shape
        N = H * W
        x = x.view(B, C, N).transpose(1, 2)
        x = self.kan(x, H, W)
        C_out = x.shape[2]
        x = x.transpose(1, 2).view(B, C_out, H, W)
        return x

class Resmambakan(nn.Module):
    def __init__(self, expected_n_channels, kan_version=3):
        super().__init__()
        self.res1 = Residual1(
            input_channels=expected_n_channels,
            stages=64,
            features_per_stage=[64] * 8,
            conv_op=nn.Conv2d,
            kernel_sizes=3,
            strides=1,
            n_blocks_per_stage=1
        )
        res1_output_dim = self.res1.fixed_filters
        self.middle = Middle(
            in_features=res1_output_dim,  # 64
            hidden_features=res1_output_dim,
            out_features=res1_output_dim,
            version=kan_version
        )
        self.res2 = Res2(self.res1, num_classes=1, n_conv_per_stage=2)
    def forward(self, x):
        skips = self.res1(x)
        middle_input = skips[-1]
        middle_output = self.middle(middle_input, middle_input.shape[2], middle_input.shape[3])  # [B, C_out, H, W]
        skips[-1] = middle_output
        out = self.res2(skips)
        return out