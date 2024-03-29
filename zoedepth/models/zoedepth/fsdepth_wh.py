# MIT License

# Copyright (c) 2022 Intelligent Systems Lab Org

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# File author: Shariq Farooq Bhat

import itertools

import torch
import torch.nn as nn
from zoedepth.models.depth_model import DepthModel
from zoedepth.models.base_models.midas import MidasCore
from zoedepth.models.layers.attractor import AttractorLayer, AttractorLayerUnnormed
from zoedepth.models.layers.dist_layers import ConditionalLogBinomial
from zoedepth.models.layers.localbins_layers import (Projector, SeedBinRegressor,
                                            SeedBinRegressorUnnormed)
from zoedepth.models.model_io import load_state_from_resource


class ZoeDepth(DepthModel):
    def __init__(self, core,  n_bins=64, bin_centers_type="softplus", bin_embedding_dim=128, min_depth=1e-3, max_depth=80,
                 n_attractors=[16, 8, 4, 1], attractor_alpha=300, attractor_gamma=2, attractor_kind='sum', attractor_type='exp', min_temp=5, max_temp=50, train_midas=True,
                 midas_lr_factor=10, encoder_lr_factor=10, pos_enc_lr_factor=10, inverse_midas=False, **kwargs):
        """ZoeDepth model. This is the version of ZoeDepth that has a single metric head

        Args:
            core (models.base_models.midas.MidasCore): The base midas model that is used for extraction of "relative" features
            n_bins (int, optional): Number of bin centers. Defaults to 64.
            bin_centers_type (str, optional): "normed" or "softplus". Activation type used for bin centers. For "normed" bin centers, linear normalization trick is applied. This results in bounded bin centers.
                                               For "softplus", softplus activation is used and thus are unbounded. Defaults to "softplus".
            bin_embedding_dim (int, optional): bin embedding dimension. Defaults to 128.
            min_depth (float, optional): Lower bound for normed bin centers. Defaults to 1e-3.
            max_depth (float, optional): Upper bound for normed bin centers. Defaults to 10.
            n_attractors (List[int], optional): Number of bin attractors at decoder layers. Defaults to [16, 8, 4, 1].
            attractor_alpha (int, optional): Proportional attractor strength. Refer to models.layers.attractor for more details. Defaults to 300.
            attractor_gamma (int, optional): Exponential attractor strength. Refer to models.layers.attractor for more details. Defaults to 2.
            attractor_kind (str, optional): Attraction aggregation "sum" or "mean". Defaults to 'sum'.
            attractor_type (str, optional): Type of attractor to use; "inv" (Inverse attractor) or "exp" (Exponential attractor). Defaults to 'exp'.
            min_temp (int, optional): Lower bound for temperature of output probability distribution. Defaults to 5.
            max_temp (int, optional): Upper bound for temperature of output probability distribution. Defaults to 50.
            train_midas (bool, optional): Whether to train "core", the base midas model. Defaults to True.
            midas_lr_factor (int, optional): Learning rate reduction factor for base midas model except its encoder and positional encodings. Defaults to 10.
            encoder_lr_factor (int, optional): Learning rate reduction factor for the encoder in midas model. Defaults to 10.
            pos_enc_lr_factor (int, optional): Learning rate reduction factor for positional encodings in the base midas model. Defaults to 10.
        """
        super().__init__()

        self.core = core
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.min_temp = min_temp
        self.bin_centers_type = bin_centers_type

        self.midas_lr_factor = midas_lr_factor
        self.encoder_lr_factor = encoder_lr_factor
        self.pos_enc_lr_factor = pos_enc_lr_factor
        self.train_midas = train_midas
        self.inverse_midas = inverse_midas

        if self.encoder_lr_factor <= 0:
            self.core.freeze_encoder(
                freeze_rel_pos=self.pos_enc_lr_factor <= 0)

        N_MIDAS_OUT = 32
        btlnck_features = self.core.output_channels[0]
        num_out_features = self.core.output_channels[1:]

        ##########cat#########
        self.conv2 = nn.Conv2d(btlnck_features + 2, btlnck_features,
                               kernel_size=1, stride=1, padding=0)  # btlnck conv

        # self.conv2 = nn.Conv2d(btlnck_features, btlnck_features,
        #                        kernel_size=1, stride=1, padding=0)  # btlnck conv

        # self.convfocal = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)  # focal conv
        #
        # self.convfocal.weight.data = torch.tensor([[[[0.1]]]])
        # self.convfocal.bias.data.zero_()
        #######cat#######
        self.focal_emb_w = nn.Parameter(torch.ones(12, 16))
        self.focal_emb_h = nn.Parameter(torch.ones(12, 16))

        #######mul#######
        # self.focal_emb = nn.Parameter(torch.ones(1))


        if bin_centers_type == "normed":
            SeedBinRegressorLayer = SeedBinRegressor
            Attractor = AttractorLayer
        elif bin_centers_type == "softplus":
            SeedBinRegressorLayer = SeedBinRegressorUnnormed
            Attractor = AttractorLayerUnnormed
        elif bin_centers_type == "hybrid1":
            SeedBinRegressorLayer = SeedBinRegressor
            Attractor = AttractorLayerUnnormed
        elif bin_centers_type == "hybrid2":
            SeedBinRegressorLayer = SeedBinRegressorUnnormed
            Attractor = AttractorLayer
        else:
            raise ValueError(
                "bin_centers_type should be one of 'normed', 'softplus', 'hybrid1', 'hybrid2'")

        self.seed_bin_regressor = SeedBinRegressorLayer(
            btlnck_features, n_bins=n_bins, min_depth=min_depth, max_depth=max_depth)
        self.seed_projector = Projector(btlnck_features, bin_embedding_dim)
        ############cat
        self.projectors = nn.ModuleList([
            Projector(num_out + 2, bin_embedding_dim)
            for num_out in num_out_features
        ])

        # self.projectors = nn.ModuleList([
        #     Projector(num_out, bin_embedding_dim)
        #     for num_out in num_out_features
        # ])

        self.attractors = nn.ModuleList([
            Attractor(bin_embedding_dim, n_bins, n_attractors=n_attractors[i], min_depth=min_depth, max_depth=max_depth,
                      alpha=attractor_alpha, gamma=attractor_gamma, kind=attractor_kind, attractor_type=attractor_type)
            for i in range(len(num_out_features))
        ])

        last_in = N_MIDAS_OUT + 1  # +1 for relative depth

        # use log binomial instead of softmax
        self.conditional_log_binomial = ConditionalLogBinomial(
            last_in, bin_embedding_dim, n_classes=n_bins, min_temp=min_temp, max_temp=max_temp)

    def forward(self, x, focal_w, focal_h, return_final_centers=False, denorm=False, return_probs=False, **kwargs):
        """
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W)
            return_final_centers (bool, optional): Whether to return the final bin centers. Defaults to False.
            denorm (bool, optional): Whether to denormalize the input image. This reverses ImageNet normalization as midas normalization is different. Defaults to False.
            return_probs (bool, optional): Whether to return the output probability distribution. Defaults to False.
        
        Returns:
            dict: Dictionary containing the following keys:
                - rel_depth (torch.Tensor): Relative depth map of shape (B, H, W)
                - metric_depth (torch.Tensor): Metric depth map of shape (B, 1, H, W)
                - bin_centers (torch.Tensor): Bin centers of shape (B, n_bins). Present only if return_final_centers is True
                - probs (torch.Tensor): Output probability distribution of shape (B, n_bins, H, W). Present only if return_probs is True

        """
        # print(x.shape)
        b, c, h, w = x.shape

        # print("x shape ", x.shape)

        self.orig_input_width = w
        self.orig_input_height = h
        rel_depth, out = self.core(x, denorm=denorm, return_rel_depth=True)

        # print("rel", rel_depth.shape)
        # focal = (focal.unsqueeze(-1)).unsqueeze(-1).unsqueeze(-1).type(torch.cuda.FloatTensor)
        focal_w = (focal_w.unsqueeze(-1)).unsqueeze(-1).unsqueeze(-1).type(torch.cuda.FloatTensor)
        focal_h = (focal_h.unsqueeze(-1)).unsqueeze(-1).unsqueeze(-1).type(torch.cuda.FloatTensor)
        # print(focal)
        # print(focal.shape)
        focal_ratio_w = (518.8579 / focal_w)
        focal_ratio_h = (518.8579 / focal_h)


        # rel_depth = rel_depth * focal_ratio

        # print("rel_depth shapes", rel_depth.shape)
        # print("output shapes", rel_depth.shape, out.shape)
        # print("output shapes", len(out))

        outconv_activation = out[0]

        btlnck = out[1]
        x_blocks = out[2:]
        # print(focal_ratio)
        # focal_emb = torch.ones(btlnck.shape[0], 1, btlnck.shape[2], btlnck.shape[3]).cuda()



        # print("focal", self.focal_emb)
        #############cat
        focal_emb_w = 100 * self.focal_emb_w * focal_ratio_w
        # print("-------------------------------------", focal_emb_w.shape)
        focal_emb_w = nn.functional.interpolate(focal_emb_w, size=btlnck.shape[2:], mode='bilinear', align_corners=True)


        focal_emb_h = 100 * self.focal_emb_h * focal_ratio_h
        focal_emb_h = nn.functional.interpolate(focal_emb_h, size=btlnck.shape[2:], mode='bilinear', align_corners=True)
        #############

        # focal_emb = self.convfocal(focal_emb)
        ###########cat
        btlnck = torch.cat([btlnck, focal_emb_w], dim=1)
        btlnck = torch.cat([btlnck, focal_emb_h], dim=1)
        ###########
        # print(btlnck.shape)

        #############mul
        # x_blocks[0] = x_blocks[0] * focal_ratio * self.focal_emb
        # x_blocks[1] = x_blocks[1] * focal_ratio * self.focal_emb
        # x_blocks[2] = x_blocks[2] * focal_ratio * self.focal_emb
        # x_blocks[3] = x_blocks[3] * focal_ratio * self.focal_emb
        #############



        x_d0 = self.conv2(btlnck)
        x = x_d0
        _, seed_b_centers = self.seed_bin_regressor(x)

        if self.bin_centers_type == 'normed' or self.bin_centers_type == 'hybrid2':
            b_prev = (seed_b_centers - self.min_depth) / \
                (self.max_depth - self.min_depth)
        else:
            b_prev = seed_b_centers

        prev_b_embedding = self.seed_projector(x)

        # unroll this loop for better performance
        for projector, attractor, x in zip(self.projectors, self.attractors, x_blocks):

            ##########cat
            focal_emb_w = nn.functional.interpolate(focal_emb_w, size=x.shape[2:], mode='bilinear', align_corners=True)
            # x = torch.cat([x, focal_emb_w], dim=1)

            focal_emb_h = nn.functional.interpolate(focal_emb_h, size=x.shape[2:], mode='bilinear', align_corners=True)
            # focal_emb = torch.cat([focal_emb_w, focal_emb_h], dim=1)
            x = torch.cat([x, focal_emb_w], dim=1)
            x = torch.cat([x, focal_emb_h], dim=1)
            ##########
            b_embedding = projector(x)
            b, b_centers = attractor(
                b_embedding, b_prev, prev_b_embedding, interpolate=True)
            b_prev = b.clone()
            prev_b_embedding = b_embedding.clone()

        last = outconv_activation
        # print("output shapes", last.shape)
        if self.inverse_midas:
            # invert depth followed by normalization
            rel_depth = 1.0 / (rel_depth + 1e-6)
            rel_depth = (rel_depth - rel_depth.min()) / \
                (rel_depth.max() - rel_depth.min())
        # concat rel depth with last. First interpolate rel depth to last size
        rel_cond = rel_depth.unsqueeze(1)
        # print("rel_cond shapes", rel_cond.shape)

        rel_cond = nn.functional.interpolate(
            rel_cond, size=last.shape[2:], mode='bilinear', align_corners=True)
        # print("rel_cond_interpolate shapes", rel_cond.shape)
        last = torch.cat([last, rel_cond], dim=1)
        # print("rel_cond_interpolate_cat_last shapes", last.shape)

        b_embedding = nn.functional.interpolate(
            b_embedding, last.shape[-2:], mode='bilinear', align_corners=True)
        # print("b_embedding shapes", b_embedding.shape)
        x = self.conditional_log_binomial(last, b_embedding)
        # print("x shapes", x.shape)

        # Now depth value is Sum px * cx , where cx are bin_centers from the last bin tensor
        # print(x.shape, b_centers.shape)
        b_centers = nn.functional.interpolate(
            b_centers, x.shape[-2:], mode='bilinear', align_corners=True)
        out = torch.sum(x * b_centers, dim=1, keepdim=True)
        # print("out shapes", out.shape)

        # Structure output dict
        output = dict(metric_depth=out)
        if return_final_centers or return_probs:
            output['bin_centers'] = b_centers

        if return_probs:
            output['probs'] = x

        return output

    def get_lr_params(self, lr):
        """
        Learning rate configuration for different layers of the model
        Args:
            lr (float) : Base learning rate
        Returns:
            list : list of parameters to optimize and their learning rates, in the format required by torch optimizers.
        """
        param_conf = []
        if self.train_midas:
            if self.encoder_lr_factor > 0:
                param_conf.append({'params': self.core.get_enc_params_except_rel_pos(
                ), 'lr': lr / self.encoder_lr_factor})

            if self.pos_enc_lr_factor > 0:
                param_conf.append(
                    {'params': self.core.get_rel_pos_params(), 'lr': lr / self.pos_enc_lr_factor})

            midas_params = self.core.core.scratch.parameters()
            midas_lr_factor = self.midas_lr_factor
            param_conf.append(
                {'params': midas_params, 'lr': lr / midas_lr_factor})

        remaining_modules = []
        for name, child in self.named_children():
            if name != 'core':
                remaining_modules.append(child)
        remaining_params = itertools.chain(
            *[child.parameters() for child in remaining_modules])

        param_conf.append({'params': remaining_params, 'lr': lr})
        param_conf.append({'params': self.focal_emb_w, 'lr': lr})
        param_conf.append({'params': self.focal_emb_h, 'lr': lr})
        return param_conf

    @staticmethod
    def build(midas_model_type="DPT_BEiT_L_384", pretrained_resource=None, use_pretrained_midas=False, train_midas=False, freeze_midas_bn=True, **kwargs):
        core = MidasCore.build(midas_model_type=midas_model_type, use_pretrained_midas=use_pretrained_midas,
                               train_midas=train_midas, fetch_features=True, freeze_bn=freeze_midas_bn, **kwargs)
        model = ZoeDepth(core, **kwargs)
        if pretrained_resource:
            assert isinstance(pretrained_resource, str), "pretrained_resource must be a string"
            model = load_state_from_resource(model, pretrained_resource)
        return model

    @staticmethod
    def build_from_config(config):
        return ZoeDepth.build(**config)
