import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from Layer import Residual_Block, RD_Down_layer, RD_Up_layer, conv_block


class RDR_model(nn.Module):
    def __init__(self, in_channel=1, out_channel=8, num_dense_layer=3, growth_rate=36):
        super(RDR_model, self).__init__()

        # in_feature = [1024, 1024, 768, 384, 192, 96, 48]
        # out_feature = [512, 512, 256, 128, 64, 32, 16]
        in_feature = [512, 512, 384, 192, 96, 32]
        out_feature = [256, 256, 128, 64, 32, 16]

        self.RB1 = Residual_Block(in_channels=in_channel, out_channels=out_channel)
        self.RDDB1 = RD_Down_layer(in_channels=out_channel, out_channels=out_channel * 4,
                                   num_dense_layer=num_dense_layer,
                                   growth_rate=growth_rate)
        self.RDDB2 = RD_Down_layer(in_channels=out_channel * 4, out_channels=out_channel * 8,
                                   num_dense_layer=num_dense_layer,
                                   growth_rate=growth_rate)
        self.RDDB3 = RD_Down_layer(in_channels=out_channel * 8, out_channels=out_channel * 16,
                                   num_dense_layer=num_dense_layer,
                                   growth_rate=growth_rate)
        self.RDDB4 = RD_Down_layer(in_channels=out_channel * 16, out_channels=out_channel * 32,
                                   num_dense_layer=num_dense_layer,
                                   growth_rate=growth_rate)
        self.RDDB5 = RD_Down_layer(in_channels=out_channel * 32, out_channels=out_channel * 64,
                                   num_dense_layer=num_dense_layer,
                                   growth_rate=growth_rate)
        # self.RDDB6 = RD_Down_layer(in_channels=out_channel * 32, num_dense_layer=num_dense_layer,
        #                            growth_rate=growth_rate)
        # self.RDDB7 = RD_Down_layer(in_channels=out_channel * 64, num_dense_layer=num_dense_layer,  模型简化
        #                            growth_rate=growth_rate)
        self.RDUB1 = RD_Up_layer(in_channels=in_feature[0], out_channels=out_feature[0],
                                 num_dense_layer=num_dense_layer, growth_rate=growth_rate)
        self.RDUB2 = RD_Up_layer(in_channels=in_feature[1], out_channels=out_feature[1],
                                 num_dense_layer=num_dense_layer, growth_rate=growth_rate)
        self.RDUB3 = RD_Up_layer(in_channels=in_feature[2], out_channels=out_feature[2],
                                 num_dense_layer=num_dense_layer, growth_rate=growth_rate)
        self.RDUB4 = RD_Up_layer(in_channels=in_feature[3], out_channels=out_feature[3],
                                 num_dense_layer=num_dense_layer, growth_rate=growth_rate)
        self.RDUB5 = RD_Up_layer(in_channels=in_feature[4], out_channels=out_feature[4],
                                 num_dense_layer=num_dense_layer, growth_rate=growth_rate)
        # self.RDUB6 = RD_Up_layer(in_channels=in_feature[5], out_channels=out_feature[5],
        #                          num_dense_layer=num_dense_layer, growth_rate=growth_rate)
        self.RB2 = Residual_Block(in_channels=in_feature[5], out_channels=out_feature[5])
        self.conv1 = conv_block(32, 16)
        self.conv2 = conv_block(16, 1)

    def forward(self, inputs):
        # input size = 256*256*1
        # pre-processing
        RB1_Layer = self.RB1(inputs)  # 256*256*8
        # Down-sampling
        RDDB1_Layer = self.RDDB1(RB1_Layer)  # 128*128*32
        RDDB2_Layer = self.RDDB2(RDDB1_Layer)  # 64*64*64
        RDDB3_Layer = self.RDDB3(RDDB2_Layer)  # 32*32*128
        RDDB4_Layer = self.RDDB4(RDDB3_Layer)  # 16*316*256
        RDDB5_Layer = self.RDDB5(RDDB4_Layer)  # 8*8*512
        # RDDB6_Layer = self.RDDB6(RDDB5_Layer)  # 8*8*512
        # RDDB7_Layer = self.RDDB7(RDDB6_Layer)  # 8*8*1024
        # Up-sampling and concatenation
        RDUB1_Layer_ = self.RDUB1(RDDB5_Layer)  # 16*16*256
        RDUB1_Layer = torch.cat((RDDB4_Layer, RDUB1_Layer_), 1)  # 16*16*512
        RDUB2_Layer_ = self.RDUB2(RDUB1_Layer)  # 32*32*256
        RDUB2_Layer = torch.cat((RDDB3_Layer, RDUB2_Layer_), 1)  # 32*32*384
        RDUB3_Layer_ = self.RDUB3(RDUB2_Layer)  # 64*64*128
        RDUB3_Layer = torch.cat((RDDB2_Layer, RDUB3_Layer_), 1)  # 64*64*192
        RDUB4_Layer_ = self.RDUB4(RDUB3_Layer)  # 128*128*64
        RDUB4_Layer = torch.cat((RDDB1_Layer, RDUB4_Layer_), 1)  # 128*128*96
        RDUB5_Layer_ = self.RDUB5(RDUB4_Layer)  # 256*256*32
        # RDUB5_Layer = torch.cat((RB1_Layer, RDUB5_Layer_), 1)  # 256*256*40
        # RDUB6_Layer_ = self.RDUB6(RDUB5_Layer)  # 512*512*32
        # RDUB6_Layer = torch.cat((RB1_Layer, RDUB6_Layer_), 1)  # 512*512*48
        RB2_layer = self.conv1(RDUB5_Layer_)
        outputs = self.conv2(RB2_layer)
        return outputs
        # RDUB1_Layer_ = self.RDUB1(RDDB7_Layer)  # 16*16*512
        # RDUB1_Layer = torch.cat((RDDB6_Layer, RDUB1_Layer_), 1)  # 16*16*1024
        # RDUB2_Layer_ = self.RDUB2(RDUB1_Layer)  # 32*32*512
        # RDUB2_Layer = torch.cat((RDDB5_Layer, RDUB2_Layer_), 1)  # 32*32*768
        # RDUB3_Layer_ = self.RDUB3(RDUB2_Layer)  # 64*64*256
        # RDUB3_Layer = torch.cat((RDDB4_Layer, RDUB3_Layer_), 1)  # 64*64*384
        # RDUB4_Layer_ = self.RDUB4(RDUB3_Layer)  # 128*128*128
        # RDUB4_Layer = torch.cat((RDDB3_Layer, RDUB4_Layer_), 1)  # 128*128*192
        # RDUB5_Layer_ = self.RDUB5(RDUB4_Layer)  # 256*256*64
        # RDUB5_Layer = torch.cat((RDDB2_Layer, RDUB5_Layer_), 1)  # 256*256*96
        # RDUB6_Layer_ = self.RDUB6(RDUB5_Layer)  # 512*512*32
        # RDUB6_Layer = torch.cat((RDDB1_Layer, RDUB6_Layer_), 1)  # 512*512*48
        # RDR_Layer = self.RDR(RDUB6_Layer)
        # outputs = self.RB2(RDR_Layer)
        # return outputs

    # def _initialize_weights(self, *stages):
    #     for modules in stages:
    #         for module in modules.modules():
    #             if isinstance(module, nn.Conv2d):
    #                 nn.init.kaiming_normal_(module.weight)
    #                 if module.bias is not None:
    #                     module.bias.data.zero_()
    #             elif isinstance(module, nn.BatchNorm2d):
    #                 module.weight.data.fill_(1)
    #                 module.bias.data.zero_()


#
Model = RDR_model()
# print(Model)
torch.save(Model,'m.onnx')

# inputs = torch.ones((8, 1, 1024, 1024))
# writer = SummaryWriter("graph")
# writer.add_graph(Model, inputs)
# writer.close()
