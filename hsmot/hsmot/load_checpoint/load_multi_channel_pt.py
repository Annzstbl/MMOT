
import torch
import torch
import torch.nn as nn
import math


def load_multi_channel_pt(path, ch_num, dst_path, version='RGBRGB'):
    '''
        读一个模型, 把第一个conv层复制
    '''

    # 假设你的pt文件存储的是一个nn.Module模型
    # 读取模型
    pt = torch.load(path)
    model = pt['model']

    # 修改第一个卷积层的输入层数
    # 找到第一个卷积层
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            assert name.startswith("model.0"), "The first convolutional layer is not the first layer of the model."
            first_conv = module
            break

    # 获取原始权重
    original_weight = first_conv.weight.data

    # 确保原始输入通道是3
    assert original_weight.shape[1] == 3, "The first convolutional layer does not have 3 input channels."

    if version == 'RGBRGB':
        # 扩展输入通道至9
        expanded_weight = original_weight.repeat(1, math.ceil(ch_num/3), 1, 1)  # 在输入通道维度上重复3次


        # 创建新的卷积层
        new_conv = nn.Conv2d(in_channels=ch_num,
                            out_channels=first_conv.out_channels,
                            kernel_size=first_conv.kernel_size,
                            stride=first_conv.stride,
                            padding=first_conv.padding,
                            dilation=first_conv.dilation,
                            groups=first_conv.groups,
                            bias=(first_conv.bias is not None))

        # 复制原始权重到新的卷积层
        new_conv.weight.data = expanded_weight[:, :ch_num, :, :]

    elif version == 'interpolate':
        assert ch_num == 8, "Only 8 channels are supported."

        R_band = 700.0
        G_band = 546.1
        B_band = 435.8
        # if output_dim == 16:
            #  bands = [465, 546, 586, 630, 474, 534, 578, 624, 485, 522, 562, 608, 496, 510, 548, 600]
        # elif output_dim == 8:
        bands = [422.5, 487.5, 550, 602.5, 660, 725, 785, 887.5]

        R_weight = original_weight[:, 0, :, :]
        G_weight = original_weight[:, 1, :, :]
        B_weight = original_weight[:, 2, :, :]

        weight_list = []
        for band in bands:
            if band <= G_band:
                weight = (B_weight * (G_band - band) + G_weight * (band - B_band)) / (G_band - B_band)
            else:
                weight = (G_weight * (R_band - band) + R_weight * (band - G_band)) / (R_band - G_band)
            weight = weight.unsqueeze(1)
            weight_list.append(weight)
        weight_concat = torch.cat(weight_list, dim=1)
        expanded_weight = weight_concat

        # 创建新的卷积层
        new_conv = nn.Conv2d(in_channels=ch_num,
                            out_channels=first_conv.out_channels,
                            kernel_size=first_conv.kernel_size,
                            stride=first_conv.stride,
                            padding=first_conv.padding,
                            dilation=first_conv.dilation,
                            groups=first_conv.groups,
                            bias=(first_conv.bias is not None))
        
        # 复制原始权重到新的卷积层
        new_conv.weight.data = expanded_weight

    # 如果原始层有bias，则复制bias
    if first_conv.bias is not None:
        new_conv.bias.data = first_conv.bias.data

    # 替换模型中的第一个卷积层
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            setattr(model, name, new_conv)
            break

    # 保存新的模型
    pt['model'] = model
    torch.save(pt, dst_path)
    return dst_path

def load_multi_channel_pt_deformableDetr(path, ch_num, dst_path, version='RGBRGB'):
    '''
        读一个模型, 把第一个conv层复制
    '''

    # 假设你的pt文件存储的是一个nn.Module模型
    # 读取模型
    pt = torch.load(path)
    model = pt['model']

    # 第一个卷积层的名字是 'backbone.0.body.conv1.weight
    original_weight = model['backbone.0.body.conv1.weight']

    # 修改第一个卷积层的输入层数

    # 获取原始权重
    # original_weight = first_conv.weight.data

    # 确保原始输入通道是3
    assert original_weight.shape[1] == 3, "The first convolutional layer does not have 3 input channels."

    if version == 'RGBRGB':
        # 扩展输入通道至9
        expanded_weight = original_weight.repeat(1, math.ceil(ch_num/3), 1, 1)  # 在输入通道维度上重复3次

        # 复制原始权重到新的卷积层
        new_conv_weight = expanded_weight[:, :ch_num, :, :]

    elif version == 'interpolate':
        assert ch_num == 8, "Only 8 channels are supported."

        R_band = 700.0
        G_band = 546.1
        B_band = 435.8
        # if output_dim == 16:
            #  bands = [465, 546, 586, 630, 474, 534, 578, 624, 485, 522, 562, 608, 496, 510, 548, 600]
        # elif output_dim == 8:
        bands = [422.5, 487.5, 550, 602.5, 660, 725, 785, 887.5]

        R_weight = original_weight[:, 0, :, :]
        G_weight = original_weight[:, 1, :, :]
        B_weight = original_weight[:, 2, :, :]

        weight_list = []
        for band in bands:
            if band <= G_band:
                weight = (B_weight * (G_band - band) + G_weight * (band - B_band)) / (G_band - B_band)
            else:
                weight = (G_weight * (R_band - band) + R_weight * (band - G_band)) / (R_band - G_band)
            weight = weight.unsqueeze(1)
            weight_list.append(weight)
        weight_concat = torch.cat(weight_list, dim=1)
        expanded_weight = weight_concat

        
        # 复制原始权重到新的卷积层
        new_conv_weight = expanded_weight

    # 替换
    model['backbone.0.body.conv1.weight'] = new_conv_weight
    # # 如果原始层有bias，则复制bias
    # if first_conv.bias is not None:
    #     new_conv.bias.data = first_conv.bias.data

    # # 替换模型中的第一个卷积层
    # for name, module in model.named_modules():
    #     if isinstance(module, nn.Conv2d):
    #         setattr(model, name, new_conv)
    #         break

    # 保存新的模型
    pt['model'] = model
    torch.save(pt, dst_path)
    return dst_path


def load_convhsi_pt_deformableDetr(path, ch_num, dst_path):

    # 假设你的pt文件存储的是一个nn.Module模型
    # 读取模型
    pt = torch.load(path)
    model = pt['model']

    # 第一个卷积层的名字是 'backbone.0.body.conv1.weight
    original_weight = model['backbone.0.body.conv1.weight']

    weight_3d = original_weight.unsqueeze(1)
    # 修改第一个卷积层的输入层数

    # 获取原始权重
    # original_weight = first_conv.weight.data

    # 确保原始输入通道是3
    assert original_weight.shape[1] == 3, "The first convolutional layer does not have 3 input channels."

    # 替换
    model['backbone.0.body.conv1.conv3d.weight'] = weight_3d
    del model['backbone.0.body.conv1.weight']
    # # 如果原始层有bias，则复制bias
    # if first_conv.bias is not None:
    #     new_conv.bias.data = first_conv.bias.data

    # # 替换模型中的第一个卷积层
    # for name, module in model.named_modules():
    #     if isinstance(module, nn.Conv2d):
    #         setattr(model, name, new_conv)
    #         break

    # 保存新的模型
    pt['model'] = model
    torch.save(pt, dst_path)
    return dst_path
#main
if __name__ == '__main__':
    # src_path = '/data3/litianhao/hsmot/motr/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth'
    # dst_path = src_path.replace('.pth', '_8ch_interpolate.pth')
    # ch_num = 8
    # version = 'interpolate'
    # load_multi_channel_pt_deformableDetr(src_path, ch_num, dst_path, version)
    # src_path = '/data3/litianhao/hsmot/memotr/dab_deformable_detr.pth'
    # dst_path = src_path.replace('.pth', '_8ch_interpolate.pth')
    # ch_num = 8
    # version = 'interpolate'
    # load_multi_channel_pt_deformableDetr(src_path, ch_num, dst_path, version)
    # print(f'save pth to {dst_path}')
    # src_path = '/data3/litianhao/hsmot/motip/r50_deformable_detr_coco.pth'
    # dst_path = src_path.replace('.pth', '_8ch_interpolate.pth')
    # ch_num = 8
    # version = 'interpolate'
    # load_multi_channel_pt_deformableDetr(src_path, ch_num, dst_path, version)
    # print(f'save pth to {dst_path}')

    # src_path = "/data3/litianhao/hsmot/motr/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth"
    # dst_path = src_path.replace('.pth', '_convhsi.pth')
    # load_convhsi_pt_deformableDetr(src_path, 8,  dst_path)
    # print(f'save pth to {dst_path}')

    src_path = "/data3/litianhao/hsmot/memotr/dab_deformable_detr.pth"
    dst_path = src_path.replace('.pth', '_convhsi.pth')
    load_convhsi_pt_deformableDetr(src_path, 8,  dst_path)
    print(f'save pth to {dst_path}')