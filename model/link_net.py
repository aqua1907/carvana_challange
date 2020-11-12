import torch
from torchvision import models
import torch.nn as nn
import pytorch_lightning as pl
from torch.nn.functional import binary_cross_entropy_with_logits as BCEWL
from pytorch_lightning import metrics


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()

        # B, C, H, W -> # B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4,
                               kernel_size=1, stride=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C/4, H*2, W*2
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4,
                                          kernel_size=3, stride=2,
                                          padding=1, output_padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        # B, C/4, H*2, W*2 -> B, C, H*2, W*2
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters,
                               kernel_size=1, stride=1, bias=False)
        self.batch_norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.relu3(x)

        return x


class LinkNet18(nn.Module):
    def __init__(self, num_classes, num_channels=3):
        super().__init__()

        resnet = models.resnet18(pretrained=True)
        filters = [64, 128, 256, 512]

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu1 = resnet.relu
        self.maxpool = resnet.maxpool

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder1 = DecoderBlock(filters[0], filters[0])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder4 = DecoderBlock(filters[3], filters[2])

        self.final_deconv1 = nn.Sequential(nn.ConvTranspose2d(filters[0], out_channels=32,
                                                              kernel_size=3, stride=2, bias=False,
                                                              padding=0, output_padding=0),
                                           nn.BatchNorm2d(32),
                                           nn.ReLU(inplace=True))

        self.final_conv1 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, bias=False),
                                         nn.BatchNorm2d(32),
                                         nn.ReLU(inplace=True))

        self.final_conv2 = nn.Conv2d(32, num_classes, kernel_size=2, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder
        d4 = self.decoder4(e4)
        d4 = d4 + e3
        d3 = self.decoder3(d4)
        d3 = d3 + e2
        d2 = self.decoder2(d3)
        d2 = d2 + e1
        d1 = self.decoder1(d2)

        # Final layers
        x = self.final_deconv1(d1)
        x = self.final_conv1(x)
        x = self.final_conv2(x)

        # # Classify
        # x = self.sigmoid(x)

        return x


class Segmentation(pl.LightningModule):
    def __init__(self, model):
        super().__init__()

        self.model = model
        self.train_acc = metrics.Accuracy(num_classes=1)
        self.val_acc = metrics.Accuracy(num_classes=1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.09, patience=3)

        return {'optimizer': optimizer,
                'scheduler': scheduler,
                'monitor': 'metric_to_track'}

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch['image'], train_batch['mask']
        logits = self.model(x)
        loss = BCEWL(logits, y, reduction='mean')
        self.train_acc(logits, y)
        # self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch['image'], val_batch['mask']
        logits = self.model(x)
        loss = BCEWL(logits, y, reduction='mean')
        self.val_acc(logits, y)
        # self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

