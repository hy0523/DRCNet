r""" Extracts intermediate features from given backbone network & layer ids """
from IPython import embed

def extract_feat_vgg(img, backbone, feat_ids, bottleneck_ids=None, lids=None):
    r""" Extract intermediate features from VGG """
    feats = []
    feat = img
    for lid, module in enumerate(backbone.features):
        feat = module(feat)
        if lid in feat_ids:
            feats.append(feat.clone())
    return feats


def extract_feat_res(img, backbone, feat_ids, bottleneck_ids, lids):
    r""" Extract intermediate features from ResNet"""
    feats = []

    # Layer 0
    feat = backbone[0].forward(img)

    # Layer 1-4
    for hid, (bid, lid) in enumerate(zip(bottleneck_ids, lids)):
        res = feat
        feat = backbone[lid][bid].conv1.forward(feat)
        feat = backbone[lid][bid].bn1.forward(feat)
        feat = backbone[lid][bid].relu.forward(feat)
        feat = backbone[lid][bid].conv2.forward(feat)
        feat = backbone[lid][bid].bn2.forward(feat)
        feat = backbone[lid][bid].relu.forward(feat)
        feat = backbone[lid][bid].conv3.forward(feat)
        feat = backbone[lid][bid].bn3.forward(feat)

        if bid == 0:
            res = backbone[lid][bid].downsample.forward(res)

        feat += res

        if hid + 1 in feat_ids:
            feats.append(feat.clone())

        feat = backbone[lid][bid].relu.forward(feat)

    return feats