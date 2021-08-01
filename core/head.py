from core.blocks import conv_head


def head(features):

    p5 = features[0]
    p4 = features[1]
    p3 = features[2]
  
    head_1 = conv_head(p5)
    head_2 = conv_head(p4)
    head_3 = conv_head(p3)

    return [head_3,head_2, head_1]


