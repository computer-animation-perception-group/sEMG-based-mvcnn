from __future__ import division
import mxnet as mx
from nose.tools import assert_equal
from . import constant


class GRL(mx.operator.CustomOp):

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], 0 + in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(
            in_grad[0],
            req[0],
            -aux[0] * out_grad[0]
        )


@mx.operator.register('GRL')
class GRLProp(mx.operator.CustomOpProp):

    def __init__(self):
        super(GRLProp, self).__init__(need_top_grad=True)

    def list_auxiliary_states(self):
        return ['lambda']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        output_shape = in_shape[0]
        return [data_shape], [output_shape], [(1,)]

    def create_operator(self, ctx, shapes, dtypes):
        return GRL()


class GradScale(mx.operator.CustomOp):

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], 0 + in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(
            in_grad[0],
            req[0],
            aux[0] * out_grad[0]
        )


@mx.operator.register('GradScale')
class GradScaleProp(mx.operator.CustomOpProp):

    def __init__(self):
        super(GradScaleProp, self).__init__(need_top_grad=True)

    def list_auxiliary_states(self):
        return ['scale']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        output_shape = in_shape[0]
        return [data_shape], [output_shape], [(1,)]

    def create_operator(self, ctx, shapes, dtypes):
        return GradScale()


class Symbol(object):

    def get_bn_orig(self, name, data):
        return mx.symbol.BatchNorm(
            name=name,
            data=data,
            fix_gamma=True,
            momentum=0.9
        )

    def infer_shape(self, data):
        net = data
        if self.num_stream == 1:
            data_shape = (10 if self.minibatch else 1,
                          self.num_channel, self.num_semg_row, self.num_semg_col)
            shape = tuple(int(s) for s in net.infer_shape(data=data_shape)[1][0])
        else:
            shape = tuple(int(s) for s in net.infer_shape(
                **{'stream%d_data' % i: (10 if self.minibatch else 1,
                                         self.num_channel[i], self.num_semg_row[i], self.num_semg_col[i])
                   for i in range(self.num_stream) if 'stream%d_data_tag' % i in net.list_attr(recursive=True)})[1][0])
        return shape

    def get_bn(self, name, data):
        if not self.bng:
            if self.minibatch:
                net = data
                if self.num_stream == 1:
                    data_shape = (10, self.num_channel, self.num_semg_row, self.num_semg_col)
                    shape = tuple(int(s) for s in net.infer_shape(data=data_shape)[1][0])
                else:
                    shape = tuple(int(s) for s in net.infer_shape(
                        **{'stream%d_data' % i: (10, self.num_channel[i], self.num_semg_row[i], self.num_semg_col[i])
                           for i in range(self.num_stream) if 'stream%d_data_tag' % i in net.list_attr(recursive=True)})[1][0])
                net = mx.symbol.Reshape(net, shape=(-1, 10 * shape[1]) + shape[2:])
                net = mx.symbol.BatchNorm(
                    name=name + '_norm',
                    data=net,
                    fix_gamma=True,
                    momentum=0.9,
                    attr={'wd_mult': '0', 'lr_mult': '0'})
                net = mx.symbol.Reshape(data=net, shape=(-1,) + shape[1:])
                if len(shape) == 4:
                    gamma = mx.symbol.Variable(name + '_gamma', shape=(1, shape[1], 1, 1))
                    beta = mx.symbol.Variable(name + '_beta', shape=(1, shape[1], 1, 1))
                else:
                    gamma = mx.symbol.Variable(name + '_gamma', shape=(1, shape[1]))
                    beta = mx.symbol.Variable(name + '_beta', shape=(1, shape[1]))
                net = mx.symbol.broadcast_mul(net, gamma)
                net = mx.symbol.broadcast_plus(net, beta, name=name + '_last')
                return net
            else:
                net = mx.symbol.BatchNorm(
                    name=name,
                    data=data,
                    fix_gamma=False,
                    momentum=0.9)
        else:
            net = self.get_bng(name, data)
        return net

    def get_bng(self, name, data):
        net = data
        if self.num_stream == 1:
            data_shape = (1, self.num_channel, self.num_semg_row, self.num_semg_col)
            shape = tuple(int(s) for s in net.infer_shape(data=data_shape)[1][0])
        else:
            shape = tuple(int(s) for s in net.infer_shape(
                **{'stream%d_data' % i: (self.num_subject, self.num_channel[i], self.num_semg_row[i], self.num_semg_col[i])
                   for i in range(self.num_stream) if 'stream%d_data_tag' % i in net.list_attr(recursive=True)})[1][0])
        if shape[1] > 1:
            net = mx.symbol.Reshape(net, shape=(0, 1, -1))
            net = mx.symbol.BatchNorm(
                name=name,
                data=net,
                fix_gamma=False,
                momentum=0.9)
        if shape[1] > 1:
            net = mx.symbol.Reshape(name=name + '_restore', data=net, shape=(-1,) + shape[1:])

    def get_bn_relu(self, name, data):
        net = data
        net = self.get_bn(name + '_bn', net)
        net = mx.symbol.Activation(name=name + '_relu', data=net, act_type='relu')
        return net

    def get_bng_relu(self, name, data):
        net = data

        net_shape = tuple(int(s) for s in net.infer_shape(
            **{'stream%d_data' % i: (self.num_subject, self.num_channel[i], self.num_semg_row[i], self.num_semg_col[i])
               for i in range(self.num_stream) if 'stream%d_data_tag' % i in net.list_attr(recursive=True)})[1][0])

        net = mx.symbol.Reshape(net, shape=(0, net_shape[1], -1))
        net = self.get_bn(name + '_bn', net)
        net = mx.symbol.Reshape(net, shape=(0, net_shape[1], net_shape[2], net_shape[3]))

        net = mx.symbol.Activation(name=name + '_relu', data=net, act_type='relu')
        return net

    def get_pixel_reduce(self, name, net, num_filter, no_bias, rows, cols):
        net = mx.symbol.Reshape(net, shape=(0, 0, -1))

        nets = mx.symbol.SliceChannel(net, num_outputs=rows * cols, axis=2)
        nets = [self.get_fc(name + '_fc%d' % i, nets[i], num_filter, no_bias) for i in range(rows * cols)]
        nets = [mx.symbol.Reshape(p, shape=(0, 0, 1)) for p in nets]
        net = mx.symbol.Concat(*net, dim=2)

        net = mx.symbol.Reshape(net, shape=(0, 0, rows, cols))
        return net

    def im2col(self, data, name, kernel, pad=(0, 0), stride=(1, 1)):
        shape = self.infer_shape(data)
        return mx.symbol.Convolution(
            name=name,
            data=data,
            num_filter=shape[1] * kernel[0] * kernel[1],
            kernel=kernel,
            stride=stride,
            pad=pad,
            no_bias=True,
            attr={'lr_mult': '0'})

    def get_smooth_pixel_reduce(self, name, net, num_filter, no_bias, rows, cols, kernel=1, stride=1, pad=0):
        if kernel != 1:
            net = self.im2col(name=name + '_im2col', data=net,
                              kernel=(kernel, kernel),
                              pad=(pad, pad),
                              stride=(stride, stride))
            return self.get_smooth_pixel_reduce(name, net, num_filter, no_bias, rows, cols)

        net = mx.symbol.Reshape(net, shape=(0, 0, -1))

        nets = mx.symbol.SliceChannel(net, num_outputs=rows * cols, axis=2)
        W = [[mx.symbol.Variable(name=name + '_fc%d_weight' % (row * cols + col))
              for col in range(cols)] for row in range(rows)]
        nets = [mx.symbol.FullyConnected(name=name + '_fc%d' % i,
                                         data=nets[i],
                                         num_hidden=num_filter,
                                         no_bias=no_bias,
                                         weight=W[i // cols][i % cols])
                for i in range(rows * cols)]
        nets = [mx.symbol.Reshape(p, shape=(0, 0, 1)) for p in nets]
        net = mx.symbol.Concat(*nets, dim=2)
        net = mx.symbol.Reshape(net, shape=(0, 0, rows, cols))

        if self.fast_pixel_reduce:
            lhs, rhs = [], []
            for rs in range(rows):
                for cs in range(cols):
                    for ro, co in [(1, 0), (0, 1)]:
                        rt = rs + ro
                        ct = cs + co
                        if rt < rows and ct < cols:
                            lhs.append(W[rs][cs])
                            rhs.append(W[rt][ct])
            lhs = mx.symbol.Concat(*lhs, dim=0)
            rhs = mx.symbol.Concat(*rhs, dim=0)
            if self.pixel_reduce_norm:
                lhs = mx.symbol.L2Normalization(lhs)
                rhs = mx.symbol.L2Normalization(rhs)
            diff = lhs - rhs
            if self.pixel_reduce_reg_out:
                diff = mx.symbol.sum(diff, axis=1)
            R = mx.symbol.sum(mx.symbol.square(diff))
        else:
            R = []
            for rs in range(rows):
                for cs in range(cols):
                    for ro, co in [(1, 0), (0, 1)]:
                        rt = rs + ro
                        ct = cs + co
                        if rt < rows and ct < cols:
                            R.append(mx.symbol.sum(mx.symbol.square(W[rs][cs] - W[rt][ct])))
            R = mx.symbol.ElementWiseSum(*R)
        loss = mx.symbol.MakeLoss(data=R, grad_scale=self.pixel_reduce_loss_weight)

        return net, loss

    def get_feature(
        self,
        data,
        num_filter,
        num_pixel,
        num_block,
        num_hidden,
        num_bottleneck,
        dropout,
        semg_row,
        semg_col,
        prefix):
        get_act = self.get_bn_relu

        net = data

        out = {}
        if not self.num_presnet:
            if not self.pool:
                for i in range(self.num_conv):
                    name = prefix + 'conv%d' % (i + 1)
                    net = Convolution(
                        name=name,
                        data=net,
                        num_filter=num_filter,
                        kernel=(3, 3),
                        stride=(1, 1),
                        pad=(1, 1),
                        no_bias=self.no_bias
                    )
                    net = get_act(name, net)
                    out[name] = net
            else:
                for i in range(4):
                    name = prefix + 'conv%d' % (i + 1)
                    net = Convolution(
                        name=name,
                        data=net,
                        num_filter=num_filter,
                        kernel=(3, 3),
                        stride=(1, 1),
                        pad=(1, 1),
                        no_bias=self.no_bias)
                    net = get_act(name, net)
                    out[name] = net
                    net = mx.symbol.Pooling(
                        name=prefix + 'pool%d' % (i + 1),
                        data=net,
                        kernel=(3, 3),
                        stride=(1, 1),
                        pad=(1, 1),
                        pool_type='max')

            if self.drop_conv:
                net = mx.symbol.Dropout(name=prefix + 'conv_drop', data=net, p=dropout)
        conv = net

        def get_conv(name, net, k3, num_filter, stride):
            return Convolution(
                name=name,
                data=net,
                num_filter=num_filter // 4,
                kernel=(3, 3) if k3 else (1, 1),
                stride=(stride, stride),
                pad=(1, 1) if k3 else (0, 0),
                no_bias=self.no_bias)

        def get_branches(net, block, first_act, num_filter, rows, cols, stride):
            act = get_act(prefix + 'block%d_branch1_conv1' % (block + 1), net) if first_act else net
            b1 = get_conv(prefix + 'block%d_branch1_conv1' % (block + 1), act, False, num_filter, 1)

            b2 = get_act(prefix + 'block%d_branch2_conv2' % (block + 1), b1)
            b2 = get_conv(prefix + 'block%d_branch2_conv2' % (block + 1), b2, True, num_filter, stride)

            b3 = get_act(prefix + 'block%d_branch3_conv3' % (block + 1), b2)
            b3 = get_conv(prefix + 'block%d_branch3_conv3' % (block + 1), b3, True, num_filter, 1)

            return b1, b2, b3


        rows = semg_row
        cols = semg_col
        num_local = num_filter

        if self.num_presnet:
            net = Convolution(
                name=prefix + 'stem',
                data=net,
                num_filter=num_filter,
                kernel=(3, 3),
                stride=(1, 1),
                pad=(1, 1),
                no_bias=False)
            if self.drop_conv:
                net = mx.symbol.Dropout(name=prefix + 'stem_drop', data=net, p=dropout)

            if isinstance(self.num_presnet, (tuple, list)):
                num_presnet = self.num_presnet
            else:
                num_presnet = [self.num_presnet]
            block = 0
            shortcuts = []
            for level in range(len(num_presnet)):
                for i in range(num_presnet[level]):
                    if i == 0:
                        net = get_act(prefix + 'block%d_pre' % (block + 1), net)
                    shortcut = net

                    if level > 0 and i == 0:
                        stride = 2
                    else:
                        stride = 1

                    num_local = num_filter * 2 ** level
                    if self.presnet_promote:
                        num_local *= 4

                    if (level > 0 or self.presnet_promote) and i == 0:
                        if self.presnet_proj_type == 'A':
                            assert False
                        elif self.presnet_proj_type == 'B':
                            shortcut = Convolution(
                                name=prefix + 'block%d_proj' % (block + 1),
                                data=shortcut,
                                num_filter=num_local,
                                kernel=(1, 1),
                                stride=(stride, stride),
                                pad=(0, 0),
                                no_bias=False)
                        else:
                            assert False
                        if self.drop_presnet_proj:
                            shortcut = mx.symbol.Dropout(name=prefix + 'block%d_proj_drop' % (block + 1),
                                                         data=shortcut, p=dropout)
                    rows = semg_row // 2 ** level
                    cols = semg_col // 2 ** level
                    branches = get_branches(net, block, i > 0, num_local, rows, cols, stride)
                    if self.presnet_branch:
                        branches = [branches[i] for i in self.presnet_branch]
                    net = mx.symbol.Concat(*branches) if len(branches) > 1 else branches[0]
                    net = get_act(prefix + 'block%d_expand' % (block + 1), net)
                    net = Convolution(
                        name=prefix + 'block%d_expand' % (block + 1),
                        data=net,
                        num_filter=num_filter,
                        kernel=(1, 1),
                        stride=(1, 1),
                        pad=(0, 0),
                        no_bias=False)
                    if self.drop_presnet_branch:
                        net = mx.symbol.Dropout(name=prefix + 'block%d_drop' % (block + 1),
                                                data=net, p=dropout)
                    shortcuts.append(shortcut)
                    if not self.presnet_dense or block == 0:
                        net = shortcut + 0.1 * net
                    else:
                        W = [mx.symbol.Variable(name=prefix + 'block%d_dense%d_zero' % (block + 1, i + 1),
                                                shape=(1, 1, 1, 1),
                                                attr={'wd_mult': '0'}) for i in range(block)]
                        W.append(mx.symbol.Variable(name=prefix + 'block%d_dense%d_one' % (block + 1, block + 1),
                                                    shape=(1, 1, 1, 1),
                                                    attr={'wd_mult': '0'}))
                        assert len(W) == len(shortcuts)
                        dense = mx.symbol.ElementWiseSum(*[mx.symbol.broadcast_mul(mx.symbol.broadcast_to(w, shape=(1, num_local, 1, 1)), s)
                                                           for w, s in zip(W, shortcuts)])
                        net = dense + 0.1 * net
                    block += 1

            net = get_act(prefix + 'presnet', net)
            if self.drop_presnet:
                net = mx.symbol.Dropout(name=prefix + 'presnet_drop', data=net, p=dropout)

        loss = []
        if num_pixel:
            for i in range(num_pixel):
                name = prefix + ('pixel%d' % (i + 1) if num_pixel > 1 else 'pixel')
                rows //= self.pixel_reduce_stride[i]
                cols //= self.pixel_reduce_stride[i]
                ret = self.get_smooth_pixel_reduce(name, net,
                                                   self.num_pixel_reduce_filter[i] or num_local,
                                                   no_bias=not self.pixel_reduce_bias,
                                                   rows=rows, cols=cols,
                                                   kernel=self.pixel_reduce_kernel[i],
                                                   stride=self.pixel_reduce_stride[i],
                                                   pad=self.pixel_reduce_pad[i])
                net = ret[0]
                if self.pixel_reduce_loss_weight > 0:
                    loss.append(ret[1])
                net = get_act(name, net)
                if i in self.drop_pixel:
                    net = Dropout(name=name + '_drop', data=net, p=dropout)
                out[name] = net
            if tuple(self.drop_pixel) == (-1,):
                net = Dropout(name=prefix + 'pixel_drop', data=net, p=dropout)
            if self.conv_shortcut:
                net = mx.symbol.Concat(mx.symbol.Flatten(conv), mx.symbol.Flatten(net), dim=1)
        out['loss'] = loss

        for i in range(1):
            name = prefix + 'fc%d' % (i + 1)
            net = self.get_fc(name, net, num_hidden, no_bias=self.no_bias)
            net = get_act(name, net)
            net = Dropout(
                name=name + '_drop',
                data=net,
                p=dropout
            )
            out[name] = net

        net = self.get_fc(prefix + 'bottleneck', net, num_hidden, no_bias=self.no_bias)
        net = get_act(prefix + 'bottleneck', net)
        out[prefix + 'bottleneck'] = net

        return out

    def get_fc(self, name, data, num_hidden, no_bias=False):
        return mx.symbol.FullyConnected(
            name=name,
            data=data,
            num_hidden=num_hidden,
            no_bias=no_bias)

    def get_2layer_lc(self, prefix, data):
        net = data
        test_shape = self.infer_shape(net)

        rows = test_shape[2]
        cols = test_shape[3]

        num_filter = 64
        name = prefix + 'conv2'

        net = Convolution(
            name=name,
            data=net,
            num_filter=num_filter,
            kernel=(3, 3),
            stride=(1, 1),
            pad=(1, 1),
            no_bias=self.no_bias)

        num_local = 64

        loss = []
        name = prefix + 'pixel1'
        rows //= self.pixel_reduce_stride[0]
        cols //= self.pixel_reduce_stride[0]
        ret = self.get_smooth_pixel_reduce(name, net,
                                           num_local,
                                           no_bias=not self.pixel_reduce_bias,
                                           rows=rows, cols=cols,
                                           kernel=self.pixel_reduce_kernel[0],
                                           stride=self.pixel_reduce_stride[0],
                                           pad=self.pixel_reduce_pad[0])
        net=ret[0]
        if self.pixel_reduce_loss_weight > 0:
            loss.append(ret[1])
        net = self.get_bn_relu(name, net)

        name = prefix + 'pixel2'
        rows //= self.pixel_reduce_stride[0]
        cols //= self.pixel_reduce_stride[0]
        ret = self.get_smooth_pixel_reduce(name, net,
                                           num_local,
                                           no_bias=not self.pixel_reduce_bias,
                                           rows=rows, cols=cols,
                                           kernel=self.pixel_reduce_kernel[0],
                                           stride=self.pixel_reduce_stride[0],
                                           pad=self.pixel_reduce_pad[0])
        net=ret[0]
        if self.pixel_reduce_loss_weight > 0:
            loss.append(ret[1])
        net = self.get_bn_relu(name, net)

        net = Dropout(name=name + '_drop', data=net, p=0.5)
        return net

    def get_branch(
        self,
        name,
        data,
        num_class,
        num_block,
        num_hidden,
        return_fc=False,
        fc_attr={},
        **kargs):
        net = data
        if num_block and num_hidden:
            for i in range(num_block):
                net = mx.symbol.FullyConnected(
                    name=name + '_fc%d' % (i + 1),
                    data=net,
                    num_hidden=num_hidden,
                    no_bias=self.no_bias,
                    attr=dict(**fc_attr))
                net = self.get_bn_relu(name + '_fc%d' % (i + 1), net)
                if self.drop_branch:
                    net = Dropout(
                        name=name + '_fc%d_drop' % (i + 1),
                        data=net,
                        p=0.5)

        net = mx.symbol.FullyConnected(
            name=name + '_last_fc',
            data=net,
            num_hidden=num_class,
            no_bias=False,
            attr=dict(**fc_attr))
        fc = net
        if self.tzeng:
            net = mx.symbol.Custom(data=net, name=name + '_gradscale', op_type='GradScale')
        if self.for_training:
            net = mx.symbol.SoftmaxOutput(name=name + '_softmax', data=net)
        else:
            net = mx.symbol.SoftmaxActivation(name=name + '_softmax', data=net)
        return (net, fc) if return_fc else net


    def get_multistream_feature(
        self,
        data,
        num_filter,
        num_pixel,
        num_block,
        num_hidden,
        num_bottleneck,
        dropout,
        semg_row,
        semg_col,
        prefix,
        weight=None):

        #sigimg_streams = ['stream%d_' % i for i in range(3)]

        get_act = self.get_bn_relu
        num_filter = 64
        net = data

        out = {}

        name = prefix + 'conv1'

        if weight is not None:
            print('Use shared conv weight!')
            net = Convolution(
                name=name,
                data=net,
                num_filter=num_filter,
                kernel=(3, 3),
                stride=(1, 1),
                pad=(1, 1),
                no_bias=self.no_bias,
                weight=weight[0])
        else:
            net = Convolution(
                name=name,
                data=net,
                num_filter=num_filter,
                kernel=(3, 3),
                stride=(1, 1),
                pad=(1, 1),
                no_bias=self.no_bias)

        net = get_act(name, net)

        out[name] = net

        if self.drop_conv:
            print('drop conv!')
            net = mx.symbol.Dropout(name=prefix + 'conv_drop', data=net, p=dropout)

        conv = net

        def get_conv(name, net, k3, num_filter, stride):
            return Convolution(
                name=name,
                data=net,
                num_filter=num_filter // 4,
                kernel=(3, 3) if k3 else (1, 1),
                stride=(stride, stride),
                pad=(1, 1) if k3 else (0, 0),
                no_bias=self.no_bias)

        def get_branches(net, block, first_act, num_filter, rows, cols, stride):
            act = get_act(prefix + 'block%d_branch1_conv1' % (block + 1), net) if first_act else net
            b1 = get_conv(prefix + 'block%d_branch1_conv1' % (block + 1), act, False, num_filter, 1)

            b2 = get_act(prefix + 'block%d_branch2_conv2' % (block + 1), b1)
            b2 = get_conv(prefix + 'block%d_branch2_conv2' % (block + 1), b2, True, num_filter, stride)

            b3 = get_act(prefix + 'block%d_branch3_conv3' % (block + 1), b2)
            b3 = get_conv(prefix + 'block%d_branch3_conv3' % (block + 1), b3, True, num_filter, 1)

            return b1, b2, b3


        out[prefix + 'bottleneck'] = net

        return out

    def __init__(
        self,
        num_gesture,
        num_subject,
        num_filter=constant.NUM_FILTER,
        num_pixel=constant.NUM_PIXEL,
        num_hidden=constant.NUM_HIDDEN,
        num_bottleneck=constant.NUM_BOTTLENECK,
        num_feature_block=constant.NUM_FEATURE_BLOCK,
        num_gesture_block=constant.NUM_GESTURE_BLOCK,
        num_subject_block=constant.NUM_SUBJECT_BLOCK,
        dropout=constant.DROPOUT,
        coral=False,
        num_channel=1,
        revgrad=False,
        tzeng=False,
        num_presnet=0,
        presnet_branch=None,
        drop_presnet=False,
        bng=False,
        soft_label=False,
        minibatch=False,
        confuse_conv=False,
        confuse_all=False,
        subject_wd=None,
        drop_branch=False,
        pool=False,
        zscore=True,
        zscore_bng=False,
        output=None,
        num_stream=1,
        for_training=False,
        presnet_promote=False,
        faug=0,
        fusion_type='single',
        drop_conv=False,
        drop_presnet_branch=False,
        drop_presnet_proj=False,
        presnet_proj_type='A',
        bn_wd_mult=0,
        pixel_reduce_loss_weight=0,
        pixel_reduce_bias=False,
        pixel_reduce_kernel=1,
        pixel_reduce_stride=1,
        pixel_reduce_pad=0,
        pixel_reduce_norm=False,
        pixel_reduce_reg_out=False,
        num_pixel_reduce_filter=None,
        fast_pixel_reduce=True,
        num_conv=2,
        drop_pixel=(-1,),
        presnet_dense=False,
        conv_shortcut=False,
        num_semg_row=constant.NUM_SEMG_ROW,
        num_semg_col=constant.NUM_SEMG_COL,
        return_bottleneck=False,
        **kargs):
            self.num_semg_row = num_semg_row
            self.num_semg_col = num_semg_col
            self.conv_shortcut = conv_shortcut
            self.presnet_dense = presnet_dense
            self.drop_pixel = drop_pixel
            self.num_conv = num_conv
            self.for_training = for_training
            self.num_channel = num_channel
            self.num_subject = num_subject
            self.num_presnet = num_presnet
            self.presnet_branch = presnet_branch
            self.drop_presnet = drop_presnet
            self.no_bias = True
            self.bng = bng
            self.fusion_type = fusion_type
            self.tzeng = tzeng
            self.minibatch = minibatch
            self.drop_branch = drop_branch
            self.pool = pool
            self.num_stream = num_stream
            self.drop_conv = drop_conv
            self.drop_presnet_branch = drop_presnet_branch
            self.drop_presnet_proj = drop_presnet_proj
            self.presnet_proj_type = presnet_proj_type
            self.bn_wd_mult = bn_wd_mult
            self.presnet_promote = presnet_promote
            self.pixel_reduce_loss_weight = pixel_reduce_loss_weight
            self.pixel_reduce_bias = pixel_reduce_bias
            if not isinstance(pixel_reduce_kernel, (list, tuple)):
                pixel_reduce_kernel = [pixel_reduce_kernel for _ in range(num_pixel)]
            self.pixel_reduce_kernel = pixel_reduce_kernel
            if not isinstance(pixel_reduce_stride, (list, tuple)):
                pixel_reduce_stride = [pixel_reduce_stride for _ in range(num_pixel)]
            self.pixel_reduce_stride = pixel_reduce_stride
            if not isinstance(pixel_reduce_pad, (list, tuple)):
                pixel_reduce_pad = [pixel_reduce_pad for _ in range(num_pixel)]
            self.pixel_reduce_pad = pixel_reduce_pad
            self.pixel_reduce_norm = pixel_reduce_norm
            self.pixel_reduce_reg_out = pixel_reduce_reg_out
            if not isinstance(num_pixel_reduce_filter, (list, tuple)):
                num_pixel_reduce_filter = [num_pixel_reduce_filter for _ in range(num_pixel)]
            self.num_pixel_reduce_filter = num_pixel_reduce_filter
            self.fast_pixel_reduce = fast_pixel_reduce

            def get_first_stage_stream(prefix):
                net = mx.symbol.Variable(name=prefix + 'data', attr={'tag': '1'})
                return net

            def get_stream(prefix, semg_row, semg_col, fusion_type=None, weight=None):
                if prefix:
                    net = mx.symbol.Variable(name=prefix + 'data', attr={'tag': '1'})
                else:
                    net = mx.symbol.Variable(name=prefix + 'data')

                if zscore:
                    print('zscore before convnet\n{}', prefix)
                    net = (self.get_bng if zscore_bng else self.get_bn)(prefix + 'zscore', net)
                    shortcut = net

                if fusion_type is None or fusion_type == 'single':
                    features = self.get_feature(
                        data=net,
                        num_filter=num_filter,
                        num_pixel=num_pixel,
                        num_block=num_feature_block,
                        num_hidden=num_hidden,
                        num_bottleneck=num_bottleneck,
                        dropout=dropout,
                        semg_row=semg_row,
                        semg_col=semg_col,
                        prefix=prefix)
                elif fusion_type == 'multi_no_imu' or fusion_type == 'multi_with_imu':
                    features = self.get_multistream_feature(
                        data=net,
                        num_filter=num_filter,
                        num_pixel=num_pixel,
                        num_block=num_feature_block,
                        num_hidden=num_hidden,
                        num_bottleneck=num_bottleneck,
                        dropout=dropout,
                        semg_row=semg_row,
                        semg_col=semg_col,
                        prefix=prefix,
                        weight=weight)
                elif fusion_type == 'single_with_imu':
                    features = self.get_feautre_v2(
                        data=net,
                        num_filter=num_filter,
                        num_pixel=num_pixel,
                        num_block=num_feature_block,
                        num_hidden=num_hidden,
                        num_bottleneck=num_bottleneck,
                        dropout=dropout,
                        semg_row=semg_row,
                        semg_col=semg_col,
                        prefix=prefix,
                        weight=weight)

                if zscore:
                    features[prefix + 'shortcut'] = shortcut

                return features[prefix + 'bottleneck'], features

            if num_stream == 1:
                feature, features = get_stream('', self.num_semg_row, self.num_semg_col)
                loss = features['loss']
            else:
                if fusion_type == 'multi_no_imu':
                    weight = [mx.symbol.Variable('conv%d_shared_weight' % (i)) for i in range(2)]

                    assert (len(self.num_semg_row) == len(self.num_semg_col))
                    assert (len(self.num_semg_row) == 3)
                    print('semg row:{}, semg col:{}', self.num_semg_row, self.num_semg_col)

                    feature_1 = get_stream('stream0_', self.num_semg_row[0], self.num_semg_col[0], fusion_type)[0]
                    feature_2 = get_stream('stream1_', self.num_semg_row[0], self.num_semg_col[0], fusion_type)[0]
                    feature_3 = get_stream('stream2_', self.num_semg_row[0], self.num_semg_col[0], fusion_type)[0]

                    print(self.infer_shape(feature_1))
                    print(self.infer_shape(feature_2))
                    print(self.infer_shape(feature_3))

                    subnet1 = mx.symbol.Concat(*[feature_1, feature_2, feature_3], dim=1)
                    name = 'subnet1_cfc'
                    subnet1 = mx.symbol.Activation(name=name + '_relu', data=subnet1, act_type='relu')
                    print('concatenated conv feature shape:', self.infer_shape(subnet1))
                    prefix = 'subnet1_'
                    subnet1 = self.get_2layer_lc(prefix=prefix, data=subnet1)
                    print('subnet1 LC2 shape:', self.infer_shape(subnet1))

                    name = 'subnet1_fc1'
                    subnet1 = self.get_fc(
                        name=name,
                        data=subnet1,
                        num_hidden=512,
                        no_bias=self.no_bias)
                    subnet1 = self.get_bn_relu(name, subnet1)
                    subnet1 = Dropout(
                        name=name + '_drop',
                        data=subnet1,
                        p=dropout)
                    gesture_label = mx.symbol.Variable(name='gesture_softmax_label')
                    fc_attr = {}

                    name = 'subnet1_fc2'
                    subnet1 = self.get_fc(
                        name=name,
                        data=subnet1,
                        num_hidden=512,
                        no_bias=self.no_bias)
                    subnet1 = self.get_bn_relu(name, subnet1)
                    name = 'subnet1_gesture'
                    subnet1 = mx.symbol.FullyConnected(
                        name=name + '_last_fc',
                        data=subnet1,
                        num_hidden=num_gesture,
                        no_bias=False,
                        attr=dict(**fc_attr))
                    subnet1 = mx.symbol.SoftmaxOutput(name=name + '_softmaxactive', data=subnet1, label=gesture_label)
                    print('subnet1 shape: {}', self.infer_shape(subnet1))


                    prefix = 'stream0_'
                    subnet2 =  feature_1
                    subnet2 = self.get_2layer_lc(prefix=prefix, data=subnet2)
                    feature_1 = subnet2

                    name = 'stream0_fusion_fc1'
                    subnet2 = feature_1
                    subnet2 = self.get_fc(
                        name=name,
                        data=subnet2,
                        num_hidden=512,
                        no_bias=self.no_bias)
                    subnet2 = self.get_bn_relu(name, subnet2)
                    subnet2 = Dropout(
                        name=name + '_drop',
                        data=subnet2,
                        p=dropout)
                    feature_1 = subnet2

                    prefix = 'stream1_'
                    subnet2 = feature_2
                    subnet2 = self.get_2layer_lc(prefix=prefix, data=subnet2)
                    feature_2 = subnet2

                    name = 'stream1_fusion_fc1'
                    subnet2 = feature_2
                    subnet2 = self.get_fc(
                        name=name,
                        data=subnet2,
                        num_hidden=512,
                        no_bias=self.no_bias)
                    subnet2 = self.get_bn_relu(name, subnet2)
                    subnet2 = Dropout(
                        name = name + '_drop',
                        data=subnet2,
                        p=dropout)
                    feature_2 = subnet2

                    prefix = 'stream2_'
                    subnet2 = feature_3
                    subnet2 = self.get_2layer_lc(prefix=prefix, data=subnet2)
                    feature_3 = subnet2

                    name = 'stream2_fusion_fc1'
                    subnet2 = feature_3
                    subnet2 = self.get_fc(
                        name=name,
                        data=subnet2,
                        num_hidden=512,
                        no_bias=self.no_bias)
                    subnet2 = self.get_bn_relu(name, subnet2)
                    subnet2 = Dropout(
                        name=name + '_drop',
                        data=subnet2,
                        p=dropout)
                    feature_3 = subnet2

                    subnet2 = mx.symbol.Concat(*[feature_1, feature_2, feature_3], dim=1)

                    name = 'subnet2_cfc'
                    subnet2 = mx.symbol.Activation(name=name + '_relu', data=subnet2, act_type='relu')

                    name = 'fusion_bottleneck'
                    subnet2 = self.get_fc(
                        name=name,
                        data=subnet2,
                        num_hidden=512,
                        no_bias=self.no_bias)
                    subnet2 = self.get_bn_relu(name, subnet2)

                    name = 'subnet2_gesture'
                    subnet2 = mx.symbol.FullyConnected(
                        name=name + '_last_fc',
                        data=subnet2,
                        num_hidden=num_gesture,
                        no_bias=False,
                        attr=dict(*fc_attr))
                    subnet2 = mx.symbol.SoftmaxOutput(name=name + '_softmaxactive', data=subnet2, label=gesture_label)
                    print('sub net2 shape: {}', self.infer_shape(subnet2))

                    gesture_softmax = mx.symbol.Concat(*[subnet1, subnet2], dim=1)
                    gesture_softmax = mx.symbol.Reshape(gesture_softmax, shape=(0, 2, -1))
                    print('gesture_softmax label: {}', self.infer_shape(gesture_softmax))
                    gesture_softmax = mx.symbol.sum(gesture_softmax, axis=1)

                    name = 'fusion_cfc'
                    gesture_softmax = mx.symbol.Activation(name=name + '_relu', data=gesture_softmax, act_type='relu')

                    feature = gesture_softmax


                elif fusion_type == 'multi_with_imu':

                    weight = [mx.symbol.Variable('conv%d_shared_weight' % (i)) for i in range(2)]

                    assert (len(self.num_semg_row) == len(self.num_semg_col))
                    assert (len(self.num_semg_row) == 4)  # last one is for imu
                    print('semg row:{}, semg col:{}', self.num_semg_row, self.num_semg_col)

                    feature_1 = get_stream('stream0_', self.num_semg_row[0], self.num_semg_col[0], fusion_type)[0]
                    feature_2 = get_stream('stream1_', self.num_semg_row[0], self.num_semg_col[0], fusion_type)[0]
                    feature_3 = get_stream('stream2_', self.num_semg_row[0], self.num_semg_col[0], fusion_type)[0]
                    feature_imu = get_stream('stream3_', self.num_semg_row[0], self.num_semg_col[0], fusion_type)[0]

                    print(self.infer_shape(feature_1))
                    print(self.infer_shape(feature_2))
                    print(self.infer_shape(feature_3))
                    print(self.infer_shape(feature_imu))

                    subnet1 = mx.symbol.Concat(*[feature_1, feature_2, feature_3, feature_imu], dim=1)
                    name = 'subnet1_cfc'
                    subnet1 = mx.symbol.Activation(name=name + '_relu', data=subnet1, act_type='relu')
                    print('concatenated conv feature shape:', self.infer_shape(subnet1))
                    prefix = 'subnet1_'
                    subnet1 = self.get_2layer_lc(prefix=prefix, data=subnet1)
                    print('subnet1 LC2 shape:', self.infer_shape(subnet1))

                    name = 'subnet1_fc1'
                    subnet1 = self.get_fc(
                        name=name,
                        data=subnet1,
                        num_hidden=512,
                        no_bias=self.no_bias)
                    subnet1 = self.get_bn_relu(name, subnet1)
                    subnet1 = Dropout(
                        name=name + '_drop',
                        data=subnet1,
                        p=dropout)
                    gesture_label = mx.symbol.Variable(name='gesture_softmax_label')
                    fc_attr = {}

                    name = 'subnet1_fc2'
                    subnet1 = self.get_fc(
                        name=name,
                        data=subnet1,
                        num_hidden=512,
                        no_bias=self.no_bias)
                    subnet1 = self.get_bn_relu(name, subnet1)
                    name = 'subnet1_gesture'
                    subnet1 = mx.symbol.FullyConnected(
                        name=name + '_last_fc',
                        data=subnet1,
                        num_hidden=num_gesture,
                        no_bias=False,
                        attr=dict(**fc_attr))
                    subnet1 = mx.symbol.SoftmaxOutput(name=name + '_softmaxactive', data=subnet1, label=gesture_label)
                    print('subnet1 shape: {}', self.infer_shape(subnet1))


                    prefix = 'stream0_'
                    subnet2 =  feature_1
                    subnet2 = self.get_2layer_lc(prefix=prefix, data=subnet2)
                    feature_1 = subnet2

                    name = 'stream0_fusion_fc1'
                    subnet2 = feature_1
                    subnet2 = self.get_fc(
                        name=name,
                        data=subnet2,
                        num_hidden=512,
                        no_bias=self.no_bias)
                    subnet2 = self.get_bn_relu(name, subnet2)
                    subnet2 = Dropout(
                        name=name + '_drop',
                        data=subnet2,
                        p=dropout)
                    feature_1 = subnet2

                    prefix = 'stream1_'
                    subnet2 = feature_2
                    subnet2 = self.get_2layer_lc(prefix=prefix, data=subnet2)
                    feature_2 = subnet2

                    name = 'stream1_fusion_fc1'
                    subnet2 = feature_2
                    subnet2 = self.get_fc(
                        name=name,
                        data=subnet2,
                        num_hidden=512,
                        no_bias=self.no_bias)
                    subnet2 = self.get_bn_relu(name, subnet2)
                    subnet2 = Dropout(
                        name = name + '_drop',
                        data=subnet2,
                        p=dropout)
                    feature_2 = subnet2

                    prefix = 'stream2_'
                    subnet2 = feature_3
                    subnet2 = self.get_2layer_lc(prefix=prefix, data=subnet2)
                    feature_3 = subnet2

                    name = 'stream2_fusion_fc1'
                    subnet2 = feature_3
                    subnet2 = self.get_fc(
                        name=name,
                        data=subnet2,
                        num_hidden=512,
                        no_bias=self.no_bias)
                    subnet2 = self.get_bn_relu(name, subnet2)
                    subnet2 = Dropout(
                        name=name + '_drop',
                        data=subnet2,
                        p=dropout)
                    feature_3 = subnet2

                    prefix = 'stream3_'
                    subnet2 = feature_imu
                    subnet2 = self.get_2layer_lc(prefix=prefix, data=subnet2)
                    feature_imu = subnet2

                    name = 'stream3_fusion_fc1'
                    subnet2 = feature_imu
                    subnet2 = self.get_fc(
                        name=name,
                        data=subnet2,
                        num_hidden=512,
                        no_bias=self.no_bias)
                    subnet2 = self.get_bn_relu(name, subnet2)
                    subnet2 = Dropout(
                        name=name + '_drop',
                        data=subnet2,
                        p=dropout)
                    feature_imu = subnet2

                    subnet2 = mx.symbol.Concat(*[feature_1, feature_2, feature_3, feature_imu], dim=1)

                    name = 'subnet2_cfc'
                    subnet2 = mx.symbol.Activation(name=name + '_relu', data=subnet2, act_type='relu')

                    name = 'fusion_bottleneck'
                    subnet2 = self.get_fc(
                        name=name,
                        data=subnet2,
                        num_hidden=512,
                        no_bias=self.no_bias)
                    subnet2 = self.get_bn_relu(name, subnet2)

                    name = 'subnet2_gesture'
                    subnet2 = mx.symbol.FullyConnected(
                        name=name + '_last_fc',
                        data=subnet2,
                        num_hidden=num_gesture,
                        no_bias=False,
                        attr=dict(*fc_attr))
                    subnet2 = mx.symbol.SoftmaxOutput(name=name + '_softmaxactive', data=subnet2, label=gesture_label)
                    print('sub net2 shape: {}', self.infer_shape(subnet2))

                    gesture_softmax = mx.symbol.Concat(*[subnet1, subnet2], dim=1)
                    gesture_softmax = mx.symbol.Reshape(gesture_softmax, shape=(0, 2, -1))
                    print('gesture_softmax label: {}', self.infer_shape(gesture_softmax))
                    gesture_softmax = mx.symbol.sum(gesture_softmax, axis=1)

                    name = 'fusion_cfc'
                    gesture_softmax = mx.symbol.Activation(name=name + '_relu', data=gesture_softmax, act_type='relu')

                    feature = gesture_softmax

            features = None
            loss = []

            bottleneck = feature
            if coral:
                feature = mx.symbol.FullyConnected(name='coral',
                                                   data=feature,
                                                   num_hidden=num_bottleneck,
                                                   no_bias=False)
            if faug:
                feature_before_faug = feature
                feature = feature + mx.symbol.Variable('faug')

            gesture_branch_kargs = {}
            if fusion_type.find('multi') != -1:
                gesture_fc = []
            else:
                gesture_softmax, gesture_fc = self.get_branch(name='gesture',
                                                              data=feature,
                                                              num_class=num_gesture,
                                                              num_block=num_gesture_block,
                                                              num_hidden=num_bottleneck,
                                                              use_ignore=True,
                                                              return_fc=True,
                                                              **gesture_branch_kargs)
            loss.insert(0, gesture_softmax)

            if soft_label:
                net = mx.symbol.SoftmaxActivation(gesture_fc / 10)
                net = mx.symbol.log(net)
                net = mx.symbol.broadcast_mul(net, mx.symbol.Variable('soft_label'))
                net = -symsum(data=net)
                net = mx.symbol.MakeLoss(data=net, grad_scale=0.1)
                loss.append(net)

            if (revgrad or tzeng) and num_subject > 0:
                assert not (confuse_conv and confuse_all)
                if confuse_conv:
                    feature = features['conv2']
                if confuse_all:
                    feature = mx.symbol.Concat(*[mx.symbol.Flatten(features[name]) for name in sorted(features)])

                if revgrad:
                    feature = mx.symbol.Custom(data=feature, name='grl', op_type='GRL')
                else:
                    feature = mx.symbol.Custom(data=feature, name='bottleneck_gradscale', op_type='GradScale')
                subject_softmax_loss, subject_fc = self.get_branch(
                    name='subject',
                    data=feature,
                    num_class=num_subject,
                    num_block=num_subject_block,
                    num_hidden=num_bottleneck,
                    grad_scale=kargs.pop('subject_softmax_weight', 0.1),
                    use_ignore=True,
                    return_fc=True,
                    fc_attr={'wd_mult': str(subject_wd)} if subject_wd is not None else {})
                loss.append(subject_softmax_loss)
                if tzeng:
                    subject_fc = mx.symbol.Custom(data=subject_fc, name='subject_confusion_gradscale', op_type='GradScale')
                    subject_softmax = mx.symbol.SoftmaxActivation(subject_fc)
                    subject_confusion_loss = mx.symbol.MakeLoss(
                        data=-symsum(data=mx.symbol.log(subject_softmax + 1e-8)) / num_subject,
                        grad_scale=kargs.pop('subject_confusion_loss_weight', 0.1))
                    loss.append(subject_confusion_loss)

            target_gesture_loss_weight = kargs.get('target_gesture_loss_weight')
            if target_gesture_loss_weight is not None:
                loss.append(mx.symbol.SoftmaxOutput(
                name='target_gesture_softmax',
                data=gesture_fc,
                grad_scale=target_gesture_loss_weight,
                use_ignore=True))

            if output is None:
                self.net = loss[0] if len(loss) == 1 else mx.sym.Group(loss)
            else:
                assert_equal(len(loss), 1)
                self.net = loss[0].get_internals()[output]

            if return_bottleneck:
                self.net = bottleneck

            if hasattr(self.net, '__dict__'):
                print('dict')
            self.net.num_semg_row = num_semg_row
            self.net.num_semg_col = num_semg_col
            self.net.num_presnet = num_presnet
            self.net.num_channel = num_channel

            self.net.data_shape_1 = self.num_channel
            self.net.num_feature = num_bottleneck

            if faug:
                self.net.feature_before_faug = feature_before_faug

            self.net.presnet_proj_type = presnet_proj_type


def symsum(data):
    return mx.symbol.sum(data, axis=1)


def get_symbol(*args, **kargs):
    return Symbol(*args, **kargs).net


def Dropout(**kargs):
    p = kargs.pop('p')
    return kargs.pop('data') if p == 0 else mx.symbol.Dropout(p=p, **kargs)


def Convolution(*args, **kargs):
    kargs['cudnn_tune'] = 'fastest'
    return mx.symbol.Convolution(*args, **kargs)




















