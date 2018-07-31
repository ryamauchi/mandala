    def _forward_cudnn(self, x, W, b, y):
        pad = (self.ph, self.pw)
        stride = (self.sy, self.sx)
        dilation = (self.dy, self.dx)
        auto_tune = configuration.config.autotune
        tensor_core = configuration.config.use_cudnn_tensor_core
        cuda.cudnn.convolution_forward(
            x, W, b, y, pad, stride, dilation, self.groups,
            auto_tune=auto_tune, tensor_core=tensor_core)
        return y,