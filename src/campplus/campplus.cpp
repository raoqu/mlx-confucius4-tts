#include "c4tts/campplus.h"

#include <cmath>
#include <string>
#include <vector>

#include "c4tts/nn.h"
#include "mlx/mlx.h"

namespace c4 {
namespace mx = mlx::core;

CAMPPlus::CAMPPlus(const WeightStore& w, const std::string& prefix)
    : w_(w), p_(prefix) {}

Tensor CAMPPlus::bn(const Tensor& x, const std::string& p, bool relu,
                    int channel_axis) const {
  Tensor mean = w_.get(p + "running_mean");
  Tensor var = w_.get(p + "running_var");
  const Tensor* g = nullptr;
  const Tensor* b = nullptr;
  Tensor gw = mx::array(0.0f), bw = mx::array(0.0f);
  if (w_.has(p + "weight")) {  // affine BN
    gw = w_.get(p + "weight");
    bw = w_.get(p + "bias");
    g = &gw;
    b = &bw;
  }
  Tensor y = nn::batch_norm(x, mean, var, g, b, 1e-5f, channel_axis);
  return relu ? mx::maximum(y, mx::array(0.0f)) : y;
}

// ---- FCM (2-D front end) ----

Tensor CAMPPlus::basic_res_block(const Tensor& x, const std::string& p,
                                 int stride) const {
  Tensor c1 = nn::conv2d(x, w_.get(p + "conv1.weight"), nullptr, stride, 1, 1, 1);
  Tensor out = bn(c1, p + "bn1.", /*relu=*/true);
  Tensor c2 = nn::conv2d(out, w_.get(p + "conv2.weight"), nullptr, 1, 1, 1, 1);
  out = bn(c2, p + "bn2.", /*relu=*/false);

  Tensor residual = x;
  if (w_.has(p + "shortcut.0.weight")) {
    Tensor s = nn::conv2d(x, w_.get(p + "shortcut.0.weight"), nullptr, stride, 1,
                          0, 0);
    residual = bn(s, p + "shortcut.1.", /*relu=*/false);
  }
  return mx::maximum(mx::add(out, residual), mx::array(0.0f));
}

Tensor CAMPPlus::fcm(const Tensor& x) const {
  // x: (B, F, T) -> (B, 1, F, T)
  Tensor h = mx::expand_dims(x, 1);
  h = bn(nn::conv2d(h, w_.get(p_ + "head.conv1.weight"), nullptr, 1, 1, 1, 1),
         p_ + "head.bn1.", true);
  for (int i = 0; i < 2; ++i) {
    h = basic_res_block(h, p_ + "head.layer1." + std::to_string(i) + ".",
                        i == 0 ? 2 : 1);
  }
  for (int i = 0; i < 2; ++i) {
    h = basic_res_block(h, p_ + "head.layer2." + std::to_string(i) + ".",
                        i == 0 ? 2 : 1);
  }
  // conv2: stride (2,1)
  h = nn::conv2d(h, w_.get(p_ + "head.conv2.weight"), nullptr, 2, 1, 1, 1);
  h = bn(h, p_ + "head.bn2.", true);
  // (B, C, H, T) -> (B, C*H, T)
  const int B = h.shape(0), C = h.shape(1), H = h.shape(2), T = h.shape(3);
  return mx::reshape(h, {B, C * H, T});
}

// ---- CAM-Dense-TDNN ----

namespace {
// avg_pool1d(kernel=stride=seg, ceil_mode=True) then broadcast back to T.
Tensor seg_pooling(const Tensor& x, int seg = 100) {
  const int B = x.shape(0), C = x.shape(1), T = x.shape(2);
  const int nseg = (T + seg - 1) / seg;
  const int pad = nseg * seg - T;
  Tensor xp = x;
  if (pad > 0) {
    xp = mx::concatenate({x, mx::zeros({B, C, pad}, x.dtype())}, 2);
  }
  Tensor xr = mx::reshape(xp, {B, C, nseg, seg});
  Tensor seg_sum = mx::sum(xr, 3);  // (B,C,nseg)

  std::vector<float> counts(static_cast<size_t>(nseg), static_cast<float>(seg));
  counts[nseg - 1] = static_cast<float>(T - (nseg - 1) * seg);
  float* ch = new float[nseg];
  for (int i = 0; i < nseg; ++i) ch[i] = counts[i];
  Tensor cnt = mx::reshape(
      mx::array(static_cast<void*>(ch), mx::Shape{nseg}, mx::float32,
                [](void* q) { delete[] static_cast<float*>(q); }),
      {1, 1, nseg});
  Tensor seg_mean = mx::divide(seg_sum, cnt);  // (B,C,nseg)

  // broadcast each segment mean over `seg`, then crop to T.
  Tensor e = mx::broadcast_to(mx::expand_dims(seg_mean, 3), {B, C, nseg, seg});
  e = mx::reshape(e, {B, C, nseg * seg});
  return mx::slice(e, mx::Shape{0, 0, 0}, mx::Shape{B, C, T});
}
}  // namespace

Tensor CAMPPlus::cam_layer(const Tensor& x, const std::string& p, int kernel,
                           int dilation) const {
  const int pad = (kernel - 1) / 2 * dilation;
  Tensor y = nn::conv1d(x, w_.get(p + "linear_local.weight"), nullptr, 1, pad,
                        dilation);
  Tensor context = mx::add(mx::mean(x, 2, true), seg_pooling(x));  // (B,bn,1)+(B,bn,T)
  // context has time dim T (seg_pooling) + broadcast mean -> (B, bn, T)
  Tensor l1w = w_.get(p + "linear1.weight"), l1b = w_.get(p + "linear1.bias");
  context = mx::maximum(nn::conv1d(context, l1w, &l1b), mx::array(0.0f));
  Tensor l2w = w_.get(p + "linear2.weight"), l2b = w_.get(p + "linear2.bias");
  Tensor m = mx::sigmoid(nn::conv1d(context, l2w, &l2b));
  return mx::multiply(y, m);
}

Tensor CAMPPlus::cam_dense_layer(const Tensor& x, const std::string& p,
                                 int kernel, int dilation) const {
  Tensor h = bn(x, p + "nonlinear1.batchnorm.", true);
  h = nn::conv1d(h, w_.get(p + "linear1.weight"), nullptr);  // 1x1, no bias
  h = bn(h, p + "nonlinear2.batchnorm.", true);
  return cam_layer(h, p + "cam_layer.", kernel, dilation);
}

Tensor CAMPPlus::cam_dense_block(const Tensor& x, const std::string& p,
                                 int num_layers, int kernel, int dilation) const {
  Tensor h = x;
  for (int i = 0; i < num_layers; ++i) {
    Tensor out = cam_dense_layer(
        h, p + "tdnnd" + std::to_string(i + 1) + ".", kernel, dilation);
    h = mx::concatenate({h, out}, 1);  // dense connectivity
  }
  return h;
}

Tensor CAMPPlus::transit(const Tensor& x, const std::string& p) const {
  Tensor h = bn(x, p + "nonlinear.batchnorm.", true);
  return nn::conv1d(h, w_.get(p + "linear.weight"), nullptr);  // 1x1, no bias
}

Tensor CAMPPlus::forward(const Tensor& fbank) const {
  Tensor x = mx::transpose(fbank, {0, 2, 1});  // (B,T,F) -> (B,F,T)
  Tensor h = fcm(x);                            // (B, 320, T)

  // xvector.tdnn: Conv1d(320,128,5,stride2,pad2) + BN-relu
  h = nn::conv1d(h, w_.get(p_ + "xvector.tdnn.linear.weight"), nullptr, 2, 2, 1);
  h = bn(h, p_ + "xvector.tdnn.nonlinear.batchnorm.", true);

  const int nlayers[3] = {12, 24, 16};
  const int dils[3] = {1, 2, 2};
  for (int b = 0; b < 3; ++b) {
    h = cam_dense_block(h, p_ + "xvector.block" + std::to_string(b + 1) + ".",
                        nlayers[b], 3, dils[b]);
    h = transit(h, p_ + "xvector.transit" + std::to_string(b + 1) + ".");
  }
  h = bn(h, p_ + "xvector.out_nonlinear.batchnorm.", true);

  // StatsPool: mean + unbiased std over time.
  const int T = h.shape(2);
  Tensor mean = mx::mean(h, 2);  // (B,C)
  Tensor mean_e = mx::expand_dims(mean, 2);
  Tensor var = mx::divide(mx::sum(mx::square(mx::subtract(h, mean_e)), 2),
                          mx::array(static_cast<float>(T - 1)));
  Tensor std = mx::sqrt(var);
  Tensor stats = mx::concatenate({mean, std}, 1);  // (B, 2C)

  // dense: Conv1d(2C,512,1,no bias) on (B,2C,1) then BN (affine=False).
  Tensor d = mx::expand_dims(stats, 2);  // (B,2C,1)
  d = nn::conv1d(d, w_.get(p_ + "xvector.dense.linear.weight"), nullptr);
  d = mx::squeeze(d, 2);  // (B,512)
  return bn(d, p_ + "xvector.dense.nonlinear.batchnorm.", false);
}

}  // namespace c4
