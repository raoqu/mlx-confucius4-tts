// CAMPPlus speaker (style) encoder for c4tts (external/campplus, 3D-Speaker).
//
// Faithful C++/MLX port of DTDNN.py:CAMPPlus — FCM (2-D conv ResNet front end)
// + DTDNN x-vector (CAM-Dense-TDNN blocks, transit layers, statistics pooling,
// dense). Produces the S2A speaker style embedding from Kaldi fbank features.
// Verified by test_campplus.

#pragma once

#include <string>

#include "c4tts/tensor.h"
#include "c4tts/weights.h"

namespace c4 {

class CAMPPlus {
 public:
  explicit CAMPPlus(const WeightStore& w, const std::string& prefix = "");

  // fbank: (B, T, feat_dim=80) -> embedding (B, 512).
  Tensor forward(const Tensor& fbank) const;

 private:
  // BatchNorm (+ optional ReLU); handles affine-absent BN.
  Tensor bn(const Tensor& x, const std::string& p, bool relu,
            int channel_axis = 1) const;
  Tensor basic_res_block(const Tensor& x, const std::string& p, int stride) const;
  Tensor fcm(const Tensor& x) const;
  Tensor cam_layer(const Tensor& x, const std::string& p, int kernel,
                   int dilation) const;
  Tensor cam_dense_layer(const Tensor& x, const std::string& p, int kernel,
                         int dilation) const;
  Tensor cam_dense_block(const Tensor& x, const std::string& p, int num_layers,
                         int kernel, int dilation) const;
  Tensor transit(const Tensor& x, const std::string& p) const;

  const WeightStore& w_;
  std::string p_;
};

}  // namespace c4
