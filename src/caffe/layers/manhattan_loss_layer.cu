#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ManhattanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  top[0]->mutable_cpu_data()[0] = diff_.asum_data() / bottom[0]->num();
}

template <typename Dtype>
void ManhattanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const int count = bottom[i]->count();
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype scale = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_gpu_sign(count, diff_.gpu_data(), bottom[i]->mutable_gpu_diff());
      caffe_gpu_scal(count, scale, bottom[i]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(ManhattanLossLayer);

}  // namespace caffe
