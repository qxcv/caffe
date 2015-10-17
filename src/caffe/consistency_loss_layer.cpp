#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ConsistencyLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels() % 4, 0)
      << "Joint location channel should have 4*j joints (where j is the "
      "number of joints per frame) in [x1 y1 x1' y1' x2 y2 x2' y2' ...] "
      "format";
  CHECK_GT(bottom[1]->width(), 0)
      << "At least some flow data should be present";
  CHECK_GT(bottom[1]->height(), 0)
      << "At least some flow data should be present";
  CHECK_EQ(bottom[1]->channels(), 2)
      << "Flow input should have two channels (dx and dy)";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void ConsistencyLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  // bottom[0] is the pose
  // bottom[1] is the flow
  int num_joints = bottom[0]->channels() / 4;
  int flow_max_x = bottom[1]->width() - 1;
  int flow_max_y = bottom[1]->height() - 1;
  Dtype loss = 0;
  const Dtype *label_data = bottom[0]->cpu_data();
  Dtype *diff_data = diff_.mutable_cpu_data();
  for (int n = 0; n < bottom[0]->num(); ++n) {
    for (int w = 0; w < bottom[0]->width(); ++w) {
      for (int h = 0; h < bottom[0]->height(); ++h) {
        for (int j = 0; j < num_joints; ++j) {
          int fst_x_idx = bottom[0]->offset(n, 4*j, h, w),
              fst_y_idx = bottom[0]->offset(n, 4*j+1, h, w),
              snd_x_idx = bottom[0]->offset(n, 4*j+2, h, w),
              snd_y_idx = bottom[0]->offset(n, 4*j+3, h, w);

          // We'll use these (x, y) coords to look up flow, but we have to clamp
          // them in [0, maximum flow coordinate in given dimension) if we want
          // to look them up (hence the {fst,snd}_{x,y}_c values)
          Dtype fst_x = label_data[fst_x_idx],
                fst_y = label_data[fst_y_idx],
                snd_x = label_data[snd_x_idx],
                snd_y = label_data[snd_y_idx];

          int fst_x_c = std::min(std::max(fst_x, Dtype(0)), Dtype(flow_max_x)),
              fst_y_c = std::min(std::max(fst_y, Dtype(0)), Dtype(flow_max_y)),
              snd_x_c = std::min(std::max(snd_x, Dtype(0)), Dtype(flow_max_x)),
              snd_y_c = std::min(std::max(snd_y, Dtype(0)), Dtype(flow_max_y));

          // Get mean flow
          Dtype mean_flow_x = 0.5 * bottom[1]->data_at(n, 0, fst_y_c, fst_x_c)
                  + 0.5 * bottom[1]->data_at(n, 0, snd_y_c, snd_x_c),
                mean_flow_y = 0.5 * bottom[1]->data_at(n, 1, fst_y_c, fst_x_c)
                  + 0.5 * bottom[1]->data_at(n, 1, fst_y_c, snd_x_c);
          Dtype x_diff = fst_x + mean_flow_x - snd_x,
                y_diff = fst_y + mean_flow_y - snd_y;

          // Add in some loss
          loss += fabs(x_diff);
          loss += fabs(y_diff);

          // Update all four diffs (these just store the derivative)
          diff_data[fst_x_idx] = caffe_sign(x_diff);
          diff_data[fst_y_idx] = caffe_sign(y_diff);
          diff_data[snd_x_idx] = -diff_data[fst_x_idx];
          diff_data[snd_y_idx] = -diff_data[fst_y_idx];
        }
      }
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / bottom[0]->num();
}

template <typename Dtype>
void ConsistencyLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to flow inputs.";
  }
  if (propagate_down[0]) {
    const int count = bottom[0]->count();
    const Dtype scale = top[0]->cpu_diff()[0] / bottom[0]->num();
    caffe_copy(count, diff_.cpu_data(), bottom[0]->mutable_cpu_diff());
    caffe_scal(count, scale, bottom[0]->mutable_cpu_diff());
  }
}

INSTANTIATE_CLASS(ConsistencyLossLayer);
REGISTER_LAYER_CLASS(ConsistencyLoss);

}  // namespace caffe
