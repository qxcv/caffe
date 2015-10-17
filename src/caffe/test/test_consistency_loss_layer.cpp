#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class ConsistencyLossLayerTest : public CPUDeviceTest<TypeParam> {
 typedef typename TypeParam::Dtype Dtype;

 protected:
  ConsistencyLossLayerTest()
      : blob_bottom_joints_(new Blob<Dtype>(10, 8, 1, 1)),
        blob_bottom_flow_(new Blob<Dtype>(10, 2, 10, 10)),
        blob_top_loss_(new Blob<Dtype>()) {
    // Fill joints so that they're around the middle of our 10x10 images
    FillerParameter joint_filler_param;
    joint_filler_param.set_mean(5);
    joint_filler_param.set_std(2);
    GaussianFiller<Dtype> joint_filler(joint_filler_param);
    joint_filler.Fill(this->blob_bottom_joints_);
    blob_bottom_vec_.push_back(blob_bottom_joints_);

    // Fill flow so that it changes gradually in each dimension from top left to
    // bottom right
    Dtype *flow_data = blob_bottom_flow_->mutable_cpu_data();
    for (int n = 0; n < 10; ++n) {
      for (int c = 0; c < 2; ++c) {
        // Generate start and end flow for this direction using Gaussian rng
        // with mean 0 and variance 2
        Dtype start, end;
        caffe_rng_gaussian(1, Dtype(0), Dtype(3), &start);
        caffe_rng_gaussian(1, start, Dtype(1), &end);
        for (int h = 0; h < 10; ++h) {
          for (int w = 0; w < 10; ++w) {
            Dtype val = start + (end - start) * Dtype(w + h) / Dtype(18);
            int offset = blob_bottom_flow_->offset(n, c, h, w);
            flow_data[offset] = val;
          }
        }
      }
    }

    blob_bottom_vec_.push_back(blob_bottom_flow_);

    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~ConsistencyLossLayerTest() {
    delete blob_bottom_flow_;
    delete blob_bottom_joints_;
    delete blob_top_loss_;
  }

  void TestForward() {
    // Get the loss without a specified objective weight -- should be
    // equivalent to explicitly specifiying a weight of 1.
    LayerParameter layer_param;
    ConsistencyLossLayer<Dtype> layer_weight_1(layer_param);
    layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_1 =
        layer_weight_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // Get the loss again with a different objective weight; check that it is
    // scaled appropriately.
    const Dtype kLossWeight = 3.7;
    layer_param.add_loss_weight(kLossWeight);
    ConsistencyLossLayer<Dtype> layer_weight_2(layer_param);
    layer_weight_2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_2 =
        layer_weight_2.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype kErrorMargin = 1e-5;
    EXPECT_NEAR(loss_weight_1 * kLossWeight, loss_weight_2, kErrorMargin);
    // Make sure the loss is non-trivial.
    const Dtype kNonTrivialAbsThresh = 1e-1;
    EXPECT_GE(fabs(loss_weight_1), kNonTrivialAbsThresh);
  }

  Blob<Dtype>* const blob_bottom_joints_;
  Blob<Dtype>* const blob_bottom_flow_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(ConsistencyLossLayerTest, TestDtypesAndDevices);

TYPED_TEST(ConsistencyLossLayerTest, TestForward) {
  this->TestForward();
}

TYPED_TEST(ConsistencyLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  ConsistencyLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(0.01, 1e-3, 1701, 0., 0.02);
  // Last argument says to only check gradient w.r.t. first bottom
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
