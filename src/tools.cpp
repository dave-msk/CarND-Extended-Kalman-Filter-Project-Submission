#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;
  unsigned long size = estimations.size();
  if (!size || size != ground_truth.size())
    return rmse;

  VectorXd residual;
  for (unsigned i = 0; i < size; ++i) {
    residual = estimations[i] - ground_truth[i];
    residual = residual.array().pow(2);
    rmse += residual;
  }
  rmse /= (float) size;
  rmse = rmse.cwiseSqrt();
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
  TODO:
    * Calculate a Jacobian here.
  */
  const float epsilon = 1e-6;
  const float limit = 1/epsilon;

  float px = x_state(0);
  float py = x_state(1);

  float d = px*px + py*py;
  float r_d =  (d < epsilon) ? limit : 1/d;
  float r_d_sqrt = sqrt(r_d);
  float r_d_sqrt_p3 = r_d*r_d_sqrt;

  float vx = x_state(2);
  float vy = x_state(3);

  MatrixXd Hj(3, 4);
  float cross_diff = vx*py - vy*px;
  Hj << px*r_d_sqrt, py*r_d_sqrt, 0, 0,
        -py*r_d, px*r_d, 0, 0,
        py*cross_diff*r_d_sqrt_p3, -px*cross_diff*r_d_sqrt_p3, px*r_d_sqrt, py*r_d_sqrt;
  return Hj;
}
