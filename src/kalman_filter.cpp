#include "kalman_filter.h"
#include "math.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
  TODO:
    * predict the state
  */
  MatrixXd Ft = F_.transpose();
  x_ = F_*x_;
  P_ = F_*P_*Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Kalman Filter equations
  */
  VectorXd y = z - H_*x_;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_*P_*Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_*Ht*Si;
  MatrixXd I = MatrixXd::Identity(K.rows(), K.rows());
  x_ += K*y;
  P_ = (I - K*H_)*P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
  TODO:
    * update the state by using Extended Kalman Filter equations
  */
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);

  float rho = sqrt(px*px + py*py);
  float phi = std::atan2(py, px);
  float rhoDot = (rho >= 1e-6) ? (px*vx + py*vy)/rho : 0;
  VectorXd h(3);
  h << rho, phi, rhoDot;

  VectorXd y = z - h;
  if (y(1) > M_PI) y(1) -= 2*M_PI;
  else if (y(1) < -M_PI) y(1) += 2*M_PI;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_*P_*Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_*Ht*Si;
  MatrixXd I = MatrixXd::Identity(K.rows(), K.rows());
  x_ += K*y;
  P_ = (I - K*H_)*P_;
}
