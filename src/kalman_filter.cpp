#include "kalman_filter.h"
#include <iostream>

using namespace std;

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

/**
  * predict the state
*/
void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

/**
  * update the state by using Kalman Filter equations
*/
void KalmanFilter::Update(const VectorXd &z) {
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

/**
  * update the state by using Extended Kalman Filter equations
*/
void KalmanFilter::UpdateEKF(const VectorXd &z) {
  //recover state parameters
  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);
  //pre-compute a set of terms to avoid repeated calculation
  float px2_plus_py2 = px * px + py * py;
  float sqrt_px2_plus_py2 = sqrt(px2_plus_py2);
  //check division by zero
  if(fabs(px2_plus_py2) < 0.0001){
    cout << "UpdateEKF() - Error - Division by Zero" << endl;
    return;
  }

  //calculate z_pred = h(x)
  VectorXd z_pred(3);
  z_pred << sqrt_px2_plus_py2, atan2(py, px), (px * vx + py * vy) / sqrt_px2_plus_py2;

  VectorXd y = z - z_pred;
  //adjust y(1) = phi - phi_pred so it fits into the range [-pi, pi]
  const float PI = 22.0 / 7;
  if (y(1) < -PI)
    y(1) += 2 * PI;
  else if (y(1) > PI)
    y(1) -= 2 * PI;

  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
