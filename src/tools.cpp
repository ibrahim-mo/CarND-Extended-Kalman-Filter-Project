#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

/**
* Calculate the RMSE here.
*/
VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  int s1=estimations.size(), s2=ground_truth.size();
  if (s1==0 || s1!=s2) {
    cout << "Invalid estimation or ground_truth data" << endl;
    return rmse;
  }

  //accumulate squared residuals
  for(int i=0; i < s1; ++i) {
    VectorXd residual = (estimations[i] - ground_truth[i]);
    residual = residual.array() * residual.array();
    rmse += residual;
  }

  //calculate the mean
  rmse = rmse / s1;

  //calculate the squared root
  rmse = rmse.array().sqrt();

  //check RMSE against required accuracy
  /*VectorXd accuracy(4);
  accuracy << 0.11, 0.11, 0.52, 0.52;
  string x_names[4] = {"px", "py", "vx", "vy"};
  for (int j=0; j<4; j++) {
    if (rmse(j) > accuracy(j)) {
      cout << "Warning - RMSE(" << x_names[j] << ") = " << rmse(j) << " violates accuracy requirement of " << accuracy(j) << endl;
    }
  }*/

  //return the result
  return rmse;
}

/**
* Calculate a Jacobian here.
*/
MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj(3,4);
  //recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  //pre-compute a set of terms to avoid repeated calculation
  float px2_plus_py2 = px * px + py * py;
  float sqrt_px2_plus_py2 = sqrt(px2_plus_py2);
  float pwr1_5_x2_plus_py2 = px2_plus_py2 * sqrt_px2_plus_py2;

  //check division by zero
  if(fabs(px2_plus_py2) < 0.0001){
    cout << "CalculateJacobian() - Error - Division by Zero" << endl;
    return Hj;
  }

  //compute the Jacobian matrix
  Hj << px / sqrt_px2_plus_py2, py / sqrt_px2_plus_py2, 0, 0,
    - py / px2_plus_py2, px / px2_plus_py2, 0, 0,
    py * (vx * py - vy * px) / pwr1_5_x2_plus_py2, px * (vy * px - vx * py) / pwr1_5_x2_plus_py2,
      px / sqrt_px2_plus_py2, py / sqrt_px2_plus_py2;

  return Hj;
}
