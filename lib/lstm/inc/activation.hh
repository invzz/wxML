#pragma once

#include <Eigen/Dense>

using namespace Eigen;

class ActivationFunction
{
  public:
  static MatrixXd sigmoid(const MatrixXd &x);
  static MatrixXd sigmoid_derivative(const MatrixXd &x);

  static MatrixXd tanh(const MatrixXd &x);
  static MatrixXd tanh_derivative(const MatrixXd &x);

  static MatrixXd relu(const MatrixXd &x);
  static MatrixXd relu_derivative(const MatrixXd &x);
};

// Inline definitions for activation functions

inline MatrixXd ActivationFunction::sigmoid(const MatrixXd &x) { return 1.0 / (1.0 + (-x.array()).exp()); }

inline MatrixXd ActivationFunction::sigmoid_derivative(const MatrixXd &x)
{
  MatrixXd sigmoid_x = sigmoid(x);
  return sigmoid_x.array() * (1.0 - sigmoid_x.array());
}

inline MatrixXd ActivationFunction::tanh(const MatrixXd &x) { return x.array().tanh(); }

inline MatrixXd ActivationFunction::tanh_derivative(const MatrixXd &x) { return 1.0 - (x.array().tanh().square()); }

inline MatrixXd ActivationFunction::relu(const MatrixXd &x) { return x.array().max(0.0); }

inline MatrixXd ActivationFunction::relu_derivative(const MatrixXd &x) { return (x.array() > 0.0).cast<double>(); }