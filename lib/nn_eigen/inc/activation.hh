/**
 * @file activation.hh
 * @author Andres Coronado (andres.coronado@bss.group)
 * @brief Header of activation functions
 * @version 0.1
 * @date 2024-03-07
 *
 * @copyright Copyright (c) 2024
 *
 */
#pragma once
#include <Eigen/Dense>

using namespace Eigen;
/**
 * @brief Base class for activation functions.
 */
class ActivationFunction
{
  public:
  /**
   * @brief Activate function.
   * @param x Input vector.
   * @return Vector after activation.
   */
  virtual VectorXd activate(const VectorXd &x) const = 0;

  /**
   * @brief Calculate derivative of activation function.
   * @param x Input vector.
   * @return Vector of derivatives.
   */
  virtual VectorXd derivative(const VectorXd &x) const = 0;

  /**
   * @brief Destructor.
   */
  virtual ~ActivationFunction() {}
};

/**
 * @brief Sigmoid activation function.
 */
class SigmoidActivation : public ActivationFunction
{
  public:
  /**
   * @brief Activate function.
   * @param x Input vector.
   * @return Vector after activation.
   */
  VectorXd activate(const VectorXd &x) const override;

  /**
   * @brief Calculate derivative of sigmoid activation function.
   * @param x Input vector.
   * @return Vector of derivatives.
   */
  VectorXd derivative(const VectorXd &x) const override;
};

/**
 * @brief ReLU activation function.
 */
class ReLUActivation : public ActivationFunction
{
  public:
  /**
   * @brief Activate function.
   * @param x Input vector.
   * @return Vector after activation.
   */
  VectorXd activate(const VectorXd &x) const override;

  /**
   * @brief Calculate derivative of ReLU activation function.
   * @param x Input vector.
   * @return Vector of derivatives.
   */
  VectorXd derivative(const VectorXd &x) const override;
};