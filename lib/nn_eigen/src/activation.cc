/**
 * @file activation.cc
 * @author your name (you@domain.com)
 * @brief implementation of activation functions
 * @version 0.1
 * @date 2024-03-07
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include "activation.hh"

/**
 * @brief Activate function.
 * @param x Input vector.
 * @return Vector after activation.
 */
VectorXd SigmoidActivation::activate(const VectorXd &x) const  { return 1.0 / (1.0 + (-x.array()).exp()); }
/**
 * @brief Calculate derivative of sigmoid activation function.
 * @param x Input vector.
 * @return Vector of derivatives.
 */
VectorXd SigmoidActivation::derivative(const VectorXd &x) const  { return x.array() * (1.0 - x.array()); }

/**
 * @brief Activate function.
 * @param x Input vector.
 * @return Vector after activation.
 */
VectorXd ReLUActivation::activate(const VectorXd &x) const  { return x.array().max(0); }
/**
 * @brief Calculate derivative of ReLU activation function.
 * @param x Input vector.
 * @return Vector of derivatives.
 */
VectorXd ReLUActivation::derivative(const VectorXd &x) const  { return (x.array() > 0).cast<double>(); }
