#ifndef NN_H
#define NN_H

#include <iostream>
#include <vector>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

// Base class for activation functions
class ActivationFunction
{
  public:
  virtual VectorXd activate(const VectorXd &x) const   = 0;
  virtual VectorXd derivative(const VectorXd &x) const = 0;
  virtual ~ActivationFunction() {}
};

// Sigmoid activation function
class SigmoidActivation : public ActivationFunction
{
  public:
  VectorXd activate(const VectorXd &x) const override { return 1.0 / (1.0 + (-x.array()).exp()); }

  VectorXd derivative(const VectorXd &x) const override { return x.array() * (1.0 - x.array()); }
};

// ReLU activation function
class ReLUActivation : public ActivationFunction
{
  public:
  VectorXd activate(const VectorXd &x) const override { return x.array().max(0); }

  VectorXd derivative(const VectorXd &x) const override { return (x.array() > 0).cast<double>(); }
};

class eig_nn
{
  private:
  vector<int>         topology;
  vector<MatrixXd>    weights;
  vector<MatrixXd>    prev_weight_update; // Store previous weight update for momentum
  vector<VectorXd>    biases;
  vector<VectorXd>    activations;
  vector<VectorXd>    deltas;
  double              learning_rate;
  double              momentum;
  ActivationFunction *activation_function;

  public:
  void setEta(double eta) { learning_rate = eta; }
  void setAlpha(double alpha) { momentum = alpha; }

  eig_nn(const vector<int> &topology, double learning_rate, double momentum, ActivationFunction *activation_function) : topology(topology), learning_rate(learning_rate), momentum(momentum), activation_function(activation_function)
  {
    // Initialize weights and biases randomly
    for(int i = 0; i < topology.size() - 1; ++i)
      {
        weights.push_back(MatrixXd::Random(topology[i + 1], topology[i]));
        prev_weight_update.push_back(MatrixXd::Zero(topology[i + 1], topology[i])); // Initialize to zero for momentum
        biases.push_back(VectorXd::Random(topology[i + 1]));
      }
  }

  Eigen::VectorXd getResults(std::function<void(string)> log = nullptr) const
  {
    // Check if activations contain data and the size is greater than 1 (output layer exists)
    if(activations.size() > 1)
      {
        if(log != nullptr)
          {
            IOFormat     CleanFmt(4, 0, ", ", "\n", "[", "]");
            stringstream message;

            message << "OUTPUT: " << endl;
            message << activations.back().format(CleanFmt) << endl;

            log(message.str());
          }
        return activations.back();
      }
    else
      {
        // Handle error when activations are not computed
        throw std::logic_error("Output layer activations not computed");
      }
  }

  void forward_propagation(const VectorXd &input, std::function<void(string)> log = nullptr)
  {
    activations.clear();
    activations.push_back(input);

    if(log != nullptr)
      {
        IOFormat     CleanFmt(4, 0, ", ", "\n", "[", "]");
        stringstream message;

        message << "Input: " << endl;
        message << input.format(CleanFmt) << endl;

        log(message.str());
      }

    for(int i = 0; i < weights.size(); ++i)
      {
        VectorXd layer_output = weights[i] * activations.back() + biases[i];
        activations.push_back(activation_function->activate(layer_output));
      }
  }

  void backpropagation(const VectorXd &target)
  {
    deltas.clear();
    VectorXd output_error = activations.back() - target;
    VectorXd output_delta = output_error.array() * activation_function->derivative(activations.back()).array();
    deltas.push_back(output_delta);

    for(int i = weights.size() - 1; i > 0; --i)
      {
        VectorXd error = weights[i].transpose() * deltas.back();
        VectorXd delta = error.array() * activation_function->derivative(activations[i]).array();
        deltas.push_back(delta);
      }

    reverse(deltas.begin(), deltas.end());

    for(int i = 0; i < weights.size(); ++i)
      {
        MatrixXd weight_update = learning_rate * (deltas[i] * activations[i].transpose());
        weights[i] -= weight_update + momentum * prev_weight_update[i]; // Add momentum term
        biases[i] -= learning_rate * deltas[i];
        prev_weight_update[i] = weight_update; // Store current weight update for momentum
      }
  }

  ~eig_nn() {}
};

#endif /* NN_H */