#include "lstm.hh"
#include "activation.hh"
#include <vector>
#include <algorithm>
#include <iostream>

#define MSE

using namespace Eigen;

// Mean Squared Error (MSE) loss function
double compute_mse_loss(const MatrixXd &targets, const MatrixXd &predictions) { return (1.0 / targets.cols()) * (targets - predictions).array().square().sum(); }

// Binary Cross-Entropy loss function
double compute_binary_crossentropy_loss(const MatrixXd &targets, const MatrixXd &predictions) { return -(1.0 / targets.cols()) * (targets.array() * predictions.array().log() + (1 - targets.array()) * (1 - predictions.array()).log()).sum(); }

// Categorical Cross-Entropy loss function
double compute_categorical_crossentropy_loss(const MatrixXd &targets, const MatrixXd &predictions) { return -(1.0 / targets.cols()) * (targets.array() * predictions.array().log()).sum(); }

// Hinge loss function
double compute_hinge_loss(const MatrixXd &targets, const MatrixXd &predictions) { return (1.0 / targets.cols()) * (targets.array() * predictions.array()).sum(); }

double compute_accuracy(const MatrixXd &predictions, const MatrixXd &targets)
{
  int num_samples = predictions.cols();
  int num_correct = 0;

  for(int i = 0; i < num_samples; ++i)
    {
      // Find the index of the class with the highest probability
      int    predicted_class = 0;
      double max_probability = predictions.col(i).maxCoeff(&predicted_class);

      // Compare the predicted class with the true class
      int true_class = targets.col(i).maxCoeff();

      if(predicted_class == true_class) { num_correct++; }
    }

  // Compute and return accuracy
  return static_cast<double>(num_correct) / num_samples;
}

// Alias for compute_loss based on selected loss function
#ifdef MSE
#define compute_loss compute_mse_loss
#elif defined(BINARY_CROSSENTROPY)
#define compute_loss compute_binary_crossentropy_loss
#elif defined(CATEGORICAL_CROSSENTROPY)
#define compute_loss compute_categorical_crossentropy_loss
#elif defined(HINGE)
#define compute_loss compute_hinge_loss
#else
#error "Please define a loss function using preprocessor directives (e.g., -DMSE)"
#endif

LSTM::LSTM(const std::vector<int> &topology) : topology_(topology)
{
  W_.resize(topology_.size() - 1);
  b_.resize(topology_.size() - 1);

  for(size_t i = 0; i < topology_.size() - 1; ++i)
    {
      W_[i] = Eigen::MatrixXd::Random(topology_[i + 1], topology_[i]);
      b_[i] = Eigen::MatrixXd::Random(topology_[i + 1], 1);
    }
  initialize_gradients();
}

void LSTM::forward(const Eigen::MatrixXd &inputs)
{
  // Initialize cell and hidden states for the first timestep
  cell_state_.resize(topology_.size());
  hidden_state_.resize(topology_.size());
  cell_state_[0]   = Eigen::MatrixXd::Zero(topology_[0], 1);
  hidden_state_[0] = Eigen::MatrixXd::Zero(topology_[0], 1);

  // Forward pass through the LSTM layers
  for(size_t i = 0; i < topology_.size() - 1; ++i)
    {
      // Compute input to the current layer
      Eigen::MatrixXd layer_input;
      if(i == 0) { layer_input = inputs; }
      else { layer_input = hidden_state_[i - 1]; }

      // Compute input gate, forget gate, output gate, and candidate cell state for the current layer
      Eigen::MatrixXd input_gate           = ActivationFunction::sigmoid(W_[i] * layer_input + b_[i]);
      Eigen::MatrixXd forget_gate          = ActivationFunction::sigmoid(W_[i] * layer_input + b_[i + topology_.size() - 1]);
      Eigen::MatrixXd output_gate          = ActivationFunction::sigmoid(W_[i] * layer_input + b_[i + 2 * (topology_.size() - 1)]);
      Eigen::MatrixXd candidate_cell_state = ActivationFunction::tanh(W_[i] * layer_input + b_[i + 3 * (topology_.size() - 1)]);

      // Update cell state for the current layer
      cell_state_[i + 1] = input_gate.array() * candidate_cell_state.array() + forget_gate.array() * cell_state_[i].array();

      // Update hidden state for the current layer
      hidden_state_[i + 1] = output_gate.array() * ActivationFunction::tanh(cell_state_[i + 1]).array();
    }
}

void LSTM::backward(const Eigen::MatrixXd &inputs, const Eigen::MatrixXd &targets)
{
  // Initialize gradients

  Eigen::MatrixXd dcell_state   = Eigen::MatrixXd::Zero(topology_.back(), 1);
  Eigen::MatrixXd dhidden_state = Eigen::MatrixXd::Zero(topology_.back(), 1);
  Eigen::MatrixXd delta;

  // Perform backward pass through the LSTM layers
  for(int i = topology_.size() - 1; i > 0; --i)
    {
      // Compute gradients for output layer
      if(i == topology_.size() - 1) { delta = hidden_state_[i] - targets; }
      else { delta = W_[i + 1].transpose() * delta; }

      Eigen::MatrixXd dsigmoid_activation = ActivationFunction::sigmoid_derivative(hidden_state_[i]);
      Eigen::MatrixXd dhidden             = delta.array() * dsigmoid_activation.array();

      // Compute gradients for input gate
      Eigen::MatrixXd dsigmoid_input_gate = ActivationFunction::sigmoid_derivative(W_[i] * hidden_state_[i - 1] + b_[i]);
      Eigen::MatrixXd dinput_gate         = dhidden.array() * ActivationFunction::tanh(cell_state_[i]).array() * dsigmoid_input_gate.array();

      // Compute gradients for forget gate
      Eigen::MatrixXd dsigmoid_forget_gate = ActivationFunction::sigmoid_derivative(W_[i] * hidden_state_[i - 1] + b_[i]);
      Eigen::MatrixXd dforget_gate         = dhidden.array() * hidden_state_[i - 1].array() * dsigmoid_forget_gate.array();

      // Compute gradients for output gate
      Eigen::MatrixXd dsigmoid_output_gate = ActivationFunction::sigmoid_derivative(W_[i] * hidden_state_[i - 1] + b_[i]);
      Eigen::MatrixXd doutput_gate         = dhidden.array() * ActivationFunction::tanh(cell_state_[i]).array() * dsigmoid_output_gate.array();

      // Compute gradients for candidate cell state
      Eigen::MatrixXd dtanh_candidate_cell_state = ActivationFunction::tanh_derivative(W_[i] * hidden_state_[i - 1] + b_[i]);
      Eigen::MatrixXd dcell                      = dhidden.array() * ActivationFunction::sigmoid(W_[i] * hidden_state_[i - 1] + b_[i]).array() * dtanh_candidate_cell_state.array();

      // Update gradients
      dW[i] = dcell * hidden_state_[i - 1].transpose();
      db[i] = dcell;
      dcell_state += (dcell.array() * ActivationFunction::sigmoid(W_[i] * hidden_state_[i - 1] + b_[i]).array()).matrix();
      dhidden_state += ((W_[i] * dcell).array() * ActivationFunction::sigmoid(b_[i]).array()).matrix();

      // Propagate gradients to the previous timestep
      if(i > 1) { delta = dhidden_state; }
    }
}

void LSTM::train(const Eigen::MatrixXd &inputs, const Eigen::MatrixXd &targets, double learning_rate)
{
  // Training implementation
  forward(inputs);
  backward(inputs, targets);
  // Update weights and biases
  for(size_t i = 0; i < topology_.size() - 1; ++i)
    {
      W_[i] -= learning_rate * dW[i];
      b_[i] -= learning_rate * db[i];
    }
}

void LSTM::train(const Eigen::MatrixXd &inputs, const Eigen::MatrixXd &targets, double learning_rate, int num_epochs, int batch_size)
{
  int num_batches = inputs.cols() / batch_size;
  if(inputs.cols() % batch_size != 0) { num_batches++; }

  for(int epoch = 0; epoch < num_epochs; ++epoch)
    {
      double total_loss = 0.0;
      for(int batch = 0; batch < num_batches; ++batch)
        {
          int start_idx = batch * batch_size;
          int end_idx   = std::min((long long)(batch + 1) * batch_size, inputs.cols());

          MatrixXd batch_inputs  = inputs.middleCols(start_idx, end_idx - start_idx);
          MatrixXd batch_targets = targets.middleCols(start_idx, end_idx - start_idx);

          // Forward pass
          forward(batch_inputs);

          // Backward pass
          backward(batch_inputs, batch_targets);

          // Update weights and biases
          for(size_t i = 0; i < topology_.size() - 1; ++i)
            {
              W_[i] -= learning_rate * dW[i];
              b_[i] -= learning_rate * db[i];
            }

          // Compute and accumulate loss (you need to define a method to compute loss)
          total_loss += compute_loss(batch_targets, get_output());
        }

      // Calculate accuracy or any other metric
      double accuracy = compute_accuracy(inputs, targets);

      // Print epoch and accuracy
      std::cout << "Epoch " << epoch << ", Loss: " << total_loss << ", Accuracy: " << accuracy << std::endl;
    }
}
