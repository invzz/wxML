/**
 * @file lstm.hh
 * @author andres coronado (invizuz@gmail.com)
 * @brief
 * @version 0.1
 * @date 2024-03-21
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef LSTM_H
#define LSTM_H

#include <Eigen/Dense>
#include <vector>

class LSTM
{
  public:
  std::vector<Eigen::MatrixXd> dW;
  std::vector<Eigen::MatrixXd> db;

  LSTM(const std::vector<int> &topology);
  ~LSTM();

  void forward(const Eigen::MatrixXd &inputs);
  void backward(const Eigen::MatrixXd &inputs, const Eigen::MatrixXd &targets);
  void train(const Eigen::MatrixXd &inputs, const Eigen::MatrixXd &targets, double learning_rate);
  void train(const Eigen::MatrixXd &inputs, const Eigen::MatrixXd &targets, double learning_rate, int num_epochs, int batch_size);

  Eigen::MatrixXd get_output()
  {
    // Assuming output is the last hidden state
    return hidden_state_.back();
  }

  void initialize_gradients()
  {
    dW.resize(topology_.size() - 1);
    db.resize(topology_.size() - 1);
    for(size_t i = 0; i < topology_.size() - 1; ++i)
      {
        dW[i] = Eigen::MatrixXd::Zero(W_[i].rows(), W_[i].cols());
        db[i] = Eigen::MatrixXd::Zero(b_[i].rows(), b_[i].cols());
      }
  }

  private:
  std::vector<int>             topology_;
  std::vector<Eigen::MatrixXd> W_; // Weight matrices
  std::vector<Eigen::MatrixXd> b_; // Bias matrices
  std::vector<Eigen::MatrixXd> cell_state_;
  std::vector<Eigen::MatrixXd> hidden_state_;
  // Add any other necessary private variables or methods
};

#endif // LSTM_H