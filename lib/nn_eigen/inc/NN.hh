/**
 * @file NN.hh
 * @author Andres Coronado (andres.coronado@bss.group)
 * @brief Header of neural network
 * @version 0.1
 * @date 2024-03-07
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef NN_H
#define NN_H

#include <iostream>
#include <fstream>
#include <vector>
#include <Eigen/Dense>
#include "activation.hh"

#define ROW_SEPARATOR      "\n"
#define MATRIX_SEPARATOR   "END-MATRIX"
#define TOPOLOGY_SEPARATOR "TOPOLOGY"
#define COLUMN_SEPARATOR   ","

using namespace Eigen;
using namespace std;

/**
 * @brief The AndresNeuralNetwork class represents a neural network.
 */
class AndresNeuralNetwork
{
  private:
  vector<int>         topology;            /**< Topology of the neural network. */
  vector<MatrixXd>    weights;             /**< Weights of the neural network. */
  vector<MatrixXd>    prev_weight_update;  /**< Previous weight update for momentum. */
  vector<VectorXd>    biases;              /**< Biases of the neural network. */
  vector<VectorXd>    activations;         /**< Activations of the neural network. */
  vector<VectorXd>    deltas;              /**< Deltas of the neural network. */
  double              learning_rate;       /**< Learning rate of the neural network. */
  double              momentum;            /**< Momentum of the neural network. */
  ActivationFunction *activation_function; /**< Activation function of the neural network. */

  public:
  /**
   * @brief Constructor.
   * @param topology Topology of the neural network.
   * @param learning_rate Learning rate of the neural network.
   * @param momentum Momentum of the neural network.
   * @param activation_function Activation function of the neural network.
   */
  AndresNeuralNetwork(const vector<int> &topology, double learning_rate, double momentum, ActivationFunction *activation_function);

  /**
   * @brief Perform backpropagation in the neural network.
   * @param target Target vector for backpropagation.
   */
  void backpropagation(const VectorXd &target);

  /**
   * @brief Perform forward propagation in the neural network.
   * @param input Input vector.
   * @param log Optional logging function.
   */
  void forwardPropagation(const VectorXd &input, std::function<void(string)> log = nullptr);

  /**
   * @brief Get the results of the neural network.
   * @param log Optional logging function.
   * @return Vector of results.
   */
  Eigen::VectorXd getResults(std::function<void(string)> log = nullptr) const;

  /**
   * @brief Load weights from a file.
   * @param filename Name of the file to load weights from.
   * @return True if weights are successfully loaded, false otherwise.
   */
  bool loadWeights(const std::string &filename);

  /**
   * @brief Save weights to a file.
   * @param filename Name of the file to save weights to.
   */
  void saveWeights(const std::string &filename) const;

  /**
   * @brief Set the learning rate of the neural network.
   * @param eta Learning rate value.
   */
  void setEta(double eta);

  /**
   * @brief Set the momentum of the neural network.
   * @param alpha Momentum value.
   */
  void setAlpha(double alpha);

  /**
   * @brief Set the topology of the neural network.
   * @param newTopology New topology to set.
   */
  void setTopology(const vector<int> &newTopology);

  /**
   * @brief Get the current topology of the neural network.
   * @return Vector representing the topology.
   */
  std::vector<int> getTopology() const;

  /**
   * @brief Destructor.
   */
  ~AndresNeuralNetwork();
};

#endif /* NN_H */