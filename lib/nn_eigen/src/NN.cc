/**
 * @file NN.cc
 * @author Andres Coronado (andres.coronado@bss.group)
 * @brief implementation of neural network
 * @version 0.1
 * @date 2024-03-07
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "NN.hh"

AndresNeuralNetwork::AndresNeuralNetwork(const vector<int> &topology, double learning_rate, double momentum, ActivationFunction *activation_function) : topology(topology), learning_rate(learning_rate), momentum(momentum), activation_function(activation_function) { setTopology(topology); }

AndresNeuralNetwork::~AndresNeuralNetwork() {}

void AndresNeuralNetwork::backpropagation(const VectorXd &target)
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
      weights[i] -= weight_update + momentum * prev_weight_update[i];
      biases[i] -= learning_rate * deltas[i];
      prev_weight_update[i] = weight_update;
    }
}

VectorXd AndresNeuralNetwork::getResults(std::function<void(string)> log) const
{
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
  else { throw std::logic_error("Output layer activations not computed"); }
}

std::vector<int> AndresNeuralNetwork::getTopology() const { return topology; }

void AndresNeuralNetwork::forwardPropagation(const VectorXd &input, std::function<void(string)> log)
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

bool AndresNeuralNetwork::loadWeights(const std::string &filename)
{
  std::ifstream file(filename);
  if(!file.is_open())
    {
      std::cerr << "Error opening file: " << filename << std::endl;
      return false;
    }

  std::vector<Eigen::MatrixXd> loadedWeights;
  std::string                  line;
  int                          weightIndex = 0;
  std::vector<double>          line_data;

  while(getline(file, line))
    {
      if(line == TOPOLOGY_SEPARATOR)
        {
          topology.clear();
          while(getline(file, line))
            {
              if(line == TOPOLOGY_SEPARATOR) { break; }
              topology.push_back(std::stoi(line));
            }
          setTopology(topology);
          continue;
        }

      if(line == MATRIX_SEPARATOR)
        {
          if(!line_data.empty())
            {
              int             rownum = weights[weightIndex].rows();
              int             colnum = weights[weightIndex].cols();
              Eigen::MatrixXd weight(rownum, colnum);
              for(int i = 0; i < rownum; ++i)
                for(int j = 0; j < colnum; ++j)
                  {
                    if(i * colnum + j < line_data.size())
                      weight(i, j) = line_data[i * colnum + j];
                    else
                      {
                        std::cerr << "Error loading weights from file: " << filename << std::endl;
                        return false;
                      }
                  }
              loadedWeights.push_back(weight);
              line_data.clear();
            }
          ++weightIndex;
        }

      else
        {
          std::stringstream ss(line);
          std::string       value;

          while(getline(ss, value, COLUMN_SEPARATOR[0])) { line_data.push_back(std::stod(value)); }
        }
    }

  file.close();

  weights = loadedWeights;
  return true;
}

void AndresNeuralNetwork::saveWeights(const std::string &filename) const
{
  const static IOFormat CSVFormat(FullPrecision, DontAlignCols, COLUMN_SEPARATOR, "\n");

  ofstream file(filename);

  if(!file.is_open())
    {
      cerr << "Error opening file: " << filename << endl;
      return;
    }

  file << TOPOLOGY_SEPARATOR << "\n";
  for(const auto &layer : topology) { file << layer << "\n"; }
  file << TOPOLOGY_SEPARATOR << "\n";
  for(const auto &weight : weights) { file << weight.format(CSVFormat) << "\n" MATRIX_SEPARATOR "\n"; }

  file.close();
}

void AndresNeuralNetwork::setAlpha(double alpha) { momentum = alpha; }

void AndresNeuralNetwork::setEta(double eta) { learning_rate = eta; }

void AndresNeuralNetwork::setTopology(const vector<int> &newTopology)
{
  topology = newTopology;
  weights.clear();
  prev_weight_update.clear();
  biases.clear();
  deltas.clear();
  activations.clear();

  for(int i = 0; i < topology.size() - 1; ++i)
    {
      weights.push_back(MatrixXd::Random(topology[i + 1], topology[i]));
      prev_weight_update.push_back(MatrixXd::Zero(topology[i + 1], topology[i]));
      biases.push_back(VectorXd::Random(topology[i + 1]));
    }
}
