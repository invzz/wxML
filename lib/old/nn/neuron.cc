
#include "neuron.hh"

double neuron::eta   = 0.15; // overall net learning rate
double neuron::alpha = 0.5;  // momentum, multiplier of last deltaWeight, [0.0..n]
// Returns random integer in closed range [low, high].

neuron::neuron(unsigned int numOutputs, unsigned int myIndex) : Id(myIndex)
{
  for(unsigned int c = 0; c < numOutputs; ++c)
    {
      UniformRandomDouble randomDoubleGenerator(-1.0, 1.0);
      double              weight      = randomDoubleGenerator.get() / sqrt(numOutputs);
      double              deltaWeight = 0.0; // Initialize delta weight to 0
      outputWeights.emplace_back(connection{weight, deltaWeight});
    }
}

void neuron::feedForward(const Layer &prevLayer)
{
  double sum = 0.0;
  for(unsigned int n = 0; n < prevLayer.size(); ++n) { sum += prevLayer[n].outputVal * prevLayer[n].outputWeights[Id].weight; }
  outputVal = neuron::transferFunction(sum);
}

double neuron::transferFunction(double x)
{
  return 1 / (1 + std::exp(-x));
  // return (x > 0) ? x : 0;
  //  std::tanh(x);
}

double neuron::transferFunctionDerivative(double x)
{
  // double y = transferFunction(x);
  // return 1 - y * y;
  // return (x > 0) ? 1 : 0;
  // return 1 / (1 + std::exp(-x));
  double sig_x = transferFunction(x);
  return sig_x * (1 - sig_x);
}

void neuron::calcOutputGradients(double targetVal)
{
  double delta = targetVal - outputVal;
  gradient     = delta * neuron::transferFunctionDerivative(outputVal);
}

void neuron::calcHiddenGradients(const Layer &nextLayer)
{
  double dow = sumDOW(nextLayer);
  gradient   = dow * neuron::transferFunctionDerivative(outputVal);
}

double neuron::sumDOW(const Layer &nextLayer) const
{
  double sum = 0.0;
  for(unsigned int n = 0; n < nextLayer.size() - 1; ++n) { sum += outputWeights[n].weight * nextLayer[n].gradient; }
  return sum;
}

void neuron::updateInputWeights(Layer &prevLayer)
{
  for(unsigned int n = 0; n < prevLayer.size(); ++n)
    {
      neuron &neuron                       = prevLayer[n];
      double  oldDeltaWeight               = neuron.outputWeights[Id].deltaWeight;
      double  newDeltaWeight               = (eta * neuron.outputVal * gradient) + (alpha * oldDeltaWeight);
      neuron.outputWeights[Id].deltaWeight = newDeltaWeight;
      neuron.outputWeights[Id].weight += newDeltaWeight;
    }
}

net::net(const std::vector<unsigned int> &topology)
{
  unsigned int numLayers = topology.size();
  for(unsigned int layerNum = 0; layerNum < numLayers; ++layerNum)
    {
      layers.push_back(Layer());

      unsigned int numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

      for(unsigned int neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum)
        {
          layers.back().push_back(neuron(numOutputs, neuronNum));
          std::cout << "Made a neuron!" << std::endl;
        }
      layers.back().back().setOutput(1.0);
    }
}

void net::feedForward(const std::vector<double> &inputs)
{
  for(unsigned int i = 0; i < inputs.size(); ++i) { layers[0][i].setOutput(inputs[i]); }

  for(unsigned int layerNum = 1; layerNum < layers.size(); ++layerNum)
    {
      Layer &prevLayer = layers[layerNum - 1];
      for(unsigned int n = 0; n < layers[layerNum].size() - 1; ++n) { layers[layerNum][n].feedForward(prevLayer); }
    }
}

void net::backProp(const std::vector<double> &targets)
{
  Layer &outputLayer = layers.back();

  // Calculate overall net error. (Root mean square (RMS) of output neuron errors).
  error = 0.0;
  for(unsigned int n = 0; n < outputLayer.size() - 1; ++n)
    {
      double delta = targets[n] - outputLayer[n].getOutputValue();
      error += delta * delta;
    }
  error /= outputLayer.size() - 1;
  error = sqrt(error);

  // Implement a recent average measurement.
  recentAverageError = (recentAverageError * recentAverageSmoothingFactor + error) / (recentAverageSmoothingFactor + 1.0);

  // Calculate output layer gradient.
  for(unsigned int n = 0; n < outputLayer.size() - 1; ++n) { outputLayer[n].calcOutputGradients(targets[n]); }
  for(unsigned int layerNum = 0; layerNum < layers.size() - 2; --layerNum)
    {
      Layer &hiddenLayer = layers[layerNum];
      Layer &nextLayer   = layers[layerNum + 1];
      for(unsigned int n = 0; n < hiddenLayer.size(); ++n) { hiddenLayer[n].calcHiddenGradients(nextLayer); }
    }

  // For all layers from output to first hidden layers. Update connection weights.
  for(unsigned int layerNum = layers.size() - 1; layerNum > 0; --layerNum)
    {
      Layer &layer_    = layers[layerNum];
      Layer &prevLayer = layers[layerNum - 1];
      for(unsigned int n = 0; n < layer_.size() - 1; ++n) { layer_[n].updateInputWeights(prevLayer); }
    }
}

void net::getResults(std::vector<double> &resultVals) const
{
  resultVals.clear();
  for(unsigned int n = 0; n < layers.back().size() - 1; ++n)
    {
      auto res = layers.back()[n].getOutputValue();
      resultVals.push_back(layers.back()[n].getOutputValue());
    }
}

void net::setEta(double value)
{
  for(Layer l : layers)
    {
      for(neuron n : l) { n.eta = value; }
    }
}

void net::setAlpha(double value)
{
  for(Layer l : layers)
    {
      for(neuron n : l) { n.alpha = value; }
    }
}
