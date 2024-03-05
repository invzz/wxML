#ifndef _H_neuron
#define _H_neuron
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <random>

typedef std::vector<double> input_t;
typedef std::vector<double> target_t;

class UniformRandomInt
{
  std::random_device                 _rd{};
  std::mt19937                       _gen{_rd()};
  std::uniform_int_distribution<int> _dist;

  public:
  UniformRandomInt() { set(1, 100); }
  UniformRandomInt(int low, int high) { set(low, high); }

  void set(int low, int high)
  {
    std::uniform_int_distribution<int>::param_type param(low, high);
    _dist.param(param);
  }

  int get() { return _dist(_gen); }
};

/// UniformRandomDouble class definition
class UniformRandomDouble
{
  std::random_device                     _rd{};
  std::mt19937                           _gen{_rd()};
  std::uniform_real_distribution<double> _dist;

  public:
  UniformRandomDouble() { set(0.0, 1.0); }
  UniformRandomDouble(double low, double high) { set(low, high); }

  void set(double low, double high)
  {
    std::uniform_real_distribution<double>::param_type param(low, high);
    _dist.param(param);
  }

  double get() { return _dist(_gen); }
};

class neuron;

typedef std::vector<neuron> Layer;

struct connection
{
  double weight;
  double deltaWeight;
}; // struct connection

class neuron
{
  public:
  neuron(unsigned int numOutputs, unsigned int myIndex);
  double        getOutputValue() const { return outputVal; }
  void          setOutput(double val) { outputVal = val; }
  void          feedForward(const Layer &prevLayer);
  void          calcOutputGradients(double targetVal);
  void          calcHiddenGradients(const Layer &nextLayer);
  void          updateInputWeights(Layer &prevLayer);
  static double eta;   // [0.0..1.0] overall net training rate
  static double alpha; // [0.0..n] multiplier of last weight change (momentum)
  static double randomWeight()
  { // Create a random number engine
    static std::mt19937_64 generator(std::random_device{}());

    // Define a distribution for the weights (e.g., uniform distribution between -1 and 1)
    std::uniform_real_distribution<double> distribution(-1.0, 1.0);

    // Generate a random weight
    return distribution(generator);
  }

  private:
  double                  outputVal;
  unsigned int            Id;
  double                  gradient;
  static double           transferFunction(double x);
  static double           transferFunctionDerivative(double x);
  double                  sumDOW(const Layer &nextLayer) const;
  std::vector<connection> outputWeights;
};

class net
{
  public:
  net(const std::vector<unsigned int> &topology);
  void feedForward(const std::vector<double> &inputVals);
  void backProp(const std::vector<double> &targetVals);
  void getResults(std::vector<double> &resultVals) const;
  void setEta(double value);
  void setAlpha(double value);

  private:
  std::vector<Layer> layers;
  double             error                        = 0;
  double             recentAverageError           = 0;
  double             recentAverageSmoothingFactor = 0;
};

#endif
