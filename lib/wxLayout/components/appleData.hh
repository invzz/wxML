#ifndef ITEMDATA_HH
#define ITEMDATA_HH
#include <string>

struct AppleData
{
  int                 id;
  std::vector<double> inputs;
  std::vector<double> outputs;
  std::vector<double> predictions;
};

#endif