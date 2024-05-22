/**
 * @file datamodel.model.hh
 * @author Andres Coronado (andres.coronado@bss.group)
 * @brief Defines the DataModel used by the application.
 * @version 0.1
 * @date 2024-03-07
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef ITEMDATA_HH
#define ITEMDATA_HH
#include <string>
#include <vector>

struct DataModel
{
  int                 id;
  std::vector<double> inputs;
  std::vector<double> outputs;
  std::vector<double> predictions;
};

#endif