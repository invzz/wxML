
/**
 * @file appleListControl.cc
 * @brief Implementation of the appleListControl class.
 */

#include "csvListControl.hh"
#include "DataModel.hh"

/**
 * @brief Template specialization for getting the text of an item in the virtual list control.
 * @tparam AppleData The type of data stored in the list control.
 * @param index The index of the item.
 * @param column The column index of the item.
 * @return The text of the item at the specified index and column.
 */
template <> wxString VirtualListControl<DataModel>::OnGetItemText(long index, long column) const
{
  DataModel item = items[index];

  int input_idx      = 0;
  int output_idx     = 0;
  int prediction_idx = 0;

  switch(column)
    {
    case 0: return std::to_string(item.id);
    case 1: return std::to_string(item.inputs[input_idx++]);
    case 2: return std::to_string(item.inputs[input_idx++]);
    case 3: return std::to_string(item.inputs[input_idx++]);
    case 4: return std::to_string(item.inputs[input_idx++]);
    case 5: return std::to_string(item.inputs[input_idx++]);
    case 6: return std::to_string(item.inputs[input_idx++]);
    case 7: return std::to_string(item.inputs[input_idx++]);
    case 8: return std::to_string(item.outputs[output_idx++]);
    case 9: return std::to_string(item.outputs[output_idx++]);
    case 10: return std::to_string(item.predictions[prediction_idx++]);
    case 11: return std::to_string(item.predictions[prediction_idx++]);
    default: return "";
    }
  return "";
}

/**
 * @brief Template specialization for getting the index of the first selected item in the virtual list control.
 * @tparam AppleData The type of data stored in the list control.
 * @return The index of the first selected item, or -1 if no item is selected.
 */
template <> long VirtualListControl<DataModel>::GetFirstSelectedItem()
{
  long item = -1;
  item      = GetNextItem(item, wxLIST_NEXT_ALL, wxLIST_STATE_SELECTED);
  return item;
}

/**
 * @brief Template specialization for sorting the items in the virtual list control by a specific column.
 * @tparam AppleData The type of data stored in the list control.
 * @param column The column index to sort by.
 */
template <> void VirtualListControl<DataModel>::sortByColumn(int column)
{
  auto cmp = [](auto a, auto b, bool asc) { return asc ? a < b : a > b; };
  bool asc = this->sortAscending;
  std::sort(items.begin(), items.end(), [column, asc, cmp](auto a, auto b) {
    switch(column)
      {
      case 0: return cmp(a.id, b.id, asc);

      case 1:
      case 2:
      case 3:
      case 4:
      case 5:
      case 6:
      case 7: return cmp(a.inputs[column - 1], b.inputs[column - 1], asc);

      case 8:
      case 9: return cmp(a.outputs[column - 8], b.outputs[column - 8], asc);

      case 10:
      case 11: return cmp(a.predictions[column - 10], b.predictions[column - 10], asc);

      default: return false;
      }
  });
}

/**
 * @brief Template specialization for resetting the data in the virtual list control.
 * @tparam AppleData The type of data stored in the list control.
 */
template <> void VirtualListControl<DataModel>::resetData()
{
  items.clear();
  int rows = doc.GetRowCount();

  int num_of_inputs  = 7;
  int num_of_outputs = 1;

  // Iterate through each row in the CSV file
  for(int i = 1; i < rows; ++i)
    {
      int current_col = 0;

      DataModel item;

      item.id = doc.GetCell<int>(current_col++, i);

      int j = current_col;

      for(j = current_col; j < current_col + num_of_inputs; j++) { item.inputs.push_back(doc.GetCell<double>(j, i)); }

      current_col = j;

      for(j = current_col; j < current_col + num_of_outputs; j++) { item.outputs.push_back(doc.GetCell<std::string>(j, i).compare("good") == 0 ? 1 : 0); }

      current_col = j;

      for(j = current_col; j < current_col + num_of_outputs; j++) { item.predictions.push_back(0); }

      items.push_back(item);
    }

  RefreshAfterUpdate();
}