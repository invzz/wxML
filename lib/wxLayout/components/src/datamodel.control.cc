
/**
 * @file datamode.control.cc
 * @brief Implementation of the appleListControl class.
 */

#include "csvlist.control.hh"
#include "datamodel.model.hh"

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

  int id_range_limit         = 1;
  int input_range_limit      = item.inputs.size() + id_range_limit;
  int output_range_limit     = item.outputs.size() + input_range_limit;
  int prediction_range_limit = item.predictions.size() + output_range_limit;

  if(column == 0) return std::to_string(item.id);
  if(column < input_range_limit) return std::to_string(item.inputs[column - id_range_limit]);
  if(column < output_range_limit) return std::to_string(item.outputs[column - input_range_limit]);
  if(column < prediction_range_limit) return std::to_string(item.predictions[column - output_range_limit]);
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

  // Iterate through each row in the CSV file
  for(int i = 1; i < rows; ++i)
    {
      int current_col = 0;

      DataModel item;

      item.id = doc.GetCell<int>(current_col++, i);

      int j = current_col;

      for(j = current_col; j < current_col + num_of_inputs; j++) { item.inputs.push_back(doc.GetCell<double>(j, i)); }

      current_col = j;

      for(j = current_col; j < current_col + num_of_outputs; j++) { item.outputs.push_back(doc.GetCell<int>(j, i)); }

      current_col = j;

      for(j = current_col; j < current_col + num_of_outputs; j++) { item.predictions.push_back(0); }

      items.push_back(item);
    }

  RefreshAfterUpdate();
}