


#pragma once

#include <vector>
#include <wx/wx.h>
#include <wx/listctrl.h>
#include <fstream>
#include <rapidcsv.h>

/**
 * @brief A template class for a virtual list control that displays data from a CSV file.
 * 
 * This class inherits from wxListCtrl and provides functionality to display and manipulate data
 * from a CSV file in a virtual list control.
 * 
 * @tparam T The type of data to be displayed in the list control.
 */
template <typename T>
class VirtualListControl : public wxListCtrl
{
public:
  /**
   * @brief Constructs a VirtualListControl object.
   * 
   * @param parent The parent window.
   * @param id The window ID.
   * @param pos The position of the control.
   * @param size The size of the control.
   * @param csv_path The path to the CSV file.
   */
  VirtualListControl(wxWindow *parent, wxWindowID id, const wxPoint &pos, const wxSize &size, std::string csv_path)
      : wxListCtrl(parent, id, pos, size, wxLC_REPORT | wxLC_VIRTUAL | wxLC_SINGLE_SEL | wxLC_HRULES)
  {
    std::string fullPath = RES_DIR "/" + csv_path;
    doc.Load(fullPath);
    auto headers = doc.GetColumnNames();
    int col_id = 0;
    for (auto header : headers)
    {
      this->AppendColumn(header);
      col_id++;
    }
    this->AppendColumn("Quality2");
    this->AppendColumn("Prediction");
    this->AppendColumn("Prediction2");

    resetData();
    for (int i = 0; i < col_id; i++)
    {
      this->SetColumnWidth(i, wxLIST_AUTOSIZE_USEHEADER);
    }

    this->Bind(wxEVT_LIST_COL_CLICK, [this](wxListEvent &event) {
      auto selected = this->GetFirstSelectedItem();
      auto selectedDesc = selected != -1 ? this->items[selected].id : 0;
      if (selected != -1)
      {
        this->SetItemState(selected, 0, wxLIST_STATE_SELECTED);
      }
      this->sortByColumn(event.GetColumn());
      if (selected != -1)
      {
        auto indexToSelect = std::find_if(this->items.begin(), this->items.end(), [selectedDesc](auto item) { return item.id == selectedDesc; }) - this->items.begin();
        this->SetItemState(indexToSelect, wxLIST_STATE_SELECTED, wxLIST_STATE_SELECTED);
        this->EnsureVisible(indexToSelect);
      }
      this->sortAscending = !this->sortAscending;
      this->RefreshAfterUpdate();
    });

    this->Bind(wxEVT_LIST_ITEM_SELECTED, [this](wxListEvent &event) {
      auto selected = this->GetFirstSelectedItem();
      if (selected != -1)
      {
        auto item = this->items[selected];
      }
    });
  }

  /**
   * @brief Refreshes the list control after updating the data.
   */
  void RefreshAfterUpdate()
  {
    this->SetItemCount(items.size());
    this->Refresh();
  }

  std::vector<T> items;                ///< The items to be displayed in the list control.
  rapidcsv::Document doc;              ///< The CSV document.
  
  /**
   * @brief Resets the data in the list control.
   */
  void resetData();

private:
  bool sortAscending = true;           ///< Flag indicating whether the list control is sorted in ascending order.

  /**
   * @brief Gets the text to be displayed for a specific item and column.
   * 
   * @param index The index of the item.
   * @param column The index of the column.
   * @return The text to be displayed.
   */
  virtual wxString OnGetItemText(long index, long column) const wxOVERRIDE;

  /**
   * @brief Gets the index of the first selected item in the list control.
   * 
   * @return The index of the first selected item, or -1 if no item is selected.
   */
  long GetFirstSelectedItem();

  /**
   * @brief Sorts the list control by the specified column.
   * 
   * @param column The index of the column to sort by.
   */
  void sortByColumn(int column);
};
