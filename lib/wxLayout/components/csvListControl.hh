#pragma once
#include <vector>
#include <wx/wx.h>
#include <wx/listctrl.h>
#include <fstream>
#include <rapidcsv.h>
template <typename T> class VirtualListControl : public wxListCtrl
{
  public:
  VirtualListControl(wxWindow *parent, wxWindowID id, const wxPoint &pos, const wxSize &size, std::string csv_path) : wxListCtrl(parent, id, pos, size, wxLC_REPORT | wxLC_VIRTUAL | wxLC_SINGLE_SEL | wxLC_HRULES /*| wxLC_VRULES | wxBORDER_SUNKEN*/)
  {
    std::string fullPath = RES_DIR "/" + csv_path;
    doc.Load(fullPath);
    auto headers = doc.GetColumnNames();
    int  col_id  = 0;
    for(auto header : headers)
      {
        this->AppendColumn(header);
        col_id++;
      }
    this->AppendColumn("Quality2");
    this->AppendColumn("Prediction");
    this->AppendColumn("Prediction2");

    resetData();
    for(int i = 0; i < col_id; i++) { this->SetColumnWidth(i, wxLIST_AUTOSIZE_USEHEADER); }
    this->Bind(wxEVT_LIST_COL_CLICK, [this](wxListEvent &event) {
      auto selected = this->GetFirstSelectedItem();
      // choose an indexer : example description
      auto selectedDesc = selected != -1 ? this->items[selected].id : 0;
      if(selected != -1) { this->SetItemState(selected, 0, wxLIST_STATE_SELECTED); }
      this->sortByColumn(event.GetColumn());
      if(selected != -1)
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
      if(selected != -1) { auto item = this->items[selected]; }
    });
  }
  void RefreshAfterUpdate()
  {
    this->SetItemCount(items.size());
    this->Refresh();
  }
  std::vector<T>     items;
  rapidcsv::Document doc;
  void               resetData();

  private:
  bool             sortAscending = true;
  virtual wxString OnGetItemText(long index, long column) const wxOVERRIDE;
  long             GetFirstSelectedItem();
  void             sortByColumn(int column);
};
