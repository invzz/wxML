#ifndef _H_LAYOUT_EX_H_
#define _H_LAYOUT_EX_H_
#include <wx/wx.h>
#include "main_frame.hh"

class MyApp : public wxApp
{
  public:
  MainFrame   *MainDataFrame;
  virtual bool OnInit();
};

DECLARE_APP(MyApp);

#endif