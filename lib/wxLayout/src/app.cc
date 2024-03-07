#include "app.hh"

wxIMPLEMENT_APP(MyApp);

bool MyApp::OnInit()
{
  MainDataFrame = new MainFrame("Model Tester", wxDefaultPosition, wxSize(800, 600));
  MainDataFrame->Show(true);
  return true;
}

