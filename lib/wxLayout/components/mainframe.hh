#ifndef _H_DATA_FRAME_H_
#define _H_DATA_FRAME_H_
#define _CRT_SECURE_NO_WARNINGS

#include <wx/wx.h>
#include <atomic>
#include <thread>
#include "appleData.hh"
#include "csvListControl.hh"
#include "NN.hh"

typedef VirtualListControl<AppleData> AppleListControl;


class MainFrame : public wxFrame
{
  public:
  MainFrame(const wxString &title, const wxPoint &pos, const wxSize &size);
  ~MainFrame()
  {
    // wxLog::SetActiveTarget(nullptr);
    delete NN;
    
  };
  void Populate(int howManyItems);

  private:
  eig_nn *NN;
  double  Epochs;
  double  ExpectedAcc = 0.0;
  double  LearningRate;
  double  Momentum;
  double  Threshold;

  bool              processing;
  std::atomic<bool> quitRequested;
  std::atomic<bool> stopRequested;

  std::thread workerThread;

  int InputLayerSize;
  int HiddenLayerSize;
  int HiddenLayerCount;
  int OutputLayerSize;

  wxLog *logger;

  std::vector<VectorXd> input_data;
  std::vector<VectorXd> output_data;

  ActivationFunction *activationFunction;

  void OnClose(wxCloseEvent &e);
  void OnTrain(wxCommandEvent &event);
  void OnReset(wxCommandEvent &event);
  void OnUpdateNN();

  void Log(const std::string &msg)  { logTxt->AppendText(msg + "\n"); };
  void train(const std::vector<VectorXd> &inputs, const std::vector<VectorXd> &targets, int batch_size);

  wxPanel *ParamPanel();
  wxPanel *DataPanel();

  wxTextCtrl       *logTxt;
  wxGauge          *progressBar;
  AppleListControl *appleList;
};
#endif
