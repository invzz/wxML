#include "mainframe.hh"
#include <wx/spinctrl.h>
#include <wx/gbsizer.h>
#include <wx/wx.h>
#include <thread>
#include <chrono>
#include <atomic>
#include <algorithm>
#include <random>
#include <numeric>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include "app.hh"
#include "macros.hh"

void fill_with_apple_data(std::vector<VectorXd> &input_data, std::vector<VectorXd> &output_data, std::vector<AppleData> &items)
{
  if(input_data.size() == 0 || output_data.size() == 0)
    {
      int num_samples = items.size();

      int num_features = 7; // Number of features in AppleData excluding id, outputs, and prediction
      int num_outputs  = 2; // Number of outputs in AppleData

      for(int i = 0; i < num_samples; ++i)
        {
          VectorXd input_vector(num_features);
          VectorXd output_vector(num_outputs);

          for(int j = 0; j < num_features; ++j) { input_vector(j) = items[i].inputs[j]; }

          output_vector(0) = items[i].outputs[0];
          output_vector(1) = items[i].outputs[0] == 1 ? 0 : 1;
          
          while(items[i].predictions.size() < 2) { items[i].predictions.push_back(0); }
          
          input_data.push_back(input_vector);
          output_data.push_back(output_vector);
          
        }
    }
}

void MainFrame::OnUpdateNN()
{
  std::vector<int> topology;
  if(this->NN != nullptr)
    {
      delete(this->NN);
      this->NN = nullptr;
    }
  topology.push_back(this->InputLayerSize);
  for(int i = 0; i < this->HiddenLayerCount; i++) { topology.push_back(this->HiddenLayerSize); }
  topology.push_back(this->OutputLayerSize);
  this->NN = new eig_nn(topology, this->LearningRate, this->Momentum, this->activationFunction);
  wxLogMessage(wxString::Format(":: RESET NN ::"));
  for(auto element : topology) { wxLogMessage(wxString::Format("topology :: %d", element)); }
}

struct gridctrl
{
  std::pair<wxGBPosition, wxGBSpan> span;
  wxWindow                         *widget;
};
wxPanel *MainFrame::ParamPanel()
{
  this->processing              = false;
  this->quitRequested           = false;
  int      rowSize              = 0;
  auto     margin               = FromDIP(10);
  auto     littleMargin         = FromDIP(3);
  auto     paramSizer           = new wxGridBagSizer(margin, margin);
  wxPanel *panel                = new wxPanel(this, wxID_ANY);
  logTxt                        = new wxTextCtrl(panel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE | wxTE_READONLY);
  wxPanel    *btn_panel         = new wxPanel(panel, wxID_ANY);
  wxButton   *btn_train         = new wxButton(btn_panel, wxID_ANY, "train model", wxDefaultPosition, wxDefaultSize);
  wxButton   *btn_reset         = new wxButton(btn_panel, wxID_ANY, "reset weights", wxDefaultPosition, wxDefaultSize);
  wxButton   *btn_stop          = new wxButton(btn_panel, wxID_ANY, "stop current", wxDefaultPosition, wxDefaultSize);
  wxButton   *btn_normalize     = new wxButton(btn_panel, wxID_ANY, "normalize input_data", wxDefaultPosition, wxDefaultSize);
  wxButton   *btn_reset_data    = new wxButton(btn_panel, wxID_ANY, "reset input_data", wxDefaultPosition, wxDefaultSize);
  wxBoxSizer *btn_sizer         = new wxBoxSizer(wxHORIZONTAL);
  auto        Topology          = new wxPanel(panel, wxID_ANY);
  auto        InputLayer        = new wxSpinCtrl(Topology, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_NO_VSCROLL);
  auto        HiddenLayer       = new wxSpinCtrl(Topology, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_NO_VSCROLL);
  auto        HiddenLayerNumber = new wxSpinCtrl(Topology, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_NO_VSCROLL);
  auto        OutputLayer       = new wxSpinCtrl(Topology, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_NO_VSCROLL);
  auto        TopologySizer     = new wxBoxSizer(wxHORIZONTAL);
  InputLayer->SetRange(7, 7);
  InputLayer->SetValue(7);
  HiddenLayer->SetRange(7, 128);
  HiddenLayer->SetValue(16);
  HiddenLayerNumber->SetRange(1, 32);
  HiddenLayerNumber->SetValue(1);
  OutputLayer->SetRange(1, 2);
  OutputLayer->SetValue(2);

  auto updateNeuralNetwork = [this]() {

  };

  InputLayer->Bind(wxEVT_SPINCTRL, [this, updateNeuralNetwork](wxSpinEvent &event) {
    this->InputLayerSize = event.GetInt();
    OnUpdateNN();
  });

  HiddenLayer->Bind(wxEVT_SPINCTRL, [this, updateNeuralNetwork](wxSpinEvent &event) {
    this->HiddenLayerSize = event.GetInt();
    OnUpdateNN();
  });

  HiddenLayerNumber->Bind(wxEVT_SPINCTRL, [this, updateNeuralNetwork](wxSpinEvent &event) {
    this->HiddenLayerCount = event.GetInt();
    OnUpdateNN();
  });

  OutputLayer->Bind(wxEVT_SPINCTRL, [this, updateNeuralNetwork](wxSpinEvent &event) {
    this->OutputLayerSize = event.GetInt();
    OnUpdateNN();
  });

  TopologySizer->Add(InputLayer, 1, wxEXPAND | wxALL, littleMargin);
  TopologySizer->Add(HiddenLayer, 1, wxEXPAND | wxALL, littleMargin);
  TopologySizer->Add(HiddenLayerNumber, 1, wxEXPAND | wxALL, littleMargin);
  TopologySizer->Add(OutputLayer, 1, wxEXPAND | wxALL, littleMargin);
  Topology->SetSizer(TopologySizer);

  btn_sizer->Add(btn_normalize, 1, wxEXPAND | wxALL, margin);
  btn_sizer->Add(btn_reset_data, 1, wxEXPAND | wxALL, margin);

  btn_train->Bind(wxEVT_BUTTON, &MainFrame::OnTrain, this);
  btn_reset->Bind(wxEVT_BUTTON, &MainFrame::OnReset, this);
  btn_stop->Bind(wxEVT_BUTTON, [this](wxCommandEvent &event) {
    this->stopRequested = true;
    wxLogMessage("Stop requested");
  });

  btn_normalize->Bind(wxEVT_BUTTON, [this](wxCommandEvent &event) { appleList->RefreshAfterUpdate(); });

  btn_reset_data->Bind(wxEVT_BUTTON, [this](wxCommandEvent &event) {
    this->input_data.clear();
    this->output_data.clear();
    appleList->items.clear();
    appleList->resetData();
    appleList->RefreshAfterUpdate();
  });

  btn_sizer->Add(btn_train, 1, wxEXPAND | wxALL, margin);
  btn_sizer->Add(btn_reset, 1, wxEXPAND | wxALL, margin);
  btn_sizer->Add(btn_stop, 1, wxEXPAND | wxALL, margin);
  btn_panel->SetSizer(btn_sizer);

  std::vector<std::pair<wxString, std::vector<int>>> Sliders = {
    {"Learning rate", {25, 0, 100}},
    {"Momentum",      {15, 0, 100}},
    {"Epochs",        {0, 0, 1000}},
    {"Threshold",     {50, 0, 100}},
  };

  std::vector<gridctrl> paramPanelItems = {
    {{{rowSize++, 0}, {1, 1}}, btn_panel},
  };
  for(auto &slider : Sliders)
    {
      auto label   = new wxStaticText(panel, wxID_ANY, slider.first, wxDefaultPosition, wxDefaultSize);
      auto wslider = new wxSlider(panel, wxID_ANY, slider.second[0], slider.second[1], slider.second[2], wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL | wxSL_LABELS);
      paramPanelItems.push_back({
        {{rowSize++, 0}, {1, 1}},
        label
      });
      paramPanelItems.push_back({
        {{rowSize++, 0}, {1, 1}},
        wslider
      });
      if(slider.first == "Learning rate")
        wslider->Bind(wxEVT_SCROLL_CHANGED, [this](wxScrollEvent &event) {
          double value = event.GetPosition() / 100.0;
          this->NN->setEta(value);
          wxLogMessage(wxString::Format("eta value :: %f", value));
        });
      else if(slider.first == "Momentum")
        wslider->Bind(wxEVT_SCROLL_CHANGED, [this](wxScrollEvent &event) {
          double value = event.GetPosition() / 100.0;
          this->NN->setAlpha(value);
          wxLogMessage(wxString::Format("alpha value :: %f", value));
        });
      else if(slider.first == "Epochs")
        wslider->Bind(wxEVT_SCROLL_CHANGED, [this](wxScrollEvent &event) {
          this->Epochs = event.GetPosition();
          wxLogMessage(wxString::Format("epoch value :: %f", this->Epochs));
        });
      else if(slider.first == "Threshold")
        wslider->Bind(wxEVT_SCROLL_CHANGED, [this](wxScrollEvent &event) {
          this->Threshold = event.GetPosition() / 100.0;
          wxLogMessage(wxString::Format("threshold value :: %f", this->Threshold));
        });
    }
  paramPanelItems.push_back({
    {{rowSize++, 0}, {1, 1}},
    Topology
  });
  paramPanelItems.push_back({
    {{rowSize++, 0}, {1, 1}},
    logTxt
  });
  logTxt->SetDefaultStyle(wxTextAttr(*wxRED));
  for(auto &item : paramPanelItems) { paramSizer->Add(item.widget, item.span.first, item.span.second, wxEXPAND | wxLEFT | wxRIGHT | wxBOTTOM, margin); }
  paramSizer->AddGrowableRow(rowSize - 1);
  paramSizer->AddGrowableCol(0);
  panel->SetSizer(paramSizer);
  auto loggerTxT = new wxLogTextCtrl(logTxt);
  wxLog::SetActiveTarget(loggerTxT);
  return panel;
}
wxPanel *MainFrame::DataPanel()
{
  int      rowSize = 0;
  auto     margin  = FromDIP(10);
  wxPanel *panel   = new wxPanel(this, wxID_ANY);

  auto sizer  = new wxGridBagSizer(margin, margin);
  progressBar = new wxGauge(panel, wxID_ANY, 100);
  appleList   = new AppleListControl(panel, wxID_ANY, wxDefaultPosition, wxDefaultSize, "apple_qlty.csv");

  std::vector<gridctrl> dataPanelItems = {
    {{{rowSize++, 0}, {1, 1}}, progressBar},
    {{{rowSize++, 0}, {1, 1}}, appleList  },
  };
  for(auto &item : dataPanelItems) { sizer->Add(item.widget, item.span.first, item.span.second, wxEXPAND | wxLEFT | wxRIGHT, margin); }
  panel->SetSizer(sizer);
  sizer->AddGrowableCol(0);
  sizer->AddGrowableRow(rowSize - 1);
  return panel;
}
MainFrame::MainFrame(const wxString &title, const wxPoint &pos, const wxSize &size) : wxFrame(NULL, wxID_ANY, title, pos, size)
{
  this->HiddenLayerCount   = 1;
  this->HiddenLayerSize    = 16;
  this->OutputLayerSize    = 2;
  this->Epochs             = 0;
  this->Threshold          = 0.5;
  this->InputLayerSize     = 7;
  this->ExpectedAcc        = 80.0;
  this->LearningRate       = 0.25;
  this->Momentum           = 0.15;
  this->activationFunction = new SigmoidActivation();
  this->NN                 = nullptr;
  this->OnUpdateNN();

  this->Bind(wxEVT_CLOSE_WINDOW, &MainFrame::OnClose, this);
  this->stopRequested = false;
  auto mainSizer      = new wxBoxSizer(wxHORIZONTAL);
  mainSizer->Add(this->ParamPanel(), 1, wxEXPAND | wxALL, 0);
  mainSizer->Add(this->DataPanel(), 3, wxEXPAND | wxALL, 0);
  this->SetSizerAndFit(mainSizer);
  this->SetMinSize(FromDIP(wxSize(800, 600)));
}

double compute_error(const std::vector<VectorXd> &predictions, const std::vector<VectorXd> &targets)
{
  // // Compute Mean Squared Error (MSE)
  double mse = 0.0;
  for(int i = 0; i < predictions.size(); ++i)
    {
      auto pred_cols = predictions[i].cols();
      auto pred_rows = predictions[i].rows();
      auto targ_cols = targets[i].cols();
      auto targ_rows = targets[i].rows();

      auto diff = predictions[i] - targets[i];

      mse += diff.dot(diff);
    }
  return mse / predictions.size(); // Average over samples
}
double compute_accuracy(const std::vector<VectorXd> &inputs, const std::vector<VectorXd> &targets, double threshold = 0.5)
{
  int correct_predictions = 0;
  for(int i = 0; i < inputs.size(); ++i)
    {
      if((inputs[i](0) >= threshold && targets[i](0) == 1) || (inputs[i](0) < threshold && targets[i](0) == 0)) { correct_predictions++; }
    }
  return static_cast<double>(correct_predictions) / inputs.size() * 100;
}

void MainFrame::train(const std::vector<VectorXd> &inputs, const std::vector<VectorXd> &targets, int batch_size = 32)
{
  this->processing = true;

  if(!NN)
    {
      this->processing = false;
      wxMessageBox("Neural network pointer is not valid.", "Error", wxICON_ERROR | wxOK);
      return;
    }

  // Initialize variables for mini-batch iteration
  int num_samples = inputs.size();
  
  bool epoch_mode = this->Epochs > 0;

  for(int epoch = 0; (epoch_mode && epoch < this->Epochs && !stopRequested) || (!epoch_mode && !stopRequested); ++epoch)
    {
      QUIT_ROUTINE();

      std::vector<VectorXd> Predictions(num_samples);
      // Iterate over each sample in the mini-batch
      for(int sample_idx = 0; sample_idx < inputs.size(); ++sample_idx)
        {
          // Perform forward propagation for the current sample
          NN->forward_propagation(inputs[sample_idx]);

          // get the results
          auto prediction = NN->getResults();

          // Perform backpropagation for the current sample
          NN->backpropagation(targets[sample_idx]);

          // Store the prediction for the current sample
          Predictions[sample_idx] = prediction;

          this->appleList->items[sample_idx].predictions[0] = prediction(0) > this->Threshold ? 1 : 0;
          this->appleList->items[sample_idx].predictions[1] = prediction(1) > this->Threshold ? 1 : 0;
        }

      // Log error and accuracy after processing the entire mini -
      double error = compute_error(Predictions, targets);

      double accuracy = compute_accuracy(Predictions, targets, this->Threshold);

      Predictions.clear();

      wxGetApp().CallAfter([this, epoch, error, accuracy] {
        appleList->Refresh();
        wxLogMessage("Epoch %d, Error: %.4f, Accuracy: %.4f", epoch, error, accuracy);
      });

      // Evaluate model on validation set
      // Adjust learning rate, momentum, etc., based on validation performance
    }
  wxGetApp().CallAfter([this] {
    progressBar->SetValue(0);
    this->stopRequested = false;
    this->processing    = false;
    this->workerThread.join();
  });
}
void MainFrame::OnReset(wxCommandEvent &event)
{
  progressBar->SetValue(0);
  OnUpdateNN();
}
void MainFrame::OnTrain(wxCommandEvent &event)
{
  // fill input_data and output_data with appleList items if they are empty
  fill_with_apple_data(this->input_data, this->output_data, appleList->items);

  if(this->input_data.size() == 0 || this->output_data.size() == 0)
    {
      wxMessageBox("No input_data to train the model.", "Error", wxICON_ERROR | wxOK);
      return;
    }

  if(!this->processing)
    {
      const auto f = [this] {
        wxLogMessage("Training started :: Thread ");
        wxGetApp().CallAfter([this] { this->Layout(); });
        this->train(this->input_data, this->output_data);
        wxLogMessage("Training ended :: Thread ");
      };
      this->workerThread = std::thread(f);
    }
  wxLogMessage("Training ended");
  progressBar->Refresh();
}
void MainFrame::OnClose(wxCloseEvent &e)
{
  if(this->processing)
    {
      e.Veto();
      this->quitRequested = true;
    }
  else { this->Destroy(); }
}