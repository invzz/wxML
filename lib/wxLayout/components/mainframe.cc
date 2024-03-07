/**
 * @file mainframe.cc
 * @brief Implementation of the application main frame.
 */
#include "mainframe.hh"

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

void fill_with_apple_data(std::vector<VectorXd> &input_data, std::vector<VectorXd> &output_data, std::vector<DataModel> &items)
{
  if(input_data.size() == 0 || output_data.size() == 0)
    {
      int num_samples  = items.size();
      int num_features = 7;
      int num_outputs  = 2;
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
  this->NN = new AndresNeuralNetwork(topology, this->LearningRate, this->Momentum, this->activationFunction);
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
  this->processing           = false;
  this->quitRequested        = false;
  int      rowSize           = 0;
  auto     margin            = FromDIP(10);
  auto     littleMargin      = FromDIP(3);
  auto     paramSizer        = new wxGridBagSizer(margin, margin);
  wxPanel *panel             = new wxPanel(this, wxID_ANY);
  logTxt                     = new wxTextCtrl(panel, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE | wxTE_READONLY);
  wxPanel    *btn_panel      = new wxPanel(panel, wxID_ANY);
  wxButton   *btn_train      = new wxButton(btn_panel, wxID_ANY, "train model", wxDefaultPosition, wxDefaultSize);
  wxButton   *btn_reset      = new wxButton(btn_panel, wxID_ANY, "reset weights", wxDefaultPosition, wxDefaultSize);
  wxButton   *btn_stop       = new wxButton(btn_panel, wxID_ANY, "stop current", wxDefaultPosition, wxDefaultSize);
  wxButton   *btn_reset_data = new wxButton(btn_panel, wxID_ANY, "reset input_data", wxDefaultPosition, wxDefaultSize);
  wxBoxSizer *btn_sizer      = new wxBoxSizer(wxHORIZONTAL);
  auto        Topology       = new wxPanel(panel, wxID_ANY);
  InputLayer                 = new wxSpinCtrl(Topology, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_NO_VSCROLL);
  HiddenLayer                = new wxSpinCtrl(Topology, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_NO_VSCROLL);
  HiddenLayerNumber          = new wxSpinCtrl(Topology, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_NO_VSCROLL);
  OutputLayer                = new wxSpinCtrl(Topology, wxID_ANY, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_NO_VSCROLL);
  auto TopologySizer         = new wxBoxSizer(wxHORIZONTAL);
  InputLayer->SetRange(7, 7);
  InputLayer->SetValue(7);
  HiddenLayer->SetRange(7, 128);
  HiddenLayer->SetValue(16);
  HiddenLayerNumber->SetRange(1, 32);
  HiddenLayerNumber->SetValue(1);
  OutputLayer->SetRange(1, 2);
  OutputLayer->SetValue(2);
  auto updateNeuralNetwork = [this]() {};
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
  btn_sizer->Add(btn_reset_data, 1, wxEXPAND | wxALL, margin);
  btn_train->Bind(wxEVT_BUTTON, &MainFrame::OnTrain, this);
  btn_reset->Bind(wxEVT_BUTTON, &MainFrame::OnReset, this);
  btn_stop->Bind(wxEVT_BUTTON, [this](wxCommandEvent &event) {
    this->stopRequested = true;
    wxLogMessage("Stop requested");
  });
  btn_reset_data->Bind(wxEVT_BUTTON, [this](wxCommandEvent &event) {
    this->input_data.clear();
    this->output_data.clear();
    appleList->items.clear();
    appleList->resetData();
    appleList->RefreshAfterUpdate();
  });
  btn_sizer->Add(btn_reset, 1, wxEXPAND | wxALL, margin);
  btn_sizer->Add(btn_train, 1, wxEXPAND | wxALL, margin);
  btn_sizer->Add(btn_stop, 1, wxEXPAND | wxALL, margin);
  btn_panel->SetSizer(btn_sizer);
  std::vector<gridctrl> paramPanelItems = {
    {{{rowSize++, 0}, {1, 1}}, btn_panel},
  };
  std::vector<std::pair<wxString, std::vector<int>>> Sliders = {
    {"Learning rate", {25, 0, 100}  },
    {"Momentum",      {15, 0, 100}  },
    {"Epochs",        {0, 0, 1000}  },
    {"Threshold",     {50, 0, 100}  },
    {"Batch Size",    {32, 16, 1024}},
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
      else if(slider.first == "Batch Size")
        wslider->Bind(wxEVT_SCROLL_CHANGED, [this](wxScrollEvent &event) {
          this->batch_size = event.GetPosition();
          wxLogMessage(wxString::Format("batch_size value :: %d", this->batch_size));
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
  int      rowSize                     = 0;
  auto     margin                      = FromDIP(10);
  wxPanel *panel                       = new wxPanel(this, wxID_ANY);
  auto     sizer                       = new wxGridBagSizer(margin, margin);
  progressBar                          = new wxGauge(panel, wxID_ANY, 100);
  appleList                            = new AppleListControl(panel, wxID_ANY, wxDefaultPosition, wxDefaultSize, "apple_qlty.csv");
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
  this->batch_size         = 32;
  this->activationFunction = new SigmoidActivation();
  this->NN                 = nullptr;
  auto mainSizer           = new wxBoxSizer(wxHORIZONTAL);
  mainSizer->Add(this->ParamPanel(), 1, wxEXPAND | wxALL, 0);
  mainSizer->Add(this->DataPanel(), 3, wxEXPAND | wxALL, 0);
  this->SetSizerAndFit(mainSizer);
  this->SetMinSize(FromDIP(wxSize(800, 600)));
  this->OnUpdateNN();
  wxMenuBar  *menuBar      = new wxMenuBar;
  wxMenu     *fileMenu     = new wxMenu;
  wxMenuItem *openMenuItem = fileMenu->Append(wxID_OPEN);
  wxMenuItem *saveMenuItem = fileMenu->Append(wxID_SAVE);
  fileMenu->AppendSeparator();
  wxMenuItem *exitMenuItem = fileMenu->Append(wxID_EXIT);
  Connect(wxID_SAVE, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler(MainFrame::OnSave));
  Connect(wxID_OPEN, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler(MainFrame::OnOpen));
  Connect(wxID_EXIT, wxEVT_COMMAND_MENU_SELECTED, wxCommandEventHandler(MainFrame::OnClose));
  menuBar->Append(fileMenu, "&File");
  SetMenuBar(menuBar);
  this->stopRequested = false;
}

double compute_error(const std::vector<VectorXd> &predictions, const std::vector<VectorXd> &targets)
{
  double mse = 0.0;
  for(int i = 0; i < predictions.size(); ++i)
    {
      auto pred_cols = predictions[i].cols();
      auto pred_rows = predictions[i].rows();
      auto targ_cols = targets[i].cols();
      auto targ_rows = targets[i].rows();
      auto diff      = predictions[i] - targets[i];
      mse += diff.dot(diff);
    }
  return mse / predictions.size();
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
  int                   num_samples = inputs.size();
  bool                  epoch_mode  = this->Epochs > 0;
  std::vector<VectorXd> Predictions(num_samples);
  for(int epoch = 0; (epoch_mode && epoch < this->Epochs && !stopRequested) || (!epoch_mode && !stopRequested); ++epoch)
    {
      QUIT_ROUTINE();
      for(int batch_start = 0; batch_start < num_samples; batch_start += batch_size)
        {
          int                   batch_end = std::min(batch_start + batch_size, num_samples);
          std::vector<VectorXd> batchInputs(inputs.begin() + batch_start, inputs.begin() + batch_end);
          std::vector<VectorXd> batchTargets(targets.begin() + batch_start, targets.begin() + batch_end);
          std::vector<VectorXd> batchPredictions(batch_end - batch_start);
          for(int sample_idx = 0; sample_idx < batchInputs.size(); ++sample_idx)
            {
              NN->forwardPropagation(batchInputs[sample_idx]);
              auto prediction = NN->getResults();
              NN->backpropagation(batchTargets[sample_idx]);
              Predictions[batch_start + sample_idx]                           = prediction;
              this->appleList->items[batch_start + sample_idx].predictions[0] = prediction(0) > this->Threshold ? 1 : 0;
              this->appleList->items[batch_start + sample_idx].predictions[1] = prediction(1) > this->Threshold ? 1 : 0;
            }
        }
      double error    = compute_error(Predictions, targets);
      double accuracy = compute_accuracy(Predictions, targets, this->Threshold);
      wxGetApp().CallAfter([this, epoch, error, accuracy] {
        appleList->Refresh();
        wxLogMessage("Epoch %d, Error: %.4f, Accuracy: %.4f", epoch, error, accuracy);
      });
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
        this->train(this->input_data, this->output_data, this->batch_size);
        wxLogMessage("Training ended :: Thread ");
      };
      this->workerThread = std::thread(f);
    }
  wxLogMessage("Training ended");
  progressBar->Refresh();
}

void MainFrame::OnOpen(wxCommandEvent &event)
{
  wxFileDialog openFileDialog(this, "Open File", "", "", "All files (*.*)|*.*", wxFD_OPEN | wxFD_FILE_MUST_EXIST);
  if(openFileDialog.ShowModal() == wxID_CANCEL) return;
  wxString filePath = openFileDialog.GetPath();
  wxLogMessage("Selected file: %s", filePath);
  bool isOpen = this->NN->loadWeights(filePath.ToStdString());
  if(!isOpen) { wxLogMessage("Error: Unable to open file."); }
  else { wxLogMessage("File opened successfully."); }
  auto t = NN->getTopology();
  InputLayer->SetValue(t.at(0));

  HiddenLayer->SetValue(t.at(1));
  HiddenLayerNumber->SetValue(t.size() - 2);
  OutputLayer->SetValue(this->NN->getTopology()[t.size() - 1]);
}

void MainFrame::OnSave(wxCommandEvent &event)
{
  wxFileDialog saveFileDialog(this, "Save File", "", "", "All files (*.*)|*.*", wxFD_SAVE | wxFD_OVERWRITE_PROMPT);
  if(saveFileDialog.ShowModal() == wxID_CANCEL) return;
  wxString filePath = saveFileDialog.GetPath();
  this->NN->saveWeights(filePath.ToStdString());
  wxLogMessage("File saved to: %s", filePath);
}

void MainFrame::OnClose(wxCommandEvent &e)
{
  wxLogMessage("Exit menu item clicked.");
  Close(true);
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