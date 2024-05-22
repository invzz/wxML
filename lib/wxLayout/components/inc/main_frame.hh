/**
 * @file main_frame.hh
 * @author Andres Coronado (andres.coronado@bss.group)
 * @brief
 * @version 0.1
 * @date 2024-03-07
 *
 * @copyright Copyright (c) 2024
 *
 */
#ifndef _H_DATA_FRAME_H_
#define _H_DATA_FRAME_H_
#define _CRT_SECURE_NO_WARNINGS

#include <wx/wx.h>
#include <wx/spinctrl.h>
#include <atomic>
#include <thread>
#include "datamodel.model.hh"
#include "csvlist.control.hh"
#include <Eigen/Dense>
#include <NN.hh>

typedef VirtualListControl<DataModel> DataListControl;

/**
 * @brief The MainFrame class represents the main frame of the application.
 *
 * It inherits from wxFrame and provides functionality for populating data, training a neural network,
 * and managing user interactions.
 */
class MainFrame : public wxFrame
{
  public:
  /**
   * @brief Constructs a MainFrame object.
   *
   * @param title The title of the main frame.
   * @param pos The initial position of the main frame.
   * @param size The initial size of the main frame.
   */
  MainFrame(const wxString &title, const wxPoint &pos, const wxSize &size);

  /**
   * @brief Destroys the MainFrame object.
   *
   * Cleans up any resources used by the MainFrame object.
   */
  ~MainFrame()
  {
    // wxLog::SetActiveTarget(nullptr);
    delete NN;
  };

  /**
   * @brief Populates the main frame with a specified number of items.
   *
   * @param howManyItems The number of items to populate.
   */
  void Populate(int howManyItems);

  private:
  AndresNeuralNetwork *NN;                /**< Pointer to the neural network object. */
  double               Epochs;            /**< The number of training epochs. */
  double               ExpectedAcc = 0.0; /**< The expected accuracy of the neural network. */
  double               LearningRate;      /**< The learning rate of the neural network. */
  double               Momentum;          /**< The momentum of the neural network. */
  double               Threshold;         /**< The threshold value for the neural network. */
  int                  batch_size = 32;   /**< The batch size for training. */
  int                  n_f        = 7;    /**< The number of features. */
  int                  n_o        = 1;    /**< The number of outputs. */

  bool              processing;    /**< Flag indicating if processing is in progress. */
  std::atomic<bool> quitRequested; /**< Flag indicating if a quit request has been made. */
  std::atomic<bool> stopRequested; /**< Flag indicating if a stop request has been made. */

  std::thread workerThread; /**< The worker thread for processing. */

  int InputLayerSize;   /**< The size of the input layer. */
  int HiddenLayerSize;  /**< The size of each hidden layer. */
  int HiddenLayerCount; /**< The number of hidden layers. */
  int OutputLayerSize;  /**< The size of the output layer. */

  wxLog *logger; /**< Pointer to the logger object. */

  std::vector<VectorXd> input_data;  /**< The input data for training. */
  std::vector<VectorXd> output_data; /**< The output data for training. */

  ActivationFunction *activationFunction; /**< Pointer to the activation function object. */

  /**
   * @brief Event handler for the close event.
   *
   * @param e The close event.
   */
  void OnClose(wxCloseEvent &e);

  /**
   * @brief Event handler for the close event.
   *
   * @param e The close event.
   */
  void OnClose(wxCommandEvent &e);

  /**
   * @brief Event handler for the open event.
   *
   * @param e The open event.
   */
  void OnOpen(wxCommandEvent &e);

  /**
   * @brief Event handler for the save event.
   *
   * @param e The save event.
   */
  void OnSave(wxCommandEvent &e);

  /**
   * @brief Event handler for the train event.
   *
   * @param event The train event.
   */
  void OnTrain(wxCommandEvent &event);

  /**
   * @brief Event handler for the reset event.
   *
   * @param event The reset event.
   */
  void OnReset(wxCommandEvent &event);

  /**
   * @brief Updates the neural network.
   */
  void fill_data_vec(std::vector<VectorXd> &input_data, std::vector<VectorXd> &output_data, std::vector<DataModel> &items);
  void OnUpdateNN();

  /**
   * @brief Appends a log message to the log text control.
   *
   * @param msg The log message to append.
   */
  void Log(const std::string &msg) { logTxt->AppendText(msg + "\n"); };

  /**
   * @brief Trains the neural network with the given inputs and targets.
   *
   * @param inputs The input data for training.
   * @param targets The target data for training.
   * @param batch_size The batch size for training.
   */
  void train(const std::vector<VectorXd> &inputs, const std::vector<VectorXd> &targets, int batch_size);

  /**
   * @brief Creates and returns the parameter panel.
   *
   * @return The parameter panel.
   */
  wxPanel *ParamPanel();

  /**
   * @brief Creates and returns the data panel.
   *
   * @return The data panel.
   */
  wxPanel *DataPanel();

  wxTextCtrl      *logTxt;      /**< Pointer to the log text control. */
  wxGauge         *progressBar; /**< Pointer to the progress bar control. */
  DataListControl *dataList;    /**< Pointer to the apple list control. */

  wxSpinCtrl *InputLayer;        /**< Pointer to the input layer spin control. */
  wxSpinCtrl *HiddenLayer;       /**< Pointer to the hidden layer spin control. */
  wxSpinCtrl *HiddenLayerNumber; /**< Pointer to the hidden layer number spin control. */
  wxSpinCtrl *OutputLayer;       /**< Pointer to the output layer spin control. */
};
#endif
